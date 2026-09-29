import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data.dataset import (CompleteObjaverseDataset, _load_or_fetch_shards,
                          build_fixed_batch, load_object, split_shards)
from models import DiTGenerator, TriplaneDecoder, ViTEncoder
from models.pose import OrbitCamera, PoseEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
print(f"[ABD3D] BASE_DIR={BASE_DIR}")


def resolve_checkpoint_dir(path) -> Path:
    if path is None:
        return CHECKPOINT_DIR
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate.resolve()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ABD3DModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.render_size = config.render_size
        self.render_samples = config.render_samples
        self.use_amp = bool(config.mixed_precision and torch.cuda.is_available())

        self.encoder = ViTEncoder(
            config.image_size, config.patch_size, config.embed_dim,
            config.encoder_depth, config.encoder_heads, config.in_channels)
        self.pose_encoder = PoseEncoder(
            config.num_rings, config.views_per_ring, config.embed_dim,
            config.azimuth_direction)
        self.generator = DiTGenerator(
            conditioning_dim=config.embed_dim,
            plane_channels=config.triplane_channels,
            plane_size=config.triplane_size,
            token_grid=config.triplane_token_size,
            depth=config.generator_depth,
            heads=config.generator_heads)
        self.decoder = TriplaneDecoder(
            config.triplane_channels, config.decoder_hidden_dim)
        self.camera = OrbitCamera(
            config.num_rings, config.views_per_ring, config.ring_elevations_deg,
            config.camera_radius, config.camera_fov_deg,
            config.azimuth_direction, config.scene_radius)

    def forward(self, images: torch.Tensor, input_view_ids: torch.Tensor,
                target_view_ids: torch.Tensor) -> dict:
        # autocast lives here (not in the train loop) so it also works under DataParallel
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            tokens = self.encoder(images)                        # [B, N, D]
            pose = self.pose_encoder(input_view_ids)             # [B, D]
            planes = self.generator(tokens, pose)                # [B, 3, C, H, W]
            origins, dirs, near, far = self.camera(
                target_view_ids, self.render_size)
            out = self.decoder.render(
                planes, origins, dirs, near, far,
                self.render_samples, jitter=self.training)
        return out


# --------------------------------------------------------------------------- #
# batch prep + losses
# --------------------------------------------------------------------------- #
def prepare_batch(batch: dict, device, render_size: int):
    inp = batch["input_image"].to(device, non_blocking=True).float() / 127.5 - 1.0
    tgt = batch["target_images"].to(device, non_blocking=True).float() / 127.5 - 1.0
    alp = batch["target_alpha"].to(device, non_blocking=True).float() / 255.0
    b, t = tgt.shape[:2]
    if tgt.shape[-1] != render_size:
        tgt = F.interpolate(tgt.reshape(b * t, *tgt.shape[2:]),
                            size=(render_size, render_size), mode="area")
        alp = F.interpolate(alp.reshape(b * t, *alp.shape[2:]),
                            size=(render_size, render_size), mode="area")
        tgt = tgt.reshape(b, t, 3, render_size, render_size)
        alp = alp.reshape(b, t, 1, render_size, render_size)
    return inp, tgt, alp


def lpips_loss(pred, target, metric, max_views: int) -> torch.Tensor:
    if metric is None:
        return pred.new_zeros(())
    h, w = pred.shape[-2:]
    p = pred.reshape(-1, 3, h, w)
    t = target.reshape(-1, 3, h, w)
    if p.shape[0] > max_views:
        idx = torch.randperm(p.shape[0], device=p.device)[:max_views]
        p, t = p[idx], t[idx]
    return metric(p, t.detach()).mean()          # both already in [-1, 1]; gradient flows to p


def compute_losses(output, tgt_rgb, tgt_alpha, config: Config, lpips_metric, step: int):
    pred = output["image"].float()
    pred_alpha = output["alpha"].float()

    weight = 1.0 + config.fg_weight * tgt_alpha
    sq = (pred - tgt_rgb).square().mean(dim=2, keepdim=True)
    mse = (weight * sq).sum() / weight.sum()
    alpha_loss = F.mse_loss(pred_alpha, tgt_alpha)
    lpips = lpips_loss(pred, tgt_rgb, lpips_metric, config.lpips_views)

    lpips_scale = min(1.0, step / max(1, config.lpips_warmup_steps))
    loss = (config.mse_weight * mse
            + config.alpha_weight * alpha_loss
            + config.lpips_weight * lpips_scale * lpips)
    parts = {
        "loss": loss.detach(),
        "mse": mse.detach(),
        "alpha": alpha_loss.detach(),
        "lpips": lpips.detach(),
        "plain_mse": F.mse_loss(pred, tgt_rgb).detach(),
    }
    return loss, parts


# --------------------------------------------------------------------------- #
# optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, config: Config):
    decay, no_decay, camera = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("camera."):
            camera.append(p)
        elif (p.ndim < 2 or "pos_embed" in name or "cls_token" in name
              or "plane_tokens" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": camera, "weight_decay": 0.0,
         "lr": config.learning_rate * config.camera_lr_scale},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, optimizer, scaler, scheduler, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, path)


def save_step_checkpoint(config: Config, model, optimizer, scaler, scheduler,
                         step: int, keep_last: int = 3) -> None:
    ckpt_dir = config.checkpoint_dir
    path = ckpt_dir / f"step_{step}.pt"
    save_checkpoint(path, model, optimizer, scaler, scheduler, step)
    print(f"[ABD3D] Checkpoint saved: step_{step}.pt ✅")

    if config.backup_dir:
        backup = Path(config.backup_dir)
        backup.mkdir(parents=True, exist_ok=True)
        new_backup = backup / f"step_{step}.pt"
        shutil.copy(path, new_backup)                 # copy first, delete old after
        for old in backup.glob("step_*.pt"):
            if old != new_backup:
                old.unlink(missing_ok=True)
        for fname in ("shard_progress.json", "shard_list.json"):
            src = BASE_DIR / "checkpoints" / fname
            if src.exists():
                shutil.copy(src, backup / fname)
        vgg = Path("/kaggle/working/torch_cache/hub/checkpoints/vgg16-397923af.pth")
        if vgg.exists() and not (backup / vgg.name).exists():
            shutil.copy(vgg, backup / vgg.name)
        print("[ABD3D] Backed up to permanent storage ✅")

    files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    while len(files) > keep_last:
        files.pop(0).unlink(missing_ok=True)


def find_latest_checkpoint(config: Config):
    candidates = []
    for d in (config.checkpoint_dir, Path(config.backup_dir) if config.backup_dir else None):
        if d is not None and d.exists():
            candidates += list(d.glob("step_*.pt"))
    if candidates:
        return max(candidates, key=lambda p: int(p.stem.split("_")[1]))
    final = config.checkpoint_dir / "final.pt"
    return final if final.exists() else None


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, val_batch, config: Config, device, lpips_metric, step: int):
    model.eval()
    total = val_batch["input_image"].shape[0]
    sums = {"mse": 0.0, "alpha": 0.0, "lpips": 0.0}
    count, first = 0, None
    for i in range(0, total, config.batch_size):
        chunk = {k: v[i:i + config.batch_size] for k, v in val_batch.items()}
        inp, tgt_rgb, tgt_alpha = prepare_batch(chunk, device, config.render_size)
        out = model(inp, chunk["input_view_id"].to(device),
                    chunk["target_view_ids"].to(device))
        pred, pred_alpha = out["image"].float(), out["alpha"].float()
        n = inp.shape[0]
        sums["mse"] += F.mse_loss(pred, tgt_rgb).item() * n
        sums["alpha"] += F.mse_loss(pred_alpha, tgt_alpha).item() * n
        if lpips_metric is not None:
            h, w = pred.shape[-2:]
            sums["lpips"] += lpips_metric(
                pred.reshape(-1, 3, h, w), tgt_rgb.reshape(-1, 3, h, w)).mean().item() * n
        count += n
        if first is None:
            first = (inp, tgt_rgb, tgt_alpha, pred, pred_alpha)
    model.train()

    metrics = {f"val_{k}": v / max(1, count) for k, v in sums.items()}
    metrics["val_psnr"] = 10.0 * math.log10(4.0 / max(metrics["val_mse"], 1e-8))

    image_path = None
    try:
        from torchvision.utils import save_image
        inp, tgt_rgb, tgt_alpha, pred, pred_alpha = first
        n = min(4, inp.shape[0])
        size = pred.shape[-1]
        inp_s = F.interpolate(inp[:n], size=(size, size), mode="area")
        rows = torch.cat((
            (inp_s + 1) / 2,
            (tgt_rgb[:n, 0] + 1) / 2,
            (pred[:n, 0] + 1) / 2,
            tgt_alpha[:n, 0].expand(-1, 3, -1, -1),
            pred_alpha[:n, 0].expand(-1, 3, -1, -1),
        ), dim=-1).clamp(0, 1)
        image_path = config.checkpoint_dir / "val_images" / f"step_{step:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(rows, str(image_path), nrow=1, padding=2)
    except Exception as exc:
        print(f"[ABD3D] Could not save validation image: {exc}")
    return metrics, image_path


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(config: Config, resume=None) -> None:
    config.checkpoint_dir = resolve_checkpoint_dir(config.checkpoint_dir)
    print(f"[ABD3D] checkpoint_dir={config.checkpoint_dir}")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"[ABD3D] GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB) ✅")
    else:
        device, num_gpus = torch.device("cpu"), 0
        print("[ABD3D] WARNING: No GPU — using CPU ⚠️")

    if config.require_multi_gpu and num_gpus < 2:
        raise RuntimeError(f"--require-multi-gpu set but only {num_gpus} GPU(s) found")

    base_model = ABD3DModel(config).to(device)
    print(f"[ABD3D] Model: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M parameters")
    if num_gpus > 1:
        model = nn.DataParallel(base_model)
        print(f"[ABD3D] DataParallel across {num_gpus} GPUs 🔥")
    else:
        model = base_model

    optimizer = build_optimizer(base_model, config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.mixed_precision and device.type == "cuda")
    warmup = max(1, config.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup))

    lpips_metric = None
    try:
        import lpips
        lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_metric.parameters():
            p.requires_grad_(False)
        print("[ABD3D] LPIPS loaded ✅")
    except ImportError:
        print("[ABD3D] LPIPS unavailable ⚠️ (LPIPS loss = 0)")

    use_wandb = False
    try:
        import wandb
        wandb.init(project="ABD3D", name=config.run_name, config={
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(config).items()})
        use_wandb = True
        print("[ABD3D] WandB initialized ✅")
    except Exception as exc:
        print(f"[ABD3D] WandB not available: {exc} ⚠️")

    # ---- resume ----
    start_step = 0
    if resume:
        resume_path = find_latest_checkpoint(config)
        if resume_path is None:
            print("[ABD3D] No checkpoint found — fresh start")
        else:
            print(f"[ABD3D] Loading: {resume_path}")
            state = torch.load(resume_path, map_location=device, weights_only=False)
            try:
                base_model.load_state_dict(state["model"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint does not match the new architecture (it is from the old "
                    "model). Start fresh without --resume, or delete old step_*.pt/final.pt."
                ) from exc
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = int(state.get("step", 0))
            if config.backup_dir:
                for fname in ("shard_progress.json", "shard_list.json"):
                    src, dst = Path(config.backup_dir) / fname, BASE_DIR / "checkpoints" / fname
                    if src.exists() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src, dst)
            print(f"[ABD3D] Resumed from step {start_step} ✅")

    # ---- data ----
    all_shards = _load_or_fetch_shards(config.dataset_name)
    train_shards, val_shards = split_shards(all_shards, config.seed, config.val_objects)

    preloaded, val_objects = None, []
    if config.overfit_num_objects > 0:
        print(f"[ABD3D] OVERFIT MODE: {config.overfit_num_objects} object(s)")
        preloaded = []
        for shard in train_shards[:config.overfit_num_objects]:
            obj = load_object(config.dataset_name, shard, config.image_size)
            if obj is not None:
                preloaded.append(obj)
        if not preloaded:
            raise RuntimeError("Could not load any object for overfit mode")
        val_objects = preloaded
    else:
        for shard in val_shards:
            try:
                obj = load_object(config.dataset_name, shard, config.image_size)
            except Exception as exc:
                print(f"[ABD3D] Validation shard failed: {exc}")
                obj = None
            if obj is not None:
                val_objects.append(obj)
    val_batch = (build_fixed_batch(val_objects, config.num_target_views,
                                   config.val_pairs_per_object)
                 if val_objects else None)
    print(f"[ABD3D] Validation objects: {len(val_objects)}")

    dataset = CompleteObjaverseDataset(
        name=config.dataset_name, shards=train_shards,
        image_size=config.image_size, num_target_views=config.num_target_views,
        pool_size=config.pool_size, samples_per_object=config.samples_per_object,
        prefetch=config.prefetch_objects, same_view_prob=config.same_view_prob,
        preloaded_objects=preloaded)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0,
                            pin_memory=(device.type == "cuda"))
    iterator = iter(dataloader)

    # ---- loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum = config.gradient_accumulation_steps
    step = start_step
    last_checkpoint = time.monotonic()
    last_log_time = time.monotonic()
    print(f"[ABD3D] Training: step {step} → {config.max_steps} 🚀 "
          f"(effective batch {config.batch_size * accum})")

    while step < config.max_steps:
        stats = {}
        for _ in range(accum):
            batch = next(iterator)
            inp, tgt_rgb, tgt_alpha = prepare_batch(batch, device, config.render_size)
            out = model(inp, batch["input_view_id"].to(device),
                        batch["target_view_ids"].to(device))
            loss, parts = compute_losses(out, tgt_rgb, tgt_alpha, config,
                                         lpips_metric, step)
            scaler.scale(loss / accum).backward()
            for k, v in parts.items():
                stats[k] = stats.get(k, 0.0) + v / accum

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        if step % config.log_every == 0 or step == 1:
            vals = {k: float(v) for k, v in stats.items()}
            if not math.isfinite(vals["loss"]):
                print("[ABD3D] ⚠️ Non-finite loss detected")
            psnr = 10.0 * math.log10(4.0 / max(vals["plain_mse"], 1e-8))
            lr = optimizer.param_groups[0]["lr"]
            now = time.monotonic()
            sec = (now - last_log_time) / (config.log_every if step != 1 else 1)
            last_log_time = now
            gpu_mem = sum(torch.cuda.memory_reserved(i) / 1e9
                          for i in range(num_gpus)) if num_gpus else 0.0
            print(f"Step {step}/{config.max_steps} | Loss: {vals['loss']:.4f} | "
                  f"MSE: {vals['mse']:.4f} | Alpha: {vals['alpha']:.4f} | "
                  f"LPIPS: {vals['lpips']:.4f} | PSNR: {psnr:.2f} | "
                  f"GradNorm: {float(grad_norm):.2f} | LR: {lr:.2e} | "
                  f"{sec:.2f}s/step | GPU: {gpu_mem:.1f}GB")
            if use_wandb:
                import wandb
                log = {"loss": vals["loss"], "mse": vals["mse"], "alpha_loss": vals["alpha"],
                       "lpips": vals["lpips"], "psnr": psnr, "grad_norm": float(grad_norm),
                       "learning_rate": lr, "gpu_gb": gpu_mem}
                log.update(base_model.camera.state())
                wandb.log(log, step=step)

        if val_batch is not None and step % config.val_interval_steps == 0:
            metrics, image_path = validate(model, val_batch, config, device,
                                           lpips_metric, step)
            print(f"[ABD3D] VAL step {step} | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            if use_wandb:
                import wandb
                log = dict(metrics)
                if image_path is not None:
                    log["val_images"] = wandb.Image(str(image_path))
                wandb.log(log, step=step)

        if time.monotonic() - last_checkpoint >= config.checkpoint_interval_minutes * 60:
            save_step_checkpoint(config, base_model, optimizer, scaler, scheduler, step)
            last_checkpoint = time.monotonic()

    save_checkpoint(config.checkpoint_dir / "final.pt",
                    base_model, optimizer, scaler, scheduler, step)
    if use_wandb:
        import wandb
        wandb.finish()
    print("[ABD3D] Training complete! 🎉")