import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from models import DiTGenerator, ImageVAE, TriplaneDecoder, ViTEncoder

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


class ABD3DModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_target_views = max(config.num_views - 1, 1)
        self.encoder = ViTEncoder(
            config.image_size, config.patch_size, config.embed_dim,
            config.encoder_depth, config.encoder_heads, config.in_channels)
        self.vae = ImageVAE(config.in_channels, config.vae_latent_dim)
        self.generator = DiTGenerator(
            config.embed_dim, config.vae_latent_dim,
            config.triplane_channels, config.triplane_size,
            config.generator_depth, config.generator_heads)
        self.decoder = TriplaneDecoder(config.triplane_channels)
        self.view_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 3, 1),
                nn.Sigmoid(),
            ) for _ in range(self.num_target_views)
        ])

    def forward(self, images: torch.Tensor) -> dict:
        _, latent, mu, logvar = self.vae(images)
        tokens = self.encoder(images)
        planes = self.generator(tokens, latent)
        output = self.decoder(planes)
        base_image = nn.functional.interpolate(
            output["image"], images.shape[-2:],
            mode="bilinear", align_corners=False)
        pred_views = torch.stack(
            [head(base_image) for head in self.view_heads], dim=1)
        output["image"] = base_image
        output["predicted_views"] = pred_views
        output.update({"mu": mu, "logvar": logvar})
        return output


def _lpips_loss(prediction, target, metric) -> torch.Tensor:
    if metric is None:
        return torch.zeros((), device=prediction.device)
    orig_device = prediction.device
    prediction = prediction.detach().cpu().reshape(-1, *prediction.shape[-3:])
    target = target.detach().cpu().reshape(-1, *target.shape[-3:])
    loss = metric.cpu()(
        prediction.mul(2).sub(1),
        target.mul(2).sub(1)
    ).mean()
    return loss.to(orig_device)


def save_checkpoint(path: Path, model, optimizer, scaler, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.module.state_dict() \
        if isinstance(model, nn.DataParallel) \
        else model.state_dict()
    torch.save({
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }, path)


def save_step_checkpoint(checkpoint_dir: Path, model, optimizer,
                         scaler, step: int, keep_last: int = 3) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step}.pt"
    save_checkpoint(path, model, optimizer, scaler, step)
    print(f"[ABD3D] Checkpoint saved: step_{step}.pt ✅")

    # Auto-backup to permanent storage (survives session end)
    permanent = Path("/kaggle/working/abd3d-checkpoints")
    if permanent.parent.exists():
        permanent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        shutil.copy(path, permanent / f"step_{step}.pt")

        # Save shard progress
        for f in [
            checkpoint_dir / "shard_progress.json",
            BASE_DIR / "checkpoints" / "shard_progress.json",
        ]:
            if f.exists():
                shutil.copy(f, permanent / "shard_progress.json")
                break

        # Save shard list
        # Save shard list
        for f in [
            checkpoint_dir / "shard_list.json",
            BASE_DIR / "checkpoints" / "shard_list.json",
        ]:
            if f.exists():
                shutil.copy(f, permanent / "shard_list.json")
                break

        # ✅ Backup VGG16 — never download again
        vgg = Path('/kaggle/working/torch_cache/hub/checkpoints/vgg16-397923af.pth')
        if vgg.exists() and not (permanent / 'vgg16-397923af.pth').exists():
            shutil.copy(vgg, permanent / 'vgg16-397923af.pth')
            print("[ABD3D] VGG16 backed up ✅")

        print(f"[ABD3D] Backed up to permanent storage ✅")

    # Keep only last N checkpoints locally
    files = sorted(checkpoint_dir.glob("step_*.pt"),
                   key=lambda p: p.stat().st_mtime)
    while len(files) > keep_last:
        files.pop(0).unlink(missing_ok=True)


def find_latest_step_checkpoint(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None
    files = sorted(checkpoint_dir.glob("step_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def get_fresh_dataloader(config: Config) -> DataLoader:
    from data import CompleteObjaverseDataset
    dataset = CompleteObjaverseDataset(
        name=config.dataset_name,
        split=config.dataset_split,
        image_size=config.image_size,
        config_name=config.dataset_config,
        image_keys=config.image_keys,
        num_workers=0,
        num_views=config.num_views,
        checkpoint_dir=config.checkpoint_dir,
    )
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=0)


def train(config: Config, resume=None) -> None:
    config.checkpoint_dir = resolve_checkpoint_dir(config.checkpoint_dir)
    print(f"[ABD3D] checkpoint_dir={config.checkpoint_dir}")
    torch.manual_seed(config.seed)

    # GPU Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"[ABD3D] GPU {i}: {name} ({mem:.1f}GB) ✅")
        print(f"[ABD3D] Total GPUs: {num_gpus} 🔥")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("[ABD3D] WARNING: No GPU — using CPU ⚠️")

    # Build model
    base_model = ABD3DModel(config).to(device)
    total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"[ABD3D] Model: {total_params:.1f}M parameters")

    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(base_model)
        print(f"[ABD3D] DataParallel across {num_gpus} GPUs 🔥")
    else:
        model = base_model
        print(f"[ABD3D] Single GPU")

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision and device.type == "cuda")

    # LPIPS — kept on CPU to avoid GPU 0 overload with DataParallel
    lpips_metric = None
    try:
        import lpips
        lpips_metric = lpips.LPIPS(net="vgg").eval()
        for p in lpips_metric.parameters():
            p.requires_grad_(False)
        print("[ABD3D] LPIPS loaded on CPU ✅")
    except ImportError:
        print("[ABD3D] LPIPS unavailable ⚠️")

    # Resume
    start_step = 0
    if resume:
        print(f"[ABD3D] Searching checkpoints in: {config.checkpoint_dir}")
        resume_path = find_latest_step_checkpoint(config.checkpoint_dir)
        if resume_path is None:
            final = config.checkpoint_dir / "final.pt"
            if final.exists():
                resume_path = str(final)
        if resume_path:
            print(f"[ABD3D] Loading: {resume_path}")
            state = torch.load(resume_path, map_location=device)
            base_model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state.get("scaler", {}))
            start_step = state.get("step", 0)
            print(f"[ABD3D] Resumed from step {start_step} ✅")
        else:
            print("[ABD3D] No checkpoint — fresh start")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint = time.monotonic()

    dataloader = get_fresh_dataloader(config)
    iterator = iter(dataloader)
    step = start_step

    print(f"[ABD3D] Training: step {step} → {config.max_steps} 🚀")

    while step < config.max_steps:

        try:
            batch = next(iterator)
        except StopIteration:
            print(f"[ABD3D] Shard done → next shard (step {step})")
            dataloader = get_fresh_dataloader(config)
            iterator = iter(dataloader)
            continue
        except Exception as e:
            print(f"[ABD3D] Error: {e} → reloading...")
            dataloader = get_fresh_dataloader(config)
            iterator = iter(dataloader)
            continue

        input_view   = batch["input_view"].to(device, non_blocking=True)
        target_views = batch["target_views"].to(device, non_blocking=True)

        if input_view.dim() == 5 and input_view.shape[1] == 1:
            input_view = input_view.squeeze(1)
        if target_views.dim() == 5 and target_views.shape[1] == 1:
            target_views = target_views.squeeze(1)

        with torch.autocast(device_type=device.type,
                            enabled=scaler.is_enabled()):
            output     = model(input_view)
            pred_views = output["predicted_views"]
            mse        = nn.functional.mse_loss(pred_views, target_views)
            perceptual = _lpips_loss(pred_views, target_views, lpips_metric)
            kl         = ImageVAE.kl_divergence(output["mu"], output["logvar"])
            loss       = (
                config.mse_weight  * mse +
                config.lpips_weight * perceptual +
                config.kl_weight   * kl
            ) / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                base_model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Checkpoint every 30 min
        if time.monotonic() - last_checkpoint >= \
                config.checkpoint_interval_minutes * 60:
            save_step_checkpoint(
                config.checkpoint_dir, model, optimizer, scaler, step + 1)
            last_checkpoint = time.monotonic()

        # Log
        gpu_mem = sum(
            torch.cuda.memory_reserved(i) / 1e9
            for i in range(num_gpus)
        ) if num_gpus > 0 else 0.0

        print(f"Step {step}/{config.max_steps - 1} | "
              f"Loss: {loss.item() * config.gradient_accumulation_steps:.4f} | "
              f"MSE: {mse.item():.4f} | "
              f"KL: {kl.item():.4f} | "
              f"GPU: {gpu_mem:.1f}GB")
        step += 1

    save_checkpoint(config.checkpoint_dir / "final.pt",
                    model, optimizer, scaler, config.max_steps)
    print("[ABD3D] Training complete! 🎉")