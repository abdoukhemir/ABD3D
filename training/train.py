import time
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from models import DiTGenerator, ImageVAE, TriplaneDecoder, ViTEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
print(f"[ABD3D] BASE_DIR={BASE_DIR}")


def resolve_checkpoint_dir(path: str | Path | None) -> Path:
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
        self.encoder = ViTEncoder(config.image_size, config.patch_size, config.embed_dim,
                                  config.encoder_depth, config.encoder_heads, config.in_channels)
        self.vae = ImageVAE(config.in_channels, config.vae_latent_dim)
        self.generator = DiTGenerator(config.embed_dim, config.vae_latent_dim,
                                      config.triplane_channels, config.triplane_size,
                                      config.generator_depth, config.generator_heads)
        self.decoder = TriplaneDecoder(config.triplane_channels)
        self.view_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
                nn.Conv2d(64, 3, 1), nn.Sigmoid(),
            ) for _ in range(self.num_target_views)
        ])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _, latent, mu, logvar = self.vae(images)
        tokens = self.encoder(images)
        planes = self.generator(tokens, latent)
        output = self.decoder(planes)
        base_image = nn.functional.interpolate(output["image"], images.shape[-2:], mode="bilinear", align_corners=False)
        pred_views = torch.stack([head(base_image) for head in self.view_heads], dim=1)
        output["image"] = base_image
        output["predicted_views"] = pred_views
        output.update({"mu": mu, "logvar": logvar})
        return output


def _lpips_loss(prediction: torch.Tensor, target: torch.Tensor, metric: nn.Module | None) -> torch.Tensor:
    if metric is None:
        return torch.zeros((), device=prediction.device)
    prediction = prediction.reshape(-1, *prediction.shape[-3:])
    target = target.reshape(-1, *target.shape[-3:])
    return metric(prediction.mul(2).sub(1), target.mul(2).sub(1)).mean()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "step": step}, path)


def save_step_checkpoint(checkpoint_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                        scaler: torch.amp.GradScaler, step: int, keep_last: int = 3) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{step}.pt"
    save_checkpoint(checkpoint_path, model, optimizer, scaler, step)
    print(f"Checkpoint saved at step {step}")

    files = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(files) > keep_last:
        oldest = files.pop(0)
        if oldest.exists():
            oldest.unlink()
        files = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)


def find_latest_step_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    step_files = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not step_files:
        return None
    return str(step_files[0])


def create_dataloader(config: Config) -> DataLoader:
    from data import CompleteObjaverseDataset
    dataset = CompleteObjaverseDataset(
        config.dataset_name,
        config.dataset_split,
        config.image_size,
        config.dataset_config,
        config.image_keys,
        num_workers=config.num_workers,
        num_views=config.num_views,
        checkpoint_dir=config.checkpoint_dir,
    )
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)


def train(config: Config, resume: str | None = None) -> None:
    config.checkpoint_dir = resolve_checkpoint_dir(config.checkpoint_dir)
    print(f"[ABD3D] training checkpoint_dir={config.checkpoint_dir}")
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataloader = create_dataloader(config)
    model = ABD3DModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    lpips_metric = None
    try:
        import lpips
        lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
        for parameter in lpips_metric.parameters():
            parameter.requires_grad_(False)
    except ImportError:
        print("LPIPS is unavailable; install the requirements to enable perceptual loss.")

    start_step = 0
    if resume:
        resume_path = resume if resume else str(find_latest_step_checkpoint(config.checkpoint_dir))
        if resume_path:
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state.get("scaler", {})); start_step = state.get("step", 0)
            print(f"Resumed from checkpoint: {resume_path} at step {start_step}")

    model.train(); last_checkpoint = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    iterator = iter(dataloader)
    step = start_step
    while step < config.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            print("Shard exhausted, reloading dataset...")
            dataloader = create_dataloader(config)
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration:
                print("Waiting for next shard to load...")
                time.sleep(5)
                iterator = iter(dataloader)
                continue

        input_view = batch["input_view"].to(device, non_blocking=True)
        target_views = batch["target_views"].to(device, non_blocking=True)

        if input_view.dim() > 4 and input_view.shape[1] == 1:
            input_view = input_view.squeeze(1)
        if target_views.dim() > 4 and target_views.shape[1] == 1:
            target_views = target_views.squeeze(1)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output = model(input_view)
            prediction_views = output["predicted_views"]
            reconstruction_loss = nn.functional.mse_loss(prediction_views, target_views)
            perceptual_loss = _lpips_loss(prediction_views, target_views, lpips_metric)
            kl_loss = ImageVAE.kl_divergence(output["mu"], output["logvar"])
            loss = (config.mse_weight * reconstruction_loss + config.lpips_weight * perceptual_loss +
                    config.kl_weight * kl_loss) / config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if time.monotonic() - last_checkpoint >= config.checkpoint_interval_minutes * 60:
            save_step_checkpoint(config.checkpoint_dir, model, optimizer, scaler, step + 1, keep_last=3)
            last_checkpoint = time.monotonic()
        display_step = step if step < config.max_steps else config.max_steps - 1
        print(f"Step {display_step}/{max(config.max_steps - 1, 0)} - Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
        step += 1
    save_checkpoint(config.checkpoint_dir / "final.pt", model, optimizer, scaler, config.max_steps)
    print("Training complete!")
