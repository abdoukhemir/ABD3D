import argparse
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from training import train


def find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    candidate_files = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidate_files:
        return None
    return str(candidate_files[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ABD3D single-image 3D generator.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in checkpoints/.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate random synthetic images instead of loading the real dataset.")
    args = parser.parse_args()
    config = DEFAULT_CONFIG
    if args.steps is not None:
        config.max_steps = args.steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.apply_memory_budget()

    resume_path = None
    if args.resume:
        resume_path = find_latest_checkpoint(config.checkpoint_dir)
        if resume_path is None:
            print("No checkpoint found; starting fresh.")
        else:
            print(f"Resuming from latest checkpoint: {resume_path}")

    if args.synthetic:
        from training.train import ABD3DModel
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = ABD3DModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
        model.train()
        for step in range(config.max_steps):
            input_view = torch.randn(config.batch_size, config.in_channels, config.image_size, config.image_size, device=device)
            target_views = torch.randn(config.batch_size, config.num_views - 1, config.in_channels, config.image_size, config.image_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                output = model(input_view)
                loss = torch.nn.functional.mse_loss(output["predicted_views"], target_views)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            last_step = max(config.max_steps - 1, 0)
            print(f"Step {step}/{last_step} - Loss: {loss.item():.4f}")
        print("Training complete!")
        return

    train(config, resume_path)


if __name__ == "__main__":
    main()
