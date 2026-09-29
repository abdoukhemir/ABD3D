import argparse

import torch

from config import DEFAULT_CONFIG
from training import train


def smoke_test(config) -> None:
    """Random-data forward/backward to catch shape errors before real training."""
    from training.train import ABD3DModel, build_optimizer

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = ABD3DModel(config).to(device)
    optimizer = build_optimizer(model, config)
    num_views = config.num_rings * config.views_per_ring
    model.train()
    for step in range(3):
        images = torch.randn(config.batch_size, 3, config.image_size,
                             config.image_size, device=device)
        in_ids = torch.randint(0, num_views, (config.batch_size,), device=device)
        tgt_ids = torch.randint(0, num_views,
                                (config.batch_size, config.num_target_views), device=device)
        out = model(images, in_ids, tgt_ids)
        loss = out["image"].float().square().mean() + out["alpha"].float().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        cam_grad = model.camera.log_radius.grad
        optimizer.step()
        print(f"step {step} | image {tuple(out['image'].shape)} "
              f"alpha {tuple(out['alpha'].shape)} | loss {loss.item():.4f} | "
              f"camera grad ok: {cam_grad is not None}")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("Smoke test passed ✅")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ABD3D single-image 3D generator.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overfit", type=int, default=None,
                        help="Train on only N objects (debug).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run a random-data smoke test and exit.")
    parser.add_argument("--require-multi-gpu", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.apply_memory_budget()
    if args.steps is not None:
        config.max_steps = args.steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.overfit is not None:
        config.overfit_num_objects = args.overfit
    if args.require_multi_gpu:
        config.require_multi_gpu = True

    print(f"[ABD3D] batch_size={config.batch_size} max_steps={config.max_steps}")

    if args.synthetic:
        smoke_test(config)
        return
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()