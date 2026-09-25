import io
import json
import os
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch import nn

from config import Config
from data.dataset import CompleteObjaverseDataset
from models import ImageVAE
from training.train import ABD3DModel, _lpips_loss, find_latest_step_checkpoint, save_step_checkpoint


RESULTS = {}


def mark_test(name: str, passed: bool):
    RESULTS[name] = passed


def test_imports():
    try:
        import data
        import models
        import training
        import config
        from data import CompleteObjaverseDataset
        from models import DiTGenerator, ImageVAE, TriplaneDecoder, ViTEncoder
        from training.train import ABD3DModel, train
        print("✅ All imports working")
        mark_test("imports", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Import test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("imports", False)
        return False


def test_model_shapes():
    try:
        config = Config(
            image_size=64,
            batch_size=4,
            num_views=12,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            vae_latent_dim=32,
            generator_depth=2,
            generator_heads=4,
            triplane_channels=8,
            triplane_size=8,
        )
        model = ABD3DModel(config)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        pred_shape = tuple(out["predicted_views"].shape)
        expected = (4, config.num_views - 1, 3, 64, 64)
        print(f"Input shape: {tuple(x.shape)}")
        print(f"Output shape: {pred_shape}")
        assert pred_shape == expected, f"Expected {expected}, got {pred_shape}"
        print("✅ Model shapes correct")
        mark_test("model_shapes", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Model shape test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("model_shapes", False)
        return False


def test_squeeze_fix():
    try:
        config = Config(
            image_size=64,
            batch_size=4,
            num_views=12,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            vae_latent_dim=32,
            generator_depth=2,
            generator_heads=4,
            triplane_channels=8,
            triplane_size=8,
        )
        model = ABD3DModel(config)
        input_view = torch.randn(4, 1, 3, 64, 64)
        squeezed = input_view.squeeze(1)
        assert list(squeezed.shape) == [4, 3, 64, 64], f"Unexpected squeezed shape: {tuple(squeezed.shape)}"
        out = model(squeezed)
        print(f"Squeezed input shape: {tuple(squeezed.shape)}")
        print(f"Output shape: {tuple(out['predicted_views'].shape)}")
        print("✅ Squeeze fix working")
        mark_test("squeeze_fix", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Squeeze fix test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("squeeze_fix", False)
        return False


def test_loss_function():
    try:
        pred = torch.randn(4, 11, 3, 64, 64, dtype=torch.float32)
        target = torch.randn(4, 11, 3, 64, 64, dtype=torch.float32)

        mse_loss = nn.functional.mse_loss(pred, target)
        lpips_metric = None
        try:
            import lpips
            lpips_metric = lpips.LPIPS(net="vgg")
            lpips_metric.eval()
            for param in lpips_metric.parameters():
                param.requires_grad_(False)
            lpips_loss = _lpips_loss(pred, target, lpips_metric)
        except Exception:
            lpips_loss = mse_loss

        mu = torch.randn(4, 8, 8, 8)
        logvar = torch.randn(4, 8, 8, 8)
        kl_loss = ImageVAE.kl_divergence(mu, logvar)
        total_loss = mse_loss + 0.1 * lpips_loss + 1e-5 * kl_loss

        if not torch.isfinite(total_loss):
            raise ValueError(f"Loss is not finite: {total_loss}")

        print(f"MSE: {mse_loss.item():.6f}")
        print(f"LPIPS: {lpips_loss.item():.6f}")
        print(f"KL: {kl_loss.item():.6f}")
        print(f"Total: {total_loss.item():.6f}")
        print("✅ Loss function working")
        mark_test("loss_function", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Loss function test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("loss_function", False)
        return False


def test_rolling_cache():
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id="zeyuanyin/complete-objaverse", repo_type="dataset")
        parquet_files = [p for p in files if p.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError("No parquet shards found in zeyuanyin/complete-objaverse")
        first_shard = parquet_files[0]
        print(f"First shard: {first_shard}")

        temp_dir = Path(tempfile.mkdtemp(prefix="abd3d_shard_"))
        local_path = Path(
            hf_hub_download(
                repo_id="zeyuanyin/complete-objaverse",
                repo_type="dataset",
                filename=first_shard,
                local_dir=str(temp_dir),
            )
        )
        assert local_path.exists(), f"Downloaded shard does not exist: {local_path}"

        df = pd.read_parquet(local_path)
        assert not df.empty, "Downloaded parquet shard is empty"
        row = df.iloc[0].to_dict()
        payload = row.get("image_png")
        if payload is None:
            raise KeyError("Downloaded shard does not contain an image_png column")
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        print(f"Image shape: {image.size}")
        assert image.size[0] > 0 and image.size[1] > 0, "Downloaded image has invalid size"

        local_path.unlink(missing_ok=True)
        assert not local_path.exists(), f"Shard was not deleted: {local_path}"
        print("✅ Rolling cache working")
        mark_test("rolling_cache", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Rolling cache test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("rolling_cache", False)
        return False


def test_training_loop():
    try:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        cfg = Config(
            image_size=64,
            batch_size=2,
            num_views=12,
            max_steps=3,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            vae_latent_dim=32,
            generator_depth=2,
            generator_heads=4,
            triplane_channels=8,
            triplane_size=8,
            checkpoint_dir=checkpoint_dir,
            mixed_precision=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ABD3DModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        losses = []
        for step in range(3):
            input_view = torch.randn(2, 3, 64, 64, device=device)
            target_views = torch.randn(2, cfg.num_views - 1, 3, 64, 64, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                output = model(input_view)
                loss = nn.functional.mse_loss(output["predicted_views"], target_views)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            save_step_checkpoint(checkpoint_dir, model, optimizer, scaler, step + 1, keep_last=3)
            print(f"Step {step + 1}/3 loss: {losses[-1]:.6f}")

        assert len(losses) == 3, f"Expected 3 losses, found {len(losses)}"
        assert all(torch.isfinite(torch.tensor(loss, dtype=torch.float32)) for loss in losses), "A loss became non-finite"

        latest = find_latest_step_checkpoint(checkpoint_dir)
        assert latest is not None, "No checkpoint was created"
        model2 = ABD3DModel(cfg).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        scaler2 = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        state = torch.load(latest, map_location=device)
        model2.load_state_dict(state["model"])
        optimizer2.load_state_dict(state["optimizer"])
        scaler2.load_state_dict(state.get("scaler", {}))
        print(f"Resume checkpoint: {latest}")
        print("✅ Full training loop working")
        mark_test("training_loop", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Full training loop test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("training_loop", False)
        return False


def main():
    test_imports()
    test_model_shapes()
    test_squeeze_fix()
    test_loss_function()
    test_rolling_cache()
    test_training_loop()

    print("\n═══════════════════════════")
    print("ABD3D PIPELINE TEST RESULTS")
    print("═══════════════════════════")
    print(f"TEST 1 - Imports:       {'✅' if RESULTS.get('imports') else '❌'}")
    print(f"TEST 2 - Model Shapes:  {'✅' if RESULTS.get('model_shapes') else '❌'}")
    print(f"TEST 3 - Squeeze Fix:   {'✅' if RESULTS.get('squeeze_fix') else '❌'}")
    print(f"TEST 4 - Loss Function: {'✅' if RESULTS.get('loss_function') else '❌'}")
    print(f"TEST 5 - Rolling Cache: {'✅' if RESULTS.get('rolling_cache') else '❌'}")
    print(f"TEST 6 - Training Loop: {'✅' if RESULTS.get('training_loop') else '❌'}")
    print("═══════════════════════════")
    passed = sum(1 for value in RESULTS.values() if value)
    print(f"{passed}/6 tests passed")


if __name__ == "__main__":
    main()
