import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int = 42
    image_size: int = 224
    in_channels: int = 3
    num_views: int = 12
    patch_size: int = 16
    embed_dim: int = 448
    encoder_depth: int = 8
    encoder_heads: int = 8
    vae_latent_dim: int = 256
    generator_depth: int = 9
    generator_heads: int = 8
    triplane_channels: int = 32
    triplane_size: int = 32
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_workers: int = 0 if os.name == "nt" else 2
    max_steps: int = 100_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    grad_clip_norm: float = 1.0
    mse_weight: float = 1.0
    lpips_weight: float = 0.1
    kl_weight: float = 1e-5
    checkpoint_interval_minutes: int = 30
    checkpoint_dir: Path = Path("checkpoints")
    dataset_name: str = "zeyuanyin/complete-objaverse"
    dataset_config: str | None = None
    dataset_split: str = "train"
    image_keys: tuple[str, ...] = ("image_png", "image", "render", "front_image")
    mixed_precision: bool = True
    device: str = "cuda"

    @property
    def device_type(self) -> str:
        return "cuda" if self.device == "cuda" else "cpu"

    def apply_memory_budget(self) -> None:
        if os.name == "nt":
            self.num_workers = 0
        if self.device_type == "cuda":
            self.batch_size = min(self.batch_size, 8)


DEFAULT_CONFIG = Config()
