import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    seed: int = 42

    # ---- model ----
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 16
    embed_dim: int = 448
    encoder_depth: int = 8
    encoder_heads: int = 8
    generator_depth: int = 6
    generator_heads: int = 8
    triplane_channels: int = 32
    triplane_size: int = 32          # final plane resolution
    triplane_token_size: int = 16    # token grid per plane (x2 upsampled -> triplane_size)
    decoder_hidden_dim: int = 128

    # ---- cameras / views (dataset: 48 views = 4 rings x 12 azimuths, ASSUMED) ----
    num_rings: int = 4
    views_per_ring: int = 12
    ring_elevations_deg: tuple = (0.0, 30.0, 60.0, -30.0)  # initial guess, learnable
    camera_radius: float = 2.0       # learnable
    camera_fov_deg: float = 40.0     # learnable
    scene_radius: float = 1.5        # ray near/far = radius -/+ scene_radius
    azimuth_direction: float = 1.0   # set to -1.0 if renders look mirrored

    # ---- rendering / supervision ----
    render_size: int = 64
    render_samples: int = 48
    num_target_views: int = 2        # target views rendered per input view
    same_view_prob: float = 0.1      # chance a target equals the input view

    # ---- data ----
    dataset_name: str = "zeyuanyin/complete-objaverse"
    val_objects: int = 8
    val_pairs_per_object: int = 2
    pool_size: int = 8               # objects mixed together at any time
    samples_per_object: int = 48     # samples drawn from an object before replacing it
    prefetch_objects: int = 4
    overfit_num_objects: int = 0     # >0: train on only this many objects (debug)

    # ---- optimisation ----
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_steps: int = 100_000         # optimizer updates
    learning_rate: float = 1e-4
    camera_lr_scale: float = 0.1
    weight_decay: float = 0.05
    warmup_steps: int = 1_000
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True

    # ---- loss ----
    mse_weight: float = 1.0
    alpha_weight: float = 1.0
    lpips_weight: float = 0.5
    lpips_warmup_steps: int = 1_000
    lpips_views: int = 4             # rendered views used for LPIPS per micro-batch
    fg_weight: float = 4.0           # extra MSE weight on object pixels

    # ---- logging / checkpoints ----
    log_every: int = 10
    val_interval_steps: int = 500
    run_name: str = "abd3d-triplane-v2"
    checkpoint_interval_minutes: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    backup_dir: str | None = "/kaggle/working/abd3d-checkpoints"
    require_multi_gpu: bool = False
    device: str = "cuda"

    @property
    def device_type(self) -> str:
        return "cuda" if self.device == "cuda" else "cpu"

    def apply_memory_budget(self) -> None:
        return None


DEFAULT_CONFIG = Config()