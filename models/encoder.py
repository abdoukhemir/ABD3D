import torch
from torch import nn
from transformers import AutoModel


class ViTEncoder(nn.Module):
    """Frozen DINOv2-small backbone + trainable projection to embed_dim."""

    def __init__(self, image_size=224, patch_size=16, embed_dim=448,
                 depth=8, heads=8, in_channels=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()      # keep the backbone frozen and deterministic
        return self

    def forward(self, images):
        x = ((images + 1) / 2 - self.mean) / self.std   # [-1,1] -> ImageNet normalization
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state   # [B, 257, 384]
        return self.proj(tokens)