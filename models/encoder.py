import torch
from torch import nn


class ViTEncoder(nn.Module):
    """Compact Vision Transformer that converts an image into conditioning tokens."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 384,
                 depth: int = 6, heads: int = 6, in_channels: int = 3):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.pos_embed
        return self.norm(self.blocks(tokens))
