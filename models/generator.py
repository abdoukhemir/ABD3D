import torch
from torch import nn


class DiTGenerator(nn.Module):
    """Diffusion-style Transformer that maps image tokens to triplane tokens."""

    def __init__(self, conditioning_dim: int = 384, latent_dim: int = 256,
                 plane_channels: int = 32, plane_size: int = 32, depth: int = 8, heads: int = 8):
        super().__init__()
        self.plane_channels = plane_channels
        self.plane_size = plane_size
        self.token_dim = plane_channels * 3
        self.input = nn.Linear(conditioning_dim + latent_dim, conditioning_dim)
        layer = nn.TransformerEncoderLayer(conditioning_dim, heads, conditioning_dim * 4,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(conditioning_dim)
        self.output = nn.Linear(conditioning_dim, self.token_dim * plane_size * plane_size)

    def forward(self, image_tokens: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        pooled_latent = latent.mean(dim=(-2, -1))
        pooled_tokens = image_tokens.mean(dim=1)
        hidden = self.input(torch.cat((pooled_tokens, pooled_latent), dim=-1)).unsqueeze(1)
        hidden = self.norm(self.blocks(hidden)).squeeze(1)
        planes = self.output(hidden).view(-1, 3, self.plane_channels, self.plane_size, self.plane_size)
        return planes
