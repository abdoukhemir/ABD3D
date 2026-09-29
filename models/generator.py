import torch
from torch import nn


class DiTGenerator(nn.Module):
    """Transformer decoder: learnable triplane queries cross-attend to ALL image
    tokens (no pooling) and are conditioned on the input-view pose.
    (Feed-forward, not diffusion; the name is kept so imports keep working.)"""

    def __init__(self, conditioning_dim: int = 448, plane_channels: int = 32,
                 plane_size: int = 32, token_grid: int = 16,
                 depth: int = 6, heads: int = 8):
        super().__init__()
        if plane_size % token_grid:
            raise ValueError("plane_size must be a multiple of token_grid")
        self.plane_channels = plane_channels
        self.token_grid = token_grid
        self.upscale = plane_size // token_grid

        self.plane_tokens = nn.Parameter(
            torch.zeros(1, 3 * token_grid * token_grid, conditioning_dim))
        nn.init.trunc_normal_(self.plane_tokens, std=0.02)

        layer = nn.TransformerDecoderLayer(
            conditioning_dim, heads, conditioning_dim * 4,
            batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerDecoder(
            layer, depth, norm=nn.LayerNorm(conditioning_dim))
        self.output = nn.Linear(
            conditioning_dim, plane_channels * self.upscale * self.upscale)

    def forward(self, image_tokens: torch.Tensor,
                pose_embedding: torch.Tensor) -> torch.Tensor:
        batch = image_tokens.shape[0]
        g, u, c = self.token_grid, self.upscale, self.plane_channels
        memory = image_tokens + pose_embedding[:, None]
        queries = self.plane_tokens.expand(batch, -1, -1) + pose_embedding[:, None]
        hidden = self.blocks(queries, memory)
        out = self.output(hidden)                                # [B, 3*g*g, C*u*u]
        out = out.view(batch, 3, g, g, c, u, u)
        out = out.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, 3, c, g * u, g * u)
        return out