import torch
from torch import nn


class ImageVAE(nn.Module):
    """Small convolutional VAE used as the continuous latent interface."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, latent_dim, 4, 2, 1), nn.GroupNorm(16, latent_dim), nn.SiLU(),
        )
        self.mu = nn.Conv2d(latent_dim, latent_dim, 1)
        self.logvar = nn.Conv2d(latent_dim, latent_dim, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(images)
        mu, logvar = self.mu(hidden), self.logvar(hidden).clamp(-30, 20)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return latent, mu, logvar

    def decode(self, latent: torch.Tensor, output_size: int | None = None) -> torch.Tensor:
        reconstruction = self.decoder(latent)
        if output_size and reconstruction.shape[-1] != output_size:
            reconstruction = nn.functional.interpolate(reconstruction, (output_size, output_size), mode="bilinear", align_corners=False)
        return reconstruction

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mu, logvar = self.encode(images)
        return self.decode(latent, images.shape[-1]), latent, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).flatten(1).mean()
