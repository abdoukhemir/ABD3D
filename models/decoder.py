import torch
import torch.nn.functional as F
from torch import nn


class TriplaneDecoder(nn.Module):
    """Triplane -> density + RGB field, with a differentiable volume renderer."""

    def __init__(self, plane_channels: int = 32, hidden_dim: int = 128,
                 density_bias: float = -2.0, background: float = 1.0):
        super().__init__()
        self.density_bias = density_bias
        self.background = background
        self.field = nn.Sequential(
            nn.Linear(plane_channels * 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    @staticmethod
    def _sample_planes(planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """planes [B,3,C,H,W], points [B,N,3] in [-1,1] -> [B,N,3C]."""
        samples = []
        for plane, (a, b) in zip(planes.unbind(1), ((1, 2), (0, 2), (0, 1))):
            grid = points[..., [a, b]].unsqueeze(2)              # [B,N,1,2]
            sampled = F.grid_sample(
                plane.float(), grid.float(), mode="bilinear",
                padding_mode="zeros", align_corners=True)         # [B,C,N,1]
            samples.append(sampled.squeeze(-1).transpose(1, 2))
        return torch.cat(samples, dim=-1)

    def query(self, planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """points [B,N,3] in [-1,1] -> [B,N,4] = (density, rgb in [-1,1]), float32."""
        raw = self.field(self._sample_planes(planes, points))
        density = F.softplus(raw[..., :1].float() + self.density_bias)
        inside = (points.abs() <= 1.0).all(dim=-1, keepdim=True).to(density.dtype)
        color = torch.tanh(raw[..., 1:].float())
        return torch.cat((density * inside, color), dim=-1)

    def render(self, planes: torch.Tensor, origins: torch.Tensor,
               dirs: torch.Tensor, near: torch.Tensor, far: torch.Tensor,
               num_samples: int = 48, jitter: bool = False) -> dict:
        """origins/dirs: [B,T,P,3] with P = size*size. Returns image [B,T,3,s,s]
        (composited on the background) and alpha [B,T,1,s,s]."""
        batch, views, pixels, _ = dirs.shape
        size = int(round(pixels ** 0.5))
        device = dirs.device

        steps = torch.arange(num_samples, device=device, dtype=torch.float32)
        offset = (torch.rand(batch, views, pixels, num_samples, device=device)
                  if jitter else 0.5)
        t = near + (far - near) * (steps + offset) / num_samples  # [B,T,P,S]
        points = origins[..., None, :] + dirs[..., None, :] * t[..., None]

        out = self.query(planes, points.reshape(batch, -1, 3))
        out = out.reshape(batch, views, pixels, num_samples, 4)
        sigma, color = out[..., 0], out[..., 1:]

        delta = (far - near) / num_samples
        alpha = 1.0 - torch.exp(-sigma * delta)
        ones = torch.ones_like(alpha[..., :1])
        trans = torch.cumprod(
            torch.cat((ones, 1.0 - alpha + 1e-6), dim=-1), dim=-1)[..., :-1]
        weights = alpha * trans                                   # [B,T,P,S]

        rgb = (weights[..., None] * color).sum(dim=-2)            # [B,T,P,3]
        acc = weights.sum(dim=-1)                                 # [B,T,P]
        image = rgb + (1.0 - acc[..., None]) * self.background

        image = image.view(batch, views, size, size, 3).permute(0, 1, 4, 2, 3)
        acc = acc.view(batch, views, 1, size, size)
        return {"image": image, "alpha": acc}

    def forward(self, planes, origins, dirs, near, far,
                num_samples: int = 48, jitter: bool = False) -> dict:
        return self.render(planes, origins, dirs, near, far, num_samples, jitter)

    @torch.no_grad()
    def extract_mesh(self, planes: torch.Tensor, resolution: int = 64,
                     threshold: float = 5.0) -> list:
        """Marching cubes on the density field (needs scikit-image)."""
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError("Install scikit-image to extract meshes.") from exc
        coords = torch.linspace(-1, 1, resolution, device=planes.device)
        grid = torch.stack(
            torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1)
        points = grid.reshape(1, -1, 3).expand(planes.shape[0], -1, -1)
        density = self.query(planes, points)[..., 0]
        density = density.reshape(-1, resolution, resolution, resolution)
        meshes = []
        for volume in density.cpu().numpy():
            vertices, faces, _, _ = marching_cubes(volume, level=threshold)
            vertices = vertices / (resolution - 1) * 2.0 - 1.0
            meshes.append({"vertices": torch.from_numpy(vertices.copy()),
                           "faces": torch.from_numpy(faces.copy())})
        return meshes