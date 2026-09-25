import torch
from torch import nn


class TriplaneDecoder(nn.Module):
    """Decodes three feature planes and provides a differentiable image proxy."""

    def __init__(self, plane_channels: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.field = nn.Sequential(
            nn.Linear(plane_channels * 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.image_head = nn.Sequential(
            nn.Conv2d(plane_channels * 3, hidden_dim, 1), nn.SiLU(),
            nn.Conv2d(hidden_dim, 3, 1), nn.Sigmoid(),
        )

    def forward(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = torch.cat((planes[:, 0], planes[:, 1], planes[:, 2]), dim=1)
        image = self.image_head(feature_map)
        return {"planes": planes, "image": image}

    def extract_mesh(self, planes: torch.Tensor, resolution: int = 64,
                     threshold: float = 0.0) -> list[dict[str, torch.Tensor]]:
        """Extract approximate meshes from triplanes when scikit-image is installed."""
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError("Install scikit-image to extract meshes.") from exc
        coordinates = torch.linspace(-1, 1, resolution, device=planes.device)
        grid = torch.stack(torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), dim=-1)
        points = grid.reshape(1, -1, 3).expand(planes.shape[0], -1, -1)
        field = self.query(planes, points)[..., 0].reshape(-1, resolution, resolution, resolution)
        meshes = []
        for volume in field.detach().cpu().numpy():
            vertices, faces, _, _ = marching_cubes(volume, level=threshold)
            meshes.append({"vertices": torch.from_numpy(vertices), "faces": torch.from_numpy(faces)})
        return meshes

    def query(self, planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Query occupancy and RGB at normalized points shaped [B, N, 3]."""
        samples = []
        for plane, coordinates in zip(planes.unbind(1), ((1, 2), (0, 2), (0, 1))):
            grid = points[..., list(coordinates)].mul(2).sub(1).unsqueeze(2)
            samples.append(nn.functional.grid_sample(plane, grid, align_corners=True).squeeze(-1).transpose(1, 2))
        return self.field(torch.cat(samples, dim=-1))
