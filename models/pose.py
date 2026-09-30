import math

import torch
import torch.nn.functional as F
from torch import nn


def view_id_to_angles(view_ids: torch.Tensor, views_per_ring: int,
                      azimuth_direction: float = 1.0):
    """view_id -> (azimuth in radians, ring index). Assumes id = ring * views_per_ring + slot."""
    ring = torch.div(view_ids, views_per_ring, rounding_mode="floor")
    slot = view_ids - ring * views_per_ring
    azimuth = azimuth_direction * slot.float() * (2.0 * math.pi / views_per_ring)
    return azimuth, ring


class PoseEncoder(nn.Module):
    """Embeds the camera pose (azimuth + ring) of the INPUT view."""

    def __init__(self, num_rings: int, views_per_ring: int, dim: int,
                 azimuth_direction: float = 1.0):
        super().__init__()
        self.num_rings = num_rings
        self.views_per_ring = views_per_ring
        self.azimuth_direction = azimuth_direction
        self.mlp = nn.Sequential(
            nn.Linear(2 + num_rings, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, view_ids: torch.Tensor) -> torch.Tensor:
        azimuth, ring = view_id_to_angles(
            view_ids, self.views_per_ring, self.azimuth_direction)
        ring = ring.clamp(0, self.num_rings - 1)
        feats = torch.cat((
            torch.sin(azimuth)[:, None],
            torch.cos(azimuth)[:, None],
            F.one_hot(ring, self.num_rings).float(),
        ), dim=-1)
        return self.mlp(feats)


class OrbitCamera(nn.Module):
    """Cameras orbiting the origin. Ring elevations, radius and FOV are learnable
    because the dataset does not tell us the real values."""

    def __init__(self, num_rings: int, views_per_ring: int,
                 ring_elevations_deg, radius: float, fov_deg: float,
                 azimuth_direction: float, scene_radius: float):
        super().__init__()
        elevations = torch.tensor(list(ring_elevations_deg), dtype=torch.float32)
        if elevations.numel() != num_rings:
            raise ValueError("ring_elevations_deg must have num_rings entries")
        self.num_rings = num_rings
        self.views_per_ring = views_per_ring
        self.azimuth_direction = azimuth_direction
        self.scene_radius = scene_radius
        self.ring_elevation = nn.Parameter(elevations * math.pi / 180.0)
        self.log_radius = nn.Parameter(torch.tensor(math.log(radius)))
        self.log_tan_half_fov = nn.Parameter(
            torch.tensor(math.log(math.tan(math.radians(fov_deg) / 2.0))))

    def forward(self, view_ids: torch.Tensor, size: int):
        """view_ids: [B, T]. Returns origins, dirs [B, T, size*size, 3], near, far."""
        azimuth, ring = view_id_to_angles(
            view_ids, self.views_per_ring, self.azimuth_direction)
        ring = ring.clamp(0, self.num_rings - 1)
        elevation = self.ring_elevation[ring].clamp(-1.56, 1.56)
        radius = self.log_radius.exp()

        cos_el = torch.cos(elevation)
        cam_pos = radius * torch.stack((
            cos_el * torch.cos(azimuth),
            cos_el * torch.sin(azimuth),
            torch.sin(elevation)), dim=-1)                       # [B, T, 3]
        forward = F.normalize(-cam_pos, dim=-1)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=cam_pos.device,
                                dtype=cam_pos.dtype).expand_as(forward)
        right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
        up = torch.cross(right, forward, dim=-1)

        half = self.log_tan_half_fov.exp()
        coords = (torch.arange(size, device=cam_pos.device,
                               dtype=cam_pos.dtype) + 0.5) / size * 2.0 - 1.0
        v, u = torch.meshgrid(-coords, coords, indexing="ij")    # top row = +1
        u = u.reshape(1, 1, -1, 1) * half
        v = v.reshape(1, 1, -1, 1) * half
        dirs = forward[:, :, None, :] + u * right[:, :, None, :] + v * up[:, :, None, :]
        dirs = F.normalize(dirs, dim=-1)                         # [B, T, P, 3]
        origins = cam_pos[:, :, None, :].expand_as(dirs)

        near = (radius - self.scene_radius).clamp(min=0.05)
        far = radius + self.scene_radius
        return origins, dirs, near, far

    @torch.no_grad()
    def state(self) -> dict:
        out = {
            "cam_radius": self.log_radius.exp().item(),
            "cam_fov_deg": math.degrees(
                2.0 * math.atan(self.log_tan_half_fov.exp().item())),
        }
        for i, e in enumerate(self.ring_elevation.tolist()):
            out[f"cam_ring{i}_elev_deg"] = math.degrees(e)
        return out