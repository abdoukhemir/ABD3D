import io
import os
import random
from collections.abc import Iterator

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms


class CompleteObjaverseDataset(IterableDataset):
    """Streams the Complete Objaverse dataset grouped by object and returns one input view plus three target views."""

    VIEWS_PER_OBJECT = 36

    def __init__(self, name: str = "zeyuanyin/complete-objaverse", split: str = "train",
                 image_size: int = 224, config_name: str | None = None,
                 image_keys: tuple[str, ...] = ("image_png", "image", "render", "front_image"),
                 num_workers: int | None = None):
        super().__init__()
        self.name = name
        self.split = split
        self.config_name = config_name
        self.image_keys = image_keys
        if num_workers is None:
            num_workers = 0 if os.name == "nt" else 2
        self.num_workers = num_workers
        self.resize = transforms.Resize((image_size, image_size))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def _decode_bytes_to_tensor(self, payload: bytes | None) -> torch.Tensor | None:
        if payload is None:
            return None
        try:
            image = Image.open(io.BytesIO(payload))
        except Exception:
            return None
        if image.mode in {"RGBA", "LA"}:
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image = self.resize(image)
        return self.rgb_transform(image)

    def _build_object_sample(self, records: list[dict]) -> dict[str, torch.Tensor] | None:
        views: list[dict[str, torch.Tensor]] = []
        for record in records:
            image = self._decode_bytes_to_tensor(record.get("image_png"))
            depth = self._decode_bytes_to_tensor(record.get("nd_png"))
            if image is not None and depth is not None:
                views.append({"image": image, "depth": depth})
        if len(views) < 4:
            return None

        input_index = random.randrange(len(views))
        target_indices = [idx for idx in range(len(views)) if idx != input_index]
        if len(target_indices) < 3:
            target_indices = target_indices + [input_index] * (3 - len(target_indices))
        selected_targets = random.sample(target_indices, 3)
        selected_indices = [input_index] + selected_targets

        input_view = views[input_index]["image"].unsqueeze(0)
        target_views = torch.stack([views[idx]["image"] for idx in selected_targets], dim=0)
        depth_maps = torch.stack([views[idx]["depth"] for idx in selected_indices], dim=0)
        return {
            "input_view": input_view,
            "target_views": target_views,
            "depth_maps": depth_maps,
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        kwargs = {"path": self.name, "split": self.split, "streaming": True}
        if self.config_name:
            kwargs["name"] = self.config_name
        stream = load_dataset(**kwargs)
        object_buffer: list[dict] = []
        for record in stream:
            object_buffer.append(record)
            if len(object_buffer) == self.VIEWS_PER_OBJECT:
                sample = self._build_object_sample(object_buffer)
                if sample is not None:
                    yield sample
                object_buffer = []


ObjaverseStreamingDataset = CompleteObjaverseDataset
