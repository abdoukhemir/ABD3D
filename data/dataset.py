import io
import json
import os
import random
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms


class CompleteObjaverseDataset(IterableDataset):
    """Loads one parquet shard at a time from the Complete Objaverse dataset and yields 1-input/11-target view samples."""

    def __init__(self, name: str = "zeyuanyin/complete-objaverse", split: str = "train",
                 image_size: int = 224, config_name: str | None = None,
                 image_keys: tuple[str, ...] = ("image_png", "image", "render", "front_image"),
                 num_workers: int | None = None, num_views: int = 12,
                 checkpoint_dir: str | Path = "checkpoints", max_cache_bytes: int = 200 * 1024 * 1024):
        super().__init__()
        self.name = name
        self.split = split
        self.config_name = config_name
        self.image_keys = image_keys
        self.num_views = max(2, int(num_views))
        self.num_target_views = max(self.num_views - 1, 1)
        if num_workers is None:
            num_workers = 0 if os.name == "nt" else 2
        self.num_workers = num_workers
        self.image_size = int(image_size)
        self.max_cache_bytes = int(max_cache_bytes)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.progress_path = self.checkpoint_dir / "shard_progress.json"
        self.cache_dir = self.checkpoint_dir / "rolling_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._progress = self._load_progress()
        self._completed = set(self._progress.get("completed_shards", []))
        self._current_shard: str | None = self._progress.get("current_shard")
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {"completed_shards": [], "current_shard": None}
        try:
            with self.progress_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            return {"completed_shards": [], "current_shard": None}
        return {"completed_shards": data.get("completed_shards", []), "current_shard": data.get("current_shard")}

    def _save_progress(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {"completed_shards": sorted(self._completed), "current_shard": self._current_shard}
        with self.progress_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _list_shards(self) -> list[str]:
        api = HfApi()
        files = api.list_repo_files(repo_id=self.name, repo_type="dataset")
        shards = sorted(
            path for path in files
            if path.endswith(".parquet") and ("train" in path.lower() or "data/" in path.lower())
        )
        return shards

    def _download_shard(self, shard_name: str) -> Path:
        local_path = hf_hub_download(
            repo_id=self.name,
            repo_type="dataset",
            filename=shard_name,
            local_dir=str(self.cache_dir),
            local_dir_use_symlinks=False,
        )
        return Path(local_path)

    def _decode_bytes_to_tensor(self, payload: bytes | None) -> torch.Tensor | None:
        if payload is None:
            return None
        try:
            image = Image.open(io.BytesIO(payload))
        except Exception:
            return None
        if image.mode not in {"RGB", "RGBA"}:
            try:
                image = image.convert("RGB")
            except Exception:
                return None
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = self.resize(image)
        return self.rgb_transform(image)

    def _infer_object_id(self, row: dict, fallback: int) -> str:
        for key in ("object_id", "uid", "id", "sample_id", "asset_id", "model_id", "hash"):
            value = row.get(key)
            if value is not None:
                return str(value)
        return f"row_{fallback}"

    def _build_object_samples(self, rows: list[dict]) -> Iterator[dict[str, torch.Tensor]]:
        views: list[dict[str, torch.Tensor]] = []
        for row in rows:
            image = self._decode_bytes_to_tensor(row.get("image_png"))
            depth = self._decode_bytes_to_tensor(row.get("nd_png"))
            if image is not None and depth is not None:
                views.append({"image": image, "depth": depth})

        if len(views) < 2:
            return

        for start in range(0, len(views), self.num_views):
            chunk = views[start:start + self.num_views]
            if len(chunk) < 2:
                continue
            if len(chunk) < self.num_views:
                while len(chunk) < self.num_views:
                    chunk.append(chunk[-1])

            input_index = random.randrange(len(chunk))
            target_indices = [idx for idx in range(len(chunk)) if idx != input_index]
            if len(target_indices) < self.num_target_views:
                target_indices = target_indices + random.choices(target_indices, k=self.num_target_views - len(target_indices))
            else:
                target_indices = random.sample(target_indices, self.num_target_views)

            input_view = chunk[input_index]["image"].unsqueeze(0)
            target_views = torch.stack([chunk[idx]["image"] for idx in target_indices], dim=0)
            depth_maps = torch.stack([chunk[idx]["depth"] for idx in [input_index, *target_indices]], dim=0)
            yield {"input_view": input_view, "target_views": target_views, "depth_maps": depth_maps}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shards = self._list_shards()
        if not shards:
            raise ValueError(f"No parquet shards were found in the dataset '{self.name}'.")

        for shard_name in shards:
            if shard_name in self._completed:
                continue

            self._current_shard = shard_name
            self._save_progress()
            parquet_path = self._download_shard(shard_name)
            try:
                dataframe = pd.read_parquet(parquet_path)
                grouped: dict[str, list[dict]] = {}
                for row_index, row in dataframe.iterrows():
                    object_key = self._infer_object_id(row.to_dict(), row_index)
                    grouped.setdefault(str(object_key), []).append(row.to_dict())

                for rows in grouped.values():
                    for sample in self._build_object_samples(rows):
                        yield sample

                self._completed.add(shard_name)
                self._save_progress()
            finally:
                if parquet_path.exists():
                    parquet_path.unlink(missing_ok=True)
                self._current_shard = None
                self._save_progress()


ObjaverseStreamingDataset = CompleteObjaverseDataset
