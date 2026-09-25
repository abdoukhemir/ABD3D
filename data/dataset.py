import io
import json
import os
import random
import threading
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "checkpoints" / "rolling_cache"
SHARD_DIR = CACHE_DIR / "shards"
META_DIR  = CACHE_DIR / "metadata"
PROGRESS_FILE   = BASE_DIR / "checkpoints" / "shard_progress.json"
SHARD_LIST_FILE = BASE_DIR / "checkpoints" / "shard_list.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
print(f"[ABD3D] BASE_DIR={BASE_DIR}")


def resolve_project_path(path) -> Path:
    if path is None:
        return BASE_DIR / "checkpoints"
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate.resolve()


class ShardBuffer:
    """Rolling shard cache — max 2 shards on disk at any time."""

    def __init__(self, repo_id: str, checkpoint_dir=None):
        self.repo_id = repo_id
        self.checkpoint_dir = resolve_project_path(checkpoint_dir)
        self.progress_path = PROGRESS_FILE
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shards = self._list_shards()
        self.progress = self._load_progress()
        self.current_index = int(self.progress.get("current_shard_index", 0))
        self.total_processed = int(self.progress.get("total_shards_processed", 0))
        self.current_path: Path | None = None
        self.next_path: Path | None = None
        self._download_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._download_ready = threading.Event()
        self._current_name: str | None = None
        self._save_progress()

    def _list_shards(self) -> list:
        # Load from local cache — NEVER call HF API more than once
        if SHARD_LIST_FILE.exists():
            try:
                data = json.loads(
                    SHARD_LIST_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    print(f"[ABD3D] Loaded {len(data)} shards from local cache ✅")
                    return data
            except Exception:
                pass

        print("[ABD3D] Fetching shard list from HuggingFace (first time only)...")
        api = HfApi()
        files = api.list_repo_files(
            repo_id=self.repo_id, repo_type="dataset")
        shards = sorted(
            f for f in files
            if f.endswith(".parquet") and "dome_objaverse" in f
        )
        SHARD_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        SHARD_LIST_FILE.write_text(
            json.dumps(shards, indent=2), encoding="utf-8")
        print(f"[ABD3D] Found {len(shards)} shards — saved to local cache ✅")
        return shards

    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {"current_shard_index": 0, "total_shards_processed": 0}
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {"current_shard_index": 0, "total_shards_processed": 0}

    def _save_progress(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_shard_index": self.current_index,
            "total_shards_processed": self.total_processed,
            "last_shard": self._current_name,
        }
        self.progress_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    def _download_shard(self, shard_name: str) -> Path:
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=shard_name,
            local_dir=str(self.cache_dir),
            local_dir_use_symlinks=False,
        )
        return Path(local_path)

    def _delete_if_exists(self, path) -> None:
        if path is not None:
            p = Path(path)
            if p.exists():
                p.unlink(missing_ok=True)

    def _start_next_download(self) -> None:
        if self._download_thread is not None and self._download_thread.is_alive():
            return
        next_index = (self.current_index + 1) % len(self.shards)
        next_name = self.shards[next_index]
        self._download_ready.clear()

        def worker() -> None:
            try:
                download_path = self._download_shard(next_name)
                with self._lock:
                    self.next_path = Path(download_path)
                    self._download_ready.set()
            except Exception:
                self._download_ready.set()

        self._download_thread = threading.Thread(target=worker, daemon=True)
        self._download_thread.start()

    def _wait_for_next_shard(self) -> Path | None:
        if self.next_path is not None:
            return self.next_path
        self._start_next_download()
        if self._download_ready.wait(timeout=120):
            return self.next_path
        return None

    def get_current_shard(self) -> Path:
        if self.current_index >= len(self.shards):
            self.current_index = 0
        shard_name = self.shards[self.current_index]
        self._current_name = shard_name
        self._save_progress()
        self.current_path = self._download_shard(shard_name)
        self._start_next_download()
        return self.current_path

    def advance(self) -> None:
        self._delete_if_exists(self.current_path)
        self.current_path = None
        self.total_processed += 1
        self.current_index = (self.current_index + 1) % len(self.shards)
        self._save_progress()
        print(f"[ABD3D] Shard {self.current_index}/{len(self.shards)} "
              f"({self.total_processed} total processed) ✅")


class CompleteObjaverseDataset(IterableDataset):
    """
    Streams zeyuanyin/complete-objaverse using a rolling shard cache.
    Each parquet file = 1 object = 48 views (view_id 0-47).
    Yields: input_view [3,H,W], target_views [11,3,H,W], depth_maps [12,3,H,W]
    """

    def __init__(
        self,
        name: str = "zeyuanyin/complete-objaverse",
        split: str = "train",
        image_size: int = 224,
        config_name=None,
        image_keys=("image_png",),
        num_workers=None,
        num_views: int = 12,
        checkpoint_dir=None,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.num_views = max(2, int(num_views))
        self.num_target_views = self.num_views - 1
        self.image_size = int(image_size)
        self.checkpoint_dir = resolve_project_path(checkpoint_dir)
        # Always 0 for IterableDataset — avoids multiprocessing issues
        self.num_workers = 0
        self.buffer = ShardBuffer(name, checkpoint_dir=self.checkpoint_dir)
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def _decode(self, payload) -> torch.Tensor | None:
        if payload is None:
            return None
        try:
            img = Image.open(io.BytesIO(payload))
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img = self.resize(img)
            return self.to_tensor(img)
        except Exception:
            return None

    def __iter__(self) -> Iterator:
        if not self.buffer.shards:
            raise ValueError(f"No shards found for dataset '{self.name}'")

        # Download current shard
        current_path = self.buffer.get_current_shard()

        try:
            df = pd.read_parquet(current_path)

            # Sort by view_id — entire file is ONE object
            if "view_id" in df.columns:
                df = df.sort_values("view_id").reset_index(drop=True)

            # Decode all views in this shard
            views = []
            for _, row in df.iterrows():
                img = self._decode(row.get("image_png"))
                dep = self._decode(row.get("nd_png"))
                if img is not None and dep is not None:
                    views.append({"image": img, "depth": dep})

            print(f"[ABD3D] Shard has {len(views)} valid views")

            if len(views) < 2:
                return

            # Pad views if fewer than num_views
            while len(views) < self.num_views:
                views.append(views[-1])

            # Yield multiple training samples from this object
            num_samples = max(1, len(views) // self.num_views)
            for _ in range(num_samples):
                # Pick random input view
                input_idx = random.randrange(len(views))

                # Pick random target views (different from input)
                all_indices = list(range(len(views)))
                all_indices.remove(input_idx)
                target_idxs = random.sample(
                    all_indices,
                    min(self.num_target_views, len(all_indices))
                )

                # Pad targets if needed
                while len(target_idxs) < self.num_target_views:
                    target_idxs.append(target_idxs[-1])

                input_view = views[input_idx]["image"]
                target_views = torch.stack(
                    [views[i]["image"] for i in target_idxs])
                depth_maps = torch.stack(
                    [views[i]["depth"] for i in [input_idx, *target_idxs]])

                yield {
                    "input_view": input_view,     # [3, H, W]
                    "target_views": target_views,  # [11, 3, H, W]
                    "depth_maps": depth_maps,      # [12, 3, H, W]
                }

        except Exception as e:
            print(f"[ABD3D] Error reading shard: {e}")

        finally:
            # Always advance — never crash
            self.buffer.advance()


# Alias for backwards compatibility
ObjaverseStreamingDataset = CompleteObjaverseDataset
ShapeNetStreamingDataset = CompleteObjaverseDataset