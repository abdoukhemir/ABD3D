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
META_DIR = CACHE_DIR / "metadata"
PROGRESS_FILE = BASE_DIR / "checkpoints" / "shard_progress.json"
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
    """A bounded rolling shard cache that keeps at most two parquet shards on disk."""

    def __init__(self, repo_id: str, checkpoint_dir=None):
        self.repo_id = repo_id
        self.checkpoint_dir = resolve_project_path(checkpoint_dir)
        self.progress_path = PROGRESS_FILE if checkpoint_dir is None else (self.checkpoint_dir / "shard_progress.json")
        self.cache_dir = CACHE_DIR if checkpoint_dir is None else (self.checkpoint_dir / "rolling_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shards = self._list_shards()
        self.progress = self._load_progress()
        self.current_index = int(self.progress.get("current_shard_index", 0))
        self.total_processed = int(self.progress.get("total_shards_processed", 0))
        self.completed = set(self.progress.get("completed_shards", []))
        self.current_path: Path | None = None
        self.next_path: Path | None = None
        self._download_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._download_ready = threading.Event()
        self._current_name: str | None = None
        self._next_name: str | None = None
        self._save_progress()

    def _list_shards(self) -> list:
        # FIX 1: Load from local cache — NEVER call HF API more than once
        if SHARD_LIST_FILE.exists():
            try:
                data = json.loads(SHARD_LIST_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    print(f"[ABD3D] Loaded {len(data)} shards from local cache ✅")
                    return data
            except Exception:
                pass

        print("[ABD3D] Fetching shard list from HuggingFace (first time only)...")
        api = HfApi()
        files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        shards = sorted(
            path for path in files
            if path.endswith(".parquet") and "dome_objaverse" in path
        )
        SHARD_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        SHARD_LIST_FILE.write_text(json.dumps(shards, indent=2), encoding="utf-8")
        print(f"[ABD3D] Found {len(shards)} shards — saved to local cache ✅")
        return shards

    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {"current_shard_index": 0, "total_shards_processed": 0, "completed_shards": [], "last_shard": None}
        try:
            data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"current_shard_index": 0, "total_shards_processed": 0, "completed_shards": [], "last_shard": None}
        return {
            "current_shard_index": int(data.get("current_shard_index", 0)),
            "total_shards_processed": int(data.get("total_shards_processed", 0)),
            "completed_shards": data.get("completed_shards", []),
            "last_shard": data.get("last_shard"),
        }

    def _save_progress(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "current_shard_index": self.current_index,
            "total_shards_processed": self.total_processed,
            "completed_shards": sorted(self.completed),
            "last_shard": self._current_name,
        }
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
        if self.next_path is not None and self.next_path.name == Path(next_name).name:
            return
        self._download_ready.clear()
        self._next_name = next_name

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

    def open_current_shard(self) -> Path:
        if self.current_index >= len(self.shards):
            self.current_index = 0

        current_name = self.shards[self.current_index]
        self._current_name = current_name
        self._save_progress()
        self.current_path = self._download_shard(current_name)
        return self.current_path

    def advance(self) -> Path | None:
        self._delete_if_exists(self.current_path)
        self.current_path = None
        self.total_processed += 1
        self.completed.add(self.shards[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.shards)
        self._save_progress()
        total = len(self.shards)
        print(f"[ABD3D] Moving to shard {self.current_index}/{total} "
              f"({self.total_processed} total processed)")

        next_ready = self._wait_for_next_shard()
        if next_ready is not None:
            self.current_path = next_ready
            self._current_name = self.shards[self.current_index]
            self._save_progress()
            self.next_path = None
            self._start_next_download()
            return self.current_path

        self.current_path = self._download_shard(self.shards[self.current_index])
        self._current_name = self.shards[self.current_index]
        self._save_progress()
        self._start_next_download()
        return self.current_path


class CompleteObjaverseDataset(IterableDataset):
    def __init__(self, name="zeyuanyin/complete-objaverse", split="train",
                 image_size=224, config_name=None,
                 image_keys=("image_png", "image", "render", "front_image"),
                 num_workers=None, num_views=12, checkpoint_dir=None):
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
        self.checkpoint_dir = resolve_project_path(checkpoint_dir)
        self.buffer = ShardBuffer(name, checkpoint_dir=self.checkpoint_dir)
        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def _decode_bytes_to_tensor(self, payload) -> torch.Tensor | None:
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

    def _build_object_samples(self, rows: list) -> Iterator:
        views = []
        for row in rows:
            image = self._decode_bytes_to_tensor(row.get("image_png"))
            depth = self._decode_bytes_to_tensor(row.get("nd_png"))
            if image is not None and depth is not None:
                views.append({"image": image, "depth": depth})

        if len(views) < 2:
            return

        while len(views) < self.num_views:
            views.append(views[-1])

        for start in range(0, len(views), self.num_views):
            chunk = views[start:start + self.num_views]
            if len(chunk) < 2:
                continue
            while len(chunk) < self.num_views:
                chunk.append(chunk[-1])

            input_index = random.randrange(len(chunk))
            target_indices = [idx for idx in range(len(chunk)) if idx != input_index]
            if len(target_indices) < self.num_target_views:
                target_indices = target_indices + random.choices(
                    target_indices, k=self.num_target_views - len(target_indices))
            else:
                target_indices = random.sample(target_indices, self.num_target_views)

            # FIX 2: NO unsqueeze(0) here — let DataLoader add batch dim
            input_view = chunk[input_index]["image"]
            target_views = torch.stack([chunk[idx]["image"] for idx in target_indices], dim=0)
            depth_maps = torch.stack([chunk[idx]["depth"] for idx in [input_index, *target_indices]], dim=0)

            yield {
                "input_view": input_view,
                "target_views": target_views,
                "depth_maps": depth_maps,
            }

    def __iter__(self) -> Iterator:
        if not self.buffer.shards:
            raise ValueError(f"No parquet shards found in dataset '{self.name}'.")

        progress = self.buffer._load_progress()
        shard_index = int(progress.get("current_shard_index", 0))
        if shard_index >= len(self.buffer.shards):
            shard_index = 0

        self.buffer.current_index = shard_index
        self.buffer.total_processed = int(progress.get("total_shards_processed", 0))
        self.buffer.completed = set(progress.get("completed_shards", []))
        self.buffer._current_name = self.buffer.shards[shard_index]
        self.buffer._save_progress()

        current_path = self.buffer._download_shard(self.buffer.shards[shard_index])
        self.buffer.current_path = current_path
        self.buffer._start_next_download()

        try:
            dataframe = pd.read_parquet(current_path)
            grouped: dict = {}
            for row_index, row in dataframe.iterrows():
                object_key = self._infer_object_id(row.to_dict(), row_index)
                grouped.setdefault(str(object_key), []).append(row.to_dict())

            for rows in grouped.values():
                yield from self._build_object_samples(rows)

        except Exception as e:
            print(f"[ABD3D] Error reading shard: {e}")

        finally:
            # FIX 3: Always advance shard in finally — never raise StopIteration
            next_index = (shard_index + 1) % len(self.buffer.shards)
            self.buffer.current_index = next_index
            self.buffer.total_processed += 1
            self.buffer.completed.add(self.buffer.shards[shard_index])
            self.buffer._current_name = self.buffer.shards[next_index]
            self.buffer._save_progress()
            self.buffer._delete_if_exists(current_path)
            self.buffer.current_path = None


ObjaverseStreamingDataset = CompleteObjaverseDataset