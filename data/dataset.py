import io
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import IterableDataset

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "checkpoints" / "rolling_cache"
PROGRESS_FILE = BASE_DIR / "checkpoints" / "shard_progress.json"
SHARD_LIST_FILE = BASE_DIR / "checkpoints" / "shard_list.json"
BACKGROUND = 255.0  # views are composited onto white

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path) -> Path:
    if path is None:
        return BASE_DIR / "checkpoints"
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate.resolve()


# --------------------------------------------------------------------------- #
# shard list / progress
# --------------------------------------------------------------------------- #
def _load_or_fetch_shards(repo_id: str) -> list:
    if SHARD_LIST_FILE.exists():
        try:
            data = json.loads(SHARD_LIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                print(f"[ABD3D] Loaded {len(data)} shards from local cache ✅")
                return data
        except Exception:
            pass

    print("[ABD3D] Fetching shard list from HuggingFace (first time only)...")
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    shards = sorted(
        f for f in files if f.endswith(".parquet") and "dome_objaverse" in f)
    SHARD_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHARD_LIST_FILE.write_text(json.dumps(shards, indent=2), encoding="utf-8")
    print(f"[ABD3D] Found {len(shards)} shards — saved locally ✅")
    return shards


def split_shards(shards: list, seed: int, val_objects: int):
    """Deterministic shuffle; the first `val_objects` shards are held out."""
    order = sorted(shards)
    random.Random(seed).shuffle(order)
    return order[val_objects:], order[:val_objects]


def _load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {"current_shard_index": 0, "total_shards_processed": 0}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"current_shard_index": 0, "total_shards_processed": 0}


def _save_progress(index: int, total: int, name: str) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({
        "current_shard_index": index,
        "total_shards_processed": total,
        "last_shard": name,
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# download + decode
# --------------------------------------------------------------------------- #
def _download_shard(repo_id: str, shard_name: str, max_retries: int = 5) -> Path:
    """Returns the local path or raises RuntimeError. Never touches progress."""
    local_path = CACHE_DIR / shard_name.replace("/", "__")
    if local_path.exists():
        return local_path
    part_path = local_path.with_suffix(local_path.suffix + ".part")

    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard_name}"

    last_error = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, timeout=(10, 30),
                              stream=True) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            os.replace(part_path, local_path)   # only complete files are "cached"
            return local_path
        except Exception as exc:
            last_error = exc
            part_path.unlink(missing_ok=True)
            print(f"[ABD3D] Download error ({shard_name}, "
                  f"attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {shard_name}: {last_error}")


def _decode_view(payload, image_size: int):
    """RGBA png -> (rgb uint8 [3,S,S] composited on white, alpha uint8 [1,S,S])."""
    if payload is None:
        return None
    try:
        arr = np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"),
                         dtype=np.float32)
        alpha = arr[..., 3:4] / 255.0
        rgb = arr[..., :3] * alpha + BACKGROUND * (1.0 - alpha)
        size = (image_size, image_size)
        rgb_img = Image.fromarray(
            np.clip(rgb.round(), 0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        alpha_img = Image.fromarray(
            np.clip(arr[..., 3].round(), 0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        rgb_t = torch.from_numpy(np.array(rgb_img)).permute(2, 0, 1).contiguous()
        alpha_t = torch.from_numpy(np.array(alpha_img)).unsqueeze(0).contiguous()
        return rgb_t, alpha_t
    except Exception:
        return None


def load_object(repo_id: str, shard_name: str, image_size: int):
    """One shard = one object (48 views). Returns a dict of uint8 tensors or None."""
    path = _download_shard(repo_id, shard_name)
    try:
        table = pq.read_table(path, columns=["view_id", "image_png"])
        view_ids = table.column("view_id").to_pylist()
        payloads = table.column("image_png").to_pylist()
    finally:
        path.unlink(missing_ok=True)

    images, alphas, ids = [], [], []
    for view_id, payload in sorted(zip(view_ids, payloads), key=lambda x: x[0]):
        decoded = _decode_view(payload, image_size)
        if decoded is None:
            continue  # view_id stays correct even when a view is dropped
        images.append(decoded[0])
        alphas.append(decoded[1])
        ids.append(view_id)
    if len(ids) < 2:
        return None
    return {
        "images": torch.stack(images),                         # [V,3,S,S] uint8
        "alphas": torch.stack(alphas),                         # [V,1,S,S] uint8
        "view_ids": torch.tensor(ids, dtype=torch.long),       # [V]
    }


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def make_sample(obj: dict, num_targets: int, same_view_prob: float, rng) -> dict:
    n = obj["view_ids"].shape[0]
    inp = rng.randrange(n)
    others = [i for i in range(n) if i != inp]
    if len(others) >= num_targets:
        targets = rng.sample(others, num_targets)
    else:
        targets = [rng.choice(others) for _ in range(num_targets)]
    targets = [inp if rng.random() < same_view_prob else t for t in targets]
    idx = torch.tensor(targets, dtype=torch.long)
    return {
        "input_image": obj["images"][inp],            # [3,S,S] uint8
        "target_images": obj["images"][idx],          # [T,3,S,S] uint8
        "target_alpha": obj["alphas"][idx],           # [T,1,S,S] uint8
        "input_view_id": obj["view_ids"][inp],        # scalar long
        "target_view_ids": obj["view_ids"][idx],      # [T] long
    }


def build_fixed_batch(objects: list, num_targets: int, pairs_per_object: int,
                      seed: int = 0) -> dict:
    """Deterministic validation batch."""
    rng = random.Random(seed)
    samples = [make_sample(o, num_targets, 0.0, rng)
               for o in objects for _ in range(pairs_per_object)]
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


class _ObjectFeeder:
    """Background thread that downloads + decodes shards so the GPU never waits."""

    def __init__(self, repo_id, shards, image_size, start_index,
                 total_processed, prefetch):
        self.repo_id = repo_id
        self.shards = shards
        self.image_size = image_size
        self.start_index = start_index
        self.total_processed = total_processed
        self.queue = queue.Queue(maxsize=max(1, prefetch))
        self.stop_event = threading.Event()
        self.fatal = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        n = len(self.shards)
        index, total, failures = self.start_index, self.total_processed, 0
        while not self.stop_event.is_set():
            name = self.shards[index % n]
            obj = None
            try:
                obj = load_object(self.repo_id, name, self.image_size)
            except Exception as exc:
                print(f"[ABD3D] Skipping shard {name}: {exc}")
            index += 1
            total += 1
            _save_progress(index % n, total, name)
            if obj is None:
                failures += 1
                if failures >= 10:
                    self.fatal = RuntimeError(
                        "10 shards in a row failed to load; check network/HF_TOKEN")
                    return
                continue
            failures = 0
            while not self.stop_event.is_set():
                try:
                    self.queue.put(obj, timeout=1.0)
                    break
                except queue.Full:
                    pass

    def get(self, timeout: float = 900.0):
        waited = 0.0
        while True:
            try:
                return self.queue.get(timeout=5.0)
            except queue.Empty:
                if self.fatal is not None:
                    raise self.fatal
                waited += 5.0
                if waited >= timeout:
                    raise RuntimeError("Timed out waiting for the next object")


class CompleteObjaverseDataset(IterableDataset):
    """Infinite stream of (input view, target views) pairs.

    Keeps `pool_size` objects in memory and draws random samples across them,
    so every batch mixes several objects. Use with DataLoader(num_workers=0).
    """

    def __init__(self, name: str, shards: list, image_size: int = 224,
                 num_target_views: int = 2, pool_size: int = 8,
                 samples_per_object: int = 48, prefetch: int = 4,
                 same_view_prob: float = 0.1, preloaded_objects=None, **_ignored):
        super().__init__()
        self.name = name
        self.shards = shards
        self.image_size = image_size
        self.num_target_views = num_target_views
        self.pool_size = pool_size
        self.samples_per_object = samples_per_object
        self.prefetch = prefetch
        self.same_view_prob = same_view_prob
        self.preloaded = preloaded_objects

    def __iter__(self):
        feeder = None
        if self.preloaded is not None:              # overfit/debug mode
            pool = [[o, 0] for o in self.preloaded]
        else:
            progress = _load_progress()
            start = int(progress.get("current_shard_index", 0)) % len(self.shards)
            total = int(progress.get("total_shards_processed", 0))
            feeder = _ObjectFeeder(self.name, self.shards, self.image_size,
                                   start, total, self.prefetch)
            feeder.start()
            print(f"[ABD3D] Filling object pool ({self.pool_size} objects)...")
            pool = [[feeder.get(), self.samples_per_object]
                    for _ in range(self.pool_size)]
            print("[ABD3D] Object pool ready ✅")

        while True:
            slot = random.randrange(len(pool))
            yield make_sample(pool[slot][0], self.num_target_views,
                              self.same_view_prob, random)
            if feeder is not None:
                pool[slot][1] -= 1
                if pool[slot][1] <= 0:
                    pool[slot] = [feeder.get(), self.samples_per_object]


ObjaverseStreamingDataset = CompleteObjaverseDataset
ShapeNetStreamingDataset = CompleteObjaverseDataset