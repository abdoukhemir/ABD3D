import io
import json
import os
import random
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "checkpoints" / "rolling_cache"
PROGRESS_FILE   = BASE_DIR / "checkpoints" / "shard_progress.json"
SHARD_LIST_FILE = BASE_DIR / "checkpoints" / "shard_list.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"[ABD3D] BASE_DIR={BASE_DIR}")


def resolve_project_path(path) -> Path:
    if path is None:
        return BASE_DIR / "checkpoints"
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate.resolve()


def _load_or_fetch_shards(repo_id: str) -> list:
    """Fetch shard list ONCE — save locally — never call HF API again."""
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
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    shards = sorted(
        f for f in files
        if f.endswith(".parquet") and "dome_objaverse" in f
    )
    SHARD_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHARD_LIST_FILE.write_text(json.dumps(shards, indent=2), encoding="utf-8")
    print(f"[ABD3D] Found {len(shards)} shards — saved locally ✅")
    return shards


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


def _download_shard_wget(repo_id: str, shard_name: str) -> Path:
    """
    Download shard using wget — reliable in ALL Kaggle modes.
    hf_hub_download hangs in commit mode ❌
    wget works perfectly ✅
    """
    local_path = CACHE_DIR / Path(shard_name).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # If already exists — skip download
    if local_path.exists():
        print(f"[ABD3D] Shard already cached ✅")
        return local_path

    token = os.environ.get('HF_TOKEN', '')
    url = (f"https://huggingface.co/datasets/{repo_id}"
           f"/resolve/main/{shard_name}")

    print(f"[ABD3D] Downloading: {shard_name}")
    result = subprocess.run([
        'wget',
        '--header', f'Authorization: Bearer {token}',
        '-q',
        '--show-progress',
        '-O', str(local_path),
        url
    ], capture_output=False)

    if result.returncode != 0 or not local_path.exists():
        raise RuntimeError(f"wget failed for {shard_name}")

    print(f"[ABD3D] Downloaded to: {local_path}")
    return local_path


class CompleteObjaverseDataset(IterableDataset):
    """
    Rolling cache dataset for zeyuanyin/complete-objaverse.
    Each parquet = 1 object = 48 views.
    Downloads ONE shard → trains → deletes → next shard.
    Max disk: ~30MB at any time.
    Uses wget for reliable downloads in all Kaggle modes.
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
        self.num_views = max(2, int(num_views))
        self.num_target_views = self.num_views - 1
        self.image_size = int(image_size)
        self.num_workers = 0
        self.shards = _load_or_fetch_shards(name)

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
        if not self.shards:
            raise ValueError(f"No shards found for '{self.name}'")

        # Load progress
        progress = _load_progress()
        shard_index = int(progress.get("current_shard_index", 0))
        total_processed = int(progress.get("total_shards_processed", 0))

        if shard_index >= len(self.shards):
            shard_index = 0

        shard_name = self.shards[shard_index]
        print(f"[ABD3D] Loading shard {shard_index}/{len(self.shards)}: {shard_name}")

        # ✅ Use wget — no more hanging in commit mode!
        local_path = _download_shard_wget(self.name, shard_name)

        try:
            df = pd.read_parquet(local_path)

            if "view_id" in df.columns:
                df = df.sort_values("view_id").reset_index(drop=True)

            views = []
            for _, row in df.iterrows():
                img = self._decode(row.get("image_png"))
                dep = self._decode(row.get("nd_png"))
                if img is not None and dep is not None:
                    views.append({"image": img, "depth": dep})

            print(f"[ABD3D] {len(views)} valid views in shard")

            if len(views) < 2:
                return

            while len(views) < self.num_views:
                views.append(views[-1])

            num_samples = max(16, len(views) * 4)
            yielded = 0
            for _ in range(num_samples):
                input_idx = random.randrange(len(views))
                all_idx = [i for i in range(len(views)) if i != input_idx]
                target_idxs = random.sample(
                    all_idx, min(self.num_target_views, len(all_idx)))

                while len(target_idxs) < self.num_target_views:
                    target_idxs.append(target_idxs[-1])

                input_view = views[input_idx]["image"]
                target_views = torch.stack(
                    [views[i]["image"] for i in target_idxs])
                depth_maps = torch.stack(
                    [views[i]["depth"] for i in [input_idx, *target_idxs]])

                yield {
                    "input_view": input_view,
                    "target_views": target_views,
                    "depth_maps": depth_maps,
                }
                yielded += 1

            print(f"[ABD3D] Yielded {yielded} samples from shard ✅")

        except Exception as e:
            print(f"[ABD3D] Error reading shard {shard_name}: {e}")

        finally:
            if local_path.exists():
                local_path.unlink()
                print(f"[ABD3D] Deleted shard {shard_name} ✅")

            next_index = (shard_index + 1) % len(self.shards)
            _save_progress(next_index, total_processed + 1, shard_name)
            print(f"[ABD3D] Advanced to shard {next_index}/{len(self.shards)}")


# Aliases
ObjaverseStreamingDataset = CompleteObjaverseDataset
ShapeNetStreamingDataset  = CompleteObjaverseDataset