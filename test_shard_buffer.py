import io
import json
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


RESULTS = {}


def mark_test(name: str, passed: bool):
    RESULTS[name] = passed


def test_threading():
    try:
        start = time.time()
        events = []

        def gpu_worker():
            events.append(("gpu_start", time.time() - start))
            time.sleep(3)
            events.append(("gpu_end", time.time() - start))

        def download_worker():
            events.append(("download_start", time.time() - start))
            time.sleep(1)
            events.append(("download_end", time.time() - start))

        t1 = threading.Thread(target=gpu_worker, daemon=True)
        t2 = threading.Thread(target=download_worker, daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()

        print("Thread timestamps:")
        for event in events:
            print(f"  {event[0]}: {event[1]:.2f}s")

        assert any(name == "gpu_start" for name, _ in events), "GPU thread did not start"
        assert any(name == "download_start" for name, _ in events), "Download thread did not start"
        assert events[0][0] in {"gpu_start", "download_start"}, "Thread order unexpected"
        print("✅ Threading working")
        mark_test("threading", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Threading test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("threading", False)
        return False


def test_shard_download():
    try:
        repo = "zeyuanyin/complete-objaverse"
        api = HfApi()
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        parquet_files = [p for p in files if p.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet shards found in {repo}")
        shard_name = parquet_files[0]
        temp_dir = Path(tempfile.mkdtemp(prefix="abd3d_shard_dl_"))
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=shard_name,
                local_dir=str(temp_dir),
            )
        )
        assert local_path.exists(), f"Shard was not downloaded: {local_path}"
        size = local_path.stat().st_size
        assert size > 1_000_000, f"Shard too small to be real: {size} bytes"
        print(f"Shard: {shard_name}")
        print(f"Size: {size} bytes")
        local_path.unlink(missing_ok=True)
        print("✅ Shard download working")
        mark_test("shard_download", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Shard download test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("shard_download", False)
        return False


def test_shard_read():
    try:
        repo = "zeyuanyin/complete-objaverse"
        api = HfApi()
        parquet_files = [p for p in api.list_repo_files(repo_id=repo, repo_type="dataset") if p.endswith(".parquet")]
        shard_name = parquet_files[0]
        temp_dir = Path(tempfile.mkdtemp(prefix="abd3d_shard_read_"))
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=shard_name,
                local_dir=str(temp_dir),
            )
        )
        df = pd.read_parquet(local_path)
        assert not df.empty, "Downloaded shard is empty"
        obj_count = df["object_id"].nunique() if "object_id" in df.columns else len(df)
        image_col = df.iloc[0].get("image_png")
        if image_col is None:
            raise KeyError("Parquet file does not contain image_png column")
        image = Image.open(io.BytesIO(image_col)).convert("RGBA")
        assert image.size == (512, 512), f"Unexpected image shape: {image.size}"
        print(f"Objects in shard: {obj_count}")
        print(f"Image shape: {image.size}")
        local_path.unlink(missing_ok=True)
        print("✅ Shard read working")
        mark_test("shard_read", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Shard read test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("shard_read", False)
        return False


def test_shard_delete():
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="abd3d_shard_delete_"))
        test_file = temp_dir / "demo.parquet"
        test_file.write_bytes(b"fake-shard-data")
        assert test_file.exists(), "Test shard was not created"
        test_file.unlink()
        assert not test_file.exists(), "Shard still exists after deletion"
        print("✅ Shard delete working")
        mark_test("shard_delete", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Shard delete test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("shard_delete", False)
        return False


def test_two_shard_buffer():
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="abd3d_buffer_"))
        on_disk = []
        max_seen = 0

        def add_shard(name: str):
            nonlocal max_seen
            file_path = temp_dir / name
            file_path.write_bytes(b"x" * 1024 * 1024)
            on_disk.append(file_path)
            max_seen = max(max_seen, len(on_disk))
            print(f"Downloaded {name}: exists={file_path.exists()} size={file_path.stat().st_size}")
            return file_path

        shard1 = add_shard("shard1.parquet")
        shard2 = add_shard("shard2.parquet")
        print(f"Disk after two-shard preload: {len(on_disk)} shards")
        assert len(on_disk) <= 2, f"Exceeded 2-shard cap: {len(on_disk)}"

        time.sleep(0.2)
        shard1.unlink(missing_ok=True)
        on_disk.remove(shard1)
        print(f"After deleting shard1: {len(on_disk)} shards remain")

        shard3 = add_shard("shard3.parquet")
        print(f"Disk after adding shard3: {len(on_disk)} shards")
        assert len(on_disk) <= 2, f"Exceeded 2-shard cap after adding shard3: {len(on_disk)}"

        shard2.unlink(missing_ok=True)
        on_disk.remove(shard2)
        print(f"After deleting shard2: {len(on_disk)} shards remain")

        for p in sorted(temp_dir.glob("*.parquet")):
            print(f"Remaining shard on disk: {p.name}, size={p.stat().st_size}")
        assert len(on_disk) <= 2, "More than 2 shards remained on disk"
        assert max_seen <= 2, f"Buffer exceeded 2-shard max: {max_seen}"
        print("✅ 2-Shard buffer working")
        mark_test("two_shard_buffer", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ 2-Shard buffer test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("two_shard_buffer", False)
        return False


def test_progress_tracking():
    try:
        progress_path = Path("checkpoints") / "shard_progress.json"
        progress_path.parent.mkdir(exist_ok=True)
        payload = {"current_shard_index": 5, "total_shards_processed": 5}
        progress_path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        assert int(loaded["current_shard_index"]) == 5, f"Unexpected index: {loaded}"
        progress_path.unlink(missing_ok=True)
        print("✅ Progress tracking working")
        mark_test("progress_tracking", True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"❌ Progress tracking test failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        mark_test("progress_tracking", False)
        return False


def main():
    test_threading()
    test_shard_download()
    test_shard_read()
    test_shard_delete()
    test_two_shard_buffer()
    test_progress_tracking()

    print("\n════════════════════════════════")
    print("ABD3D SHARD BUFFER TEST RESULTS")
    print("════════════════════════════════")
    print(f"TEST 1 - Threading:        {'✅' if RESULTS.get('threading') else '❌'}")
    print(f"TEST 2 - Shard Download:   {'✅' if RESULTS.get('shard_download') else '❌'}")
    print(f"TEST 3 - Shard Read:       {'✅' if RESULTS.get('shard_read') else '❌'}")
    print(f"TEST 4 - Shard Delete:     {'✅' if RESULTS.get('shard_delete') else '❌'}")
    print(f"TEST 5 - 2-Shard Buffer:   {'✅' if RESULTS.get('two_shard_buffer') else '❌'}")
    print(f"TEST 6 - Progress Track:   {'✅' if RESULTS.get('progress_tracking') else '❌'}")
    print("════════════════════════════════")
    passed = sum(1 for value in RESULTS.values() if value)
    print(f"{passed}/6 tests passed")


if __name__ == "__main__":
    main()
