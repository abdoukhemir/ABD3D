# inspect_data.py
import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from data.dataset import _load_or_fetch_shards, _download_shard

REPO = "zeyuanyin/complete-objaverse"
OUT = "inspect_out"
import os; os.makedirs(OUT, exist_ok=True)

shards = _load_or_fetch_shards(REPO)
path = _download_shard(REPO, shards[0])

pf = pq.ParquetFile(path)
print("=== SCHEMA ===")
print(pf.schema_arrow)
print("total rows:", pf.metadata.num_rows)

rows = next(pf.iter_batches(batch_size=24)).to_pylist()

print("\n=== NON-IMAGE COLUMNS (first 24 rows) ===")
for i, r in enumerate(rows):
    meta = {k: v for k, v in r.items() if not isinstance(v, (bytes, bytearray))}
    print(i, meta)

print("\n=== ALPHA CHECK on raw image_png ===")
for i in range(0, 24, 6):
    img = Image.open(io.BytesIO(rows[i]["image_png"]))
    print(f"row {i}: mode={img.mode}, size={img.size}")
    if img.mode == "RGBA":
        a = np.array(img)[..., 3]
        rgb = np.array(img)[..., :3]
        transparent = a == 0
        print(f"  alpha min/max: {a.min()}/{a.max()}, "
              f"transparent fraction: {transparent.mean():.2f}")
        if transparent.any():
            print(f"  mean RGB under transparent pixels: "
                  f"{rgb[transparent].mean(axis=0)}")
        Image.fromarray(a).save(f"{OUT}/alpha_row{i}.png")
    img.convert("RGB").save(f"{OUT}/naive_rgb_row{i}.png")  # what your code produces

print("\n=== CONTACT SHEET: 24 consecutive rows (2 rows of 12) ===")
S = 128
sheet = Image.new("RGB", (S * 12, S * 2))
for i, r in enumerate(rows):
    im = Image.open(io.BytesIO(r["image_png"])).convert("RGB").resize((S, S))
    sheet.paste(im, ((i % 12) * S, (i // 12) * S))
sheet.save(f"{OUT}/contact_sheet_rgb.png")

sheet = Image.new("RGB", (S * 12, S * 2))
for i, r in enumerate(rows):
    im = Image.open(io.BytesIO(r["nd_png"])).convert("RGB").resize((S, S))
    sheet.paste(im, ((i % 12) * S, (i // 12) * S))
sheet.save(f"{OUT}/contact_sheet_nd.png")
print(f"saved to ./{OUT}/")