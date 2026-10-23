"""Xuất ảnh demo (PNG) từ GTSRB để upload vào app — không train, không sửa dữ liệu gốc.

Ví dụ::

    python tests/export_demo_images.py --split val  --n 8 --out outputs/demo_images
    python tests/export_demo_images.py --split test --n 8 --out outputs/demo_images
"""

from __future__ import annotations

import argparse
import csv
import sys as _sys
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))

from src.config import TEST_GT_CSV, TEST_IMAGES_DIR  # noqa: E402
from src.data import load_cache, load_split  # noqa: E402

CLASS_NAMES = json.loads((ROOT / "configs/class_names.json").read_text(encoding="utf-8"))


def export_val(n: int, out: Path) -> list[dict]:
    cache = load_cache(64)
    split = load_split("stratified_group")
    indices = sorted(split["val"])
    labels = cache["train"]["y"]
    paths = cache["train"]["paths"]
    rows = []
    for index in indices:
        label = int(labels[index])
        if any(r["label"] == label for r in rows):
            continue  # mỗi lớp 1 ảnh cho đa dạng
        rows.append({"path": paths[index], "label": label, "split": "val"})
        if len(rows) == n:
            break
    return rows


def export_test(n: int, out: Path) -> list[dict]:
    rows = []
    with TEST_GT_CSV.open(encoding="utf-8") as fh:
        for record in csv.DictReader(fh, delimiter=";"):
            label = int(record["ClassId"])
            if any(r["label"] == label for r in rows):
                continue
            rows.append(
                {
                    "path": str((TEST_IMAGES_DIR / record["Filename"]).relative_to(ROOT)).replace("\\", "/"),
                    "label": label,
                    "split": "test",
                }
            )
            if len(rows) == n:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất ảnh GTSRB thành PNG để demo")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--n", type=int, default=8, help="số ảnh (mỗi lớp tối đa 1 ảnh)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/demo_images")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    picked = export_val(args.n, out) if args.split == "val" else export_test(args.n, out)

    manifest_path = out / "demo_manifest.csv"
    existing: list[dict] = []
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as fh:
            existing = list(csv.DictReader(fh))
    rows = [r for r in existing if r["file"] not in {f"{args.split}_{i['label']:02d}.png" for i in picked}]
    for item in picked:
        source = ROOT / item["path"]
        if not source.is_file():
            print(f"[WARN] thiếu {source}")
            continue
        target = out / f"{args.split}_{item['label']:02d}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(target, optimize=True)
        rows.append(
            {
                "file": target.name,
                "split": item["split"],
                "label": item["label"],
                "class_vi": CLASS_NAMES[str(item["label"])]["vi"],
                "source": item["path"],
            }
        )
        print(f"  {target.name:20s} lớp {item['label']:2d} — {CLASS_NAMES[str(item['label'])]['vi']}")

    fieldnames = ["file", "split", "label", "class_vi", "source"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã xuất {len(rows)} ảnh + {manifest_path.relative_to(ROOT)}")
    print("Lưu ý: ảnh val/test chỉ dùng để demo, KHÔNG dùng để chọn lại mô hình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
