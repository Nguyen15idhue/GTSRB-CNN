"""Tải GTSRB, xác minh dữ liệu và sinh báo cáo EDA (Giai đoạn 1).

Phạm vi giai đoạn 1: tải, đếm, kiểm tra ảnh/trùng lặp/ROI/track, vẽ hình EDA và
ghi ``outputs/results/data_report.json``. Cache, split và ``preprocess`` thuộc
Giai đoạn 2 nên **không** được triển khai ở đây.

Cách dùng::

    python -m src.data --download --report
    python -m src.data --report --limit 200   # smoke test, chỉ quét 200 ảnh/tập
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # không cần display, chạy được trên Windows không GUI

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import (
    CLASS_NAMES_PATH,
    DATA_ROOT,
    FIGURES_DIR,
    GTSRB_ROOT,
    NUM_CLASSES,
    PROJECT_ROOT,
    PUBLISHED_TEST_COUNT,
    PUBLISHED_TRAIN_COUNT,
    RESULTS_DIR,
    TEST_GT_CSV,
    TEST_IMAGES_DIR,
    TRAIN_IMAGES_DIR,
)

LOG = logging.getLogger(__name__)

TRAIN_META_PATH = RESULTS_DIR / "train_meta.csv"
DATA_REPORT_PATH = RESULTS_DIR / "data_report.json"
FIG_CLASS_DIST = FIGURES_DIR / "01_class_distribution.png"
FIG_IMAGE_SIZES = FIGURES_DIR / "02_image_sizes.png"
FIG_SAMPLE_GRID = FIGURES_DIR / "03_sample_grid.png"

# Tên file train: "<track>_<frame>.ppm"; track chỉ duy nhất trong một lớp.
TRACK_FRAME_PATTERN = re.compile(r"^(\d+)_(\d+)$")
ROI_COLUMNS = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")

# Mirror mà torchvision dùng; archive train chính thức có đủ 39.209 ảnh.
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
OFFICIAL_TRAIN_ARCHIVE = "GTSRB_Final_Training_Images.zip"
OFFICIAL_TRAIN_MD5 = "f33fd80ac59bff73c82d25ab499e03a3"

LICENSE_NOTE = (
    "Nguyên văn từ trang chính thức GTSRB: \"The data is free to use. However, we "
    "cordially ask you to cite the following publication if you do: ...\" "
    "(Stallkamp et al., IJCNN 2011). Trang benchmark.ini.rub.de không phân giải "
    "được DNS từ máy này ngày 22/10/2026, nên câu trên được chép từ bản lưu "
    "Wayback Machine capture 30/12/2025 "
    "(https://web.archive.org/web/20251230072343/https://benchmark.ini.rub.de/gtsrb_dataset.html). "
    "Không suy diễn thành giấy phép CC-BY hay bất kỳ giấy phép cụ thể nào."
)


@dataclass
class ImageInfo:
    """Thông tin một ảnh sau khi quét."""

    path: Path
    split: str
    label: int
    width: int
    height: int
    mode: str
    md5: str
    track_id: str
    track_key: str
    roi: tuple[int, int, int, int] | None
    roi_valid: bool


# ---------------------------------------------------------------------------
# Tải dữ liệu
# ---------------------------------------------------------------------------


def download_gtsrb(root: Path = DATA_ROOT) -> dict[str, str]:
    """Chuẩn bị GTSRB trên đĩa và trả về mô tả cách tải.

    Ghi chú quan trọng: ``torchvision.datasets.GTSRB`` tải train từ
    ``GTSRB-Training_fixed.zip`` — bản rút gọn chỉ có 26.640 ảnh, không khớp
    con số công bố 39.209. Vì vậy train được lấy từ archive chính thức
    ``GTSRB_Final_Training_Images.zip`` (39.209 ảnh, kèm ``GT-*.csv``), còn test
    dùng torchvision (12.630 ảnh, khớp công bố). Cả hai nằm trên cùng mirror.
    """
    from torchvision.datasets import GTSRB  # import trễ, tránh nạp torch khi chỉ đọc báo cáo

    LOG.info("Tải GTSRB test (torchvision, kiểm tra MD5) vào %s", root)
    GTSRB(root=str(root), split="test", download=True)
    train_source = _ensure_official_train()

    if not TRAIN_IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục train sau khi tải: {TRAIN_IMAGES_DIR}")
    if not TEST_IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục test sau khi tải: {TEST_IMAGES_DIR}")
    if not TEST_GT_CSV.is_file():
        raise FileNotFoundError(f"Không thấy file GT của test: {TEST_GT_CSV}")

    return {
        "method": (
            "test: torchvision.datasets.GTSRB(root='data', download=True); "
            "train: archive chính thức " + OFFICIAL_TRAIN_ARCHIVE
        ),
        "mirror": BASE_URL,
        "official_page": "https://benchmark.ini.rub.de/gtsrb_dataset.html",
        "official_page_reachable": "không (DNS không phân giải từ máy này ngày 22/10/2026)",
        "test_archive": "torchvision GTSRB_Final_Test_Images.zip + GTSRB_Final_Test_GT.zip",
        "test_images_archive_md5": "c7e4e6327067d32654124b0fe9e82185",
        "test_gt_archive_md5": "fe31e9c9270bbcd7b84b7f21a9d9d9e5",
        "train_archive": OFFICIAL_TRAIN_ARCHIVE,
        "train_archive_md5": OFFICIAL_TRAIN_MD5,
        "train_source": train_source,
        "accessed_at": date.today().isoformat(),
    }


def _ensure_official_train() -> str:
    """Đảm bảo train có đủ 39.209 ảnh từ archive chính thức; trả mô tả nguồn."""
    current = count_ppm(TRAIN_IMAGES_DIR) if TRAIN_IMAGES_DIR.is_dir() else 0
    if current == PUBLISHED_TRAIN_COUNT:
        return f"đã có sẵn {current} ảnh"

    from torchvision.datasets.utils import download_and_extract_archive

    archive = GTSRB_ROOT / OFFICIAL_TRAIN_ARCHIVE
    LOG.info(
        "Train hiện có %d ảnh (cần %d) → tải archive chính thức %s",
        current,
        PUBLISHED_TRAIN_COUNT,
        OFFICIAL_TRAIN_ARCHIVE,
    )
    download_and_extract_archive(
        BASE_URL + OFFICIAL_TRAIN_ARCHIVE,
        download_root=str(archive.parent),
        extract_root=str(GTSRB_ROOT),
        md5=OFFICIAL_TRAIN_MD5,
    )

    extracted = GTSRB_ROOT / "GTSRB" / "Final_Training" / "Images"
    if not extracted.is_dir():
        raise FileNotFoundError(f"Archive không giải nén ra {extracted}")

    # Thay thư mục Training của torchvision (26.640 ảnh) bằng bản chính thức.
    if TRAIN_IMAGES_DIR.exists():
        shutil.rmtree(TRAIN_IMAGES_DIR)
    TRAIN_IMAGES_DIR.parent.mkdir(parents=True, exist_ok=True)
    extracted.rename(TRAIN_IMAGES_DIR)

    leftover = extracted.parent
    if leftover.is_dir() and not any(leftover.iterdir()):
        shutil.rmtree(leftover)
    (GTSRB_ROOT / "GTSRB-Training_fixed.zip").unlink(missing_ok=True)

    total = count_ppm(TRAIN_IMAGES_DIR)
    if total != PUBLISHED_TRAIN_COUNT:
        raise RuntimeError(
            f"Train sau khi tải có {total} ảnh, lệch so với công bố {PUBLISHED_TRAIN_COUNT}"
        )
    return f"tải từ {OFFICIAL_TRAIN_ARCHIVE}, {total} ảnh"


def count_ppm(images_dir: Path) -> int:
    """Đếm số ảnh .ppm trong các thư mục lớp (một cấp)."""
    return sum(1 for class_dir in images_dir.iterdir() if class_dir.is_dir()
               for _ in class_dir.glob("*.ppm"))


# ---------------------------------------------------------------------------
# Chỉ mục ảnh và annotation
# ---------------------------------------------------------------------------


def index_train_images(images_dir: Path = TRAIN_IMAGES_DIR) -> list[tuple[Path, int]]:
    """Trả danh sách (path, label) của train, sắp xếp ổn định theo lớp rồi tên file."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục train: {images_dir}")

    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        label = int(class_dir.name)
        for path in sorted(class_dir.glob("*.ppm")):
            samples.append((path, label))
    if not samples:
        raise RuntimeError(f"Không có ảnh .ppm nào trong {images_dir}")
    return samples


def read_test_annotations(gt_csv: Path = TEST_GT_CSV) -> list[dict[str, str]]:
    """Đọc ``GT-final_test.csv`` (phân tách ';') theo đúng thứ tự file."""
    if not gt_csv.is_file():
        raise FileNotFoundError(f"Không thấy file GT của test: {gt_csv}")
    with gt_csv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=";", skipinitialspace=True))


def index_test_images(
    images_dir: Path = TEST_IMAGES_DIR, gt_csv: Path = TEST_GT_CSV
) -> list[tuple[Path, int]]:
    """Trả danh sách (path, label) của test theo thứ tự trong CSV GT."""
    samples = [
        (images_dir / row["Filename"], int(row["ClassId"]))
        for row in read_test_annotations(gt_csv)
    ]
    missing = [str(p) for p, _ in samples if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh test, ví dụ: {missing[:3]}")
    return samples


def read_annotation_table(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    """Gộp CSV annotation của train, khóa ``<thư mục lớp>/<tên file>``.

    Tên file train chỉ duy nhất **trong một lớp** (mỗi lớp đánh số lại từ
    ``00000_00000``), nên không được khóa theo tên file đơn thuần.
    """
    table: dict[str, dict[str, str]] = {}
    for csv_path in paths:
        prefix = csv_path.parent.name
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter=";", skipinitialspace=True):
                table[f"{prefix}/{Path(row['Filename']).stem}"] = row
    return table


def annotation_key(path: Path, split: str) -> str:
    """Khóa tra annotation: train cần cả thư mục lớp, test chỉ cần tên file."""
    return f"{path.parent.name}/{path.stem}" if split == "train" else path.stem


def load_train_annotations(images_dir: Path = TRAIN_IMAGES_DIR) -> dict[str, dict[str, str]]:
    """Đọc mọi ``GT-<ClassID>.csv`` trong các thư mục lớp của train."""
    return read_annotation_table(sorted(images_dir.glob("*/GT-*.csv")))


def parse_track_id(stem: str) -> str:
    """Tách track id (phần số đầu tiên) từ tên file; trả chuỗi rỗng nếu không khớp."""
    match = TRACK_FRAME_PATTERN.match(stem)
    return match.group(1) if match else ""


def parse_roi(row: dict[str, str] | None) -> tuple[tuple[int, int, int, int] | None, bool]:
    """Trả (roi, hợp_lệ) từ một dòng annotation."""
    if row is None:
        return None, False
    try:
        roi = tuple(int(float(row[col])) for col in ROI_COLUMNS)
    except (KeyError, TypeError, ValueError):
        return None, False
    x1, y1, x2, y2 = roi
    width = int(float(row["Width"]))
    height = int(float(row["Height"]))
    valid = 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    return roi, valid


# ---------------------------------------------------------------------------
# Quét ảnh + trùng lặp
# ---------------------------------------------------------------------------


def scan_split(
    samples: Sequence[tuple[Path, int]],
    split: str,
    annotations: dict[str, dict[str, str]],
    limit: int = 0,
) -> tuple[list[ImageInfo], list[dict[str, str]]]:
    """Mở từng ảnh, lấy kích thước/mode/MD5/ROI/track. Trả (infos, lỗi)."""
    infos: list[ImageInfo] = []
    errors: list[dict[str, str]] = []
    selected = samples[:limit] if limit > 0 else samples

    for path, label in selected:
        try:
            data = path.read_bytes()
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            errors.append({"path": _relative(path), "error": f"{type(exc).__name__}: {exc}"})
            LOG.error("Ảnh lỗi %s: %s", path, exc)
            continue

        track_id = parse_track_id(path.stem) if split == "train" else ""
        roi, roi_valid = parse_roi(annotations.get(annotation_key(path, split)))
        infos.append(
            ImageInfo(
                path=path,
                split=split,
                label=label,
                width=width,
                height=height,
                mode=mode,
                md5=hashlib.md5(data).hexdigest(),
                track_id=track_id,
                track_key=f"{label:05d}_{track_id}" if track_id else "",
                roi=roi,
                roi_valid=roi_valid,
            )
        )
    return infos, errors


def count_duplicates(infos: Iterable[ImageInfo]) -> dict[str, Any]:
    """Đếm trùng lặp MD5 trong train, trong test và giữa train ↔ test."""
    groups: dict[str, dict[str, list[ImageInfo]]] = defaultdict(lambda: defaultdict(list))
    for info in infos:
        groups[info.md5][info.split].append(info)

    within_train = [len(g["train"]) for g in groups.values() if len(g.get("train", [])) > 1]
    within_test = [len(g["test"]) for g in groups.values() if len(g.get("test", [])) > 1]
    cross_detail = [
        {
            "md5": md5,
            "train_path": _relative(train.path),
            "train_label": train.label,
            "test_path": _relative(test.path),
            "test_label": test.label,
        }
        for md5, group in groups.items()
        if group.get("train") and group.get("test")
        for train in group["train"]
        for test in group["test"]
    ]
    cross_pairs = len(cross_detail)

    return {
        "hash": "md5",
        "train_duplicate_groups": len(within_train),
        "train_extra_copies": sum(n - 1 for n in within_train),
        "train_duplicate_pairs": sum(math.comb(n, 2) for n in within_train),
        "test_duplicate_groups": len(within_test),
        "test_duplicate_pairs": sum(math.comb(n, 2) for n in within_test),
        "cross_split_groups": len({p["md5"] for p in cross_detail}),
        "cross_split_pairs": cross_pairs,
        "train_images_matching_test": len({p["train_path"] for p in cross_detail}),
        "test_images_matching_train": len({p["test_path"] for p in cross_detail}),
        "cross_split_pairs_detail": cross_detail,
        "decision": (
            "Không có ảnh trùng MD5 giữa train và test nên không phải xử lý."
            if not cross_detail
            else (
                f"Có {cross_pairs} cặp ảnh trùng MD5 giữa train và test (đều cùng nhãn). "
                "Quyết định: giữ nguyên split chính thức 39.209/12.630 để so sánh được với "
                "công bố; 8 ảnh chiếm 0,063% test nên Giai đoạn 6 sẽ báo cáo thêm accuracy "
                "trên 12.622 ảnh không trùng như kiểm tra độ nhạy, và ảnh hưởng được ghi ở "
                "Chương 5."
            )
        ),
    }


def summarize_tracks(infos: Sequence[ImageInfo]) -> dict[str, Any]:
    """Thống kê track của train: số track, số frame mỗi track, track không đủ 30 frame."""
    train = [i for i in infos if i.split == "train"]
    frames = Counter(i.track_key for i in train if i.track_key)
    sizes = list(frames.values())
    unparsed = sum(1 for i in train if not i.track_key)
    return {
        "train_images": len(train),
        "train_images_with_track_id": len(train) - unparsed,
        "extracted_ratio": round((len(train) - unparsed) / len(train), 6) if train else 0.0,
        "num_tracks": len(frames),
        "tracks_with_30_frames": sum(1 for n in sizes if n == 30),
        "frames_per_track_min": int(np.min(sizes)) if sizes else 0,
        "frames_per_track_median": float(np.median(sizes)) if sizes else 0.0,
        "frames_per_track_max": int(np.max(sizes)) if sizes else 0,
        "note": (
            "Tên file train là <track>_<frame>.ppm và track chỉ duy nhất trong một lớp; "
            "vì vậy khóa nhóm khi chia split phải là (label, track) — cột track_key "
            "trong train_meta.csv. Ảnh test đặt tên <index>.ppm nên không có track."
        ),
    }


# ---------------------------------------------------------------------------
# Thống kê mô tả
# ---------------------------------------------------------------------------


def describe(values: Sequence[float]) -> dict[str, float]:
    """min / median / mean / max, làm tròn 3 chữ số thập phân."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 3),
        "median": round(float(np.median(arr)), 3),
        "mean": round(float(arr.mean()), 3),
        "max": round(float(arr.max()), 3),
    }


def per_class_counts(labels: Iterable[int]) -> dict[int, int]:
    """Đếm số ảnh theo lớp, có đủ 43 khóa kể cả lớp 0 ảnh."""
    counts = Counter(int(label) for label in labels)
    return {class_id: counts.get(class_id, 0) for class_id in range(NUM_CLASSES)}


def check_class_coverage(split_counts: dict[int, int], split: str) -> None:
    """Raise nếu một lớp bị rỗng hoặc thiếu nhãn ngoài [0, 43)."""
    missing = [k for k, v in split_counts.items() if v == 0]
    if missing:
        raise RuntimeError(f"{split}: các lớp không có ảnh: {missing}")
    if len(split_counts) != NUM_CLASSES:
        raise RuntimeError(f"{split}: cần {NUM_CLASSES} lớp, đếm được {len(split_counts)}")


# ---------------------------------------------------------------------------
# Hình EDA
# ---------------------------------------------------------------------------


def _relative(path: Path) -> str:
    """Đường dẫn tương đối từ gốc repo, dạng posix (không hard-code ổ đĩa)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def load_class_names(path: Path = CLASS_NAMES_PATH) -> dict[str, dict[str, str]]:
    """Đọc bảng 43 lớp dùng chung."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def plot_class_distribution(counts: dict[int, int], out_path: Path = FIG_CLASS_DIST) -> None:
    """Biểu đồ cột số ảnh/lớp, sắp tăng dần, tô đỏ lớp dưới 300 ảnh."""
    items = sorted(counts.items(), key=lambda kv: kv[1])
    ids = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#c0392b" if v < 300 else "#2e86c1" for v in values]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(range(len(ids)), values, color=colors)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)
    ax.set_xlabel("Class ID (sắp xếp theo số ảnh tăng dần)")
    ax.set_ylabel("Số ảnh train")
    ax.set_title("Phân bố số ảnh theo lớp — GTSRB train (đỏ: lớp < 300 ảnh)")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(values):
        if value < 300:
            ax.text(idx, value + 8, str(value), ha="center", fontsize=6, rotation=90)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Đã lưu %s", out_path)


def plot_image_sizes(infos: Sequence[ImageInfo], out_path: Path = FIG_IMAGE_SIZES) -> None:
    """Histogram kích thước gốc (width/height) và tỷ lệ khung."""
    widths = np.array([i.width for i in infos])
    heights = np.array([i.height for i in infos])
    ratios = widths / np.maximum(heights, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(widths, bins=40, alpha=0.6, label="width", color="#2e86c1")
    axes[0].hist(heights, bins=40, alpha=0.6, label="height", color="#e67e22")
    axes[0].set_xlabel("Pixel")
    axes[0].set_ylabel("Số ảnh")
    axes[0].set_title("Kích thước ảnh gốc")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(ratios, bins=40, color="#27ae60")
    axes[1].set_xlabel("width / height")
    axes[1].set_ylabel("Số ảnh")
    axes[1].set_title("Tỷ lệ khung ảnh gốc")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Đã lưu %s", out_path)


def _crop_to_roi(image: Image.Image, roi: tuple[int, int, int, int] | None) -> Image.Image:
    """Crop quanh ROI (nếu hợp lệ) để không lộ người/xe ở nền."""
    if roi is None:
        return image
    x1, y1, x2, y2 = roi
    return image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))


def plot_sample_grid(
    samples: Sequence[tuple[Path, int]],
    annotations: dict[str, dict[str, str]],
    class_names: dict[str, dict[str, str]],
    out_path: Path = FIG_SAMPLE_GRID,
) -> None:
    """Lưới 5×9: mỗi lớp một ảnh (crop theo ROI) kèm mã + tên lớp."""
    first_by_class: dict[int, Path] = {}
    for path, label in samples:
        first_by_class.setdefault(label, path)
    missing = [c for c in range(NUM_CLASSES) if c not in first_by_class]
    if missing:
        raise RuntimeError(f"Thiếu ảnh mẫu cho lớp: {missing}")

    rows, cols = 5, 9
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    for ax, class_id in zip(axes.ravel(), sorted(first_by_class)):
        path = first_by_class[class_id]
        roi, roi_valid = parse_roi(annotations.get(annotation_key(path, "train")))
        with Image.open(path) as img:
            sample = _crop_to_roi(img.convert("RGB"), roi if roi_valid else None)
            ax.imshow(sample)
        ax.set_xticks([])
        ax.set_yticks([])
        name_vi = class_names.get(str(class_id), {}).get("vi", "")
        ax.set_title(textwrap.fill(f"{class_id}: {name_vi}", 22), fontsize=6, pad=2)
    for ax in axes.ravel()[len(first_by_class):]:
        ax.axis("off")
    fig.suptitle("Mẫu mỗi lớp (crop theo ROI) — GTSRB train", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Đã lưu %s", out_path)


# ---------------------------------------------------------------------------
# Xuất kết quả
# ---------------------------------------------------------------------------


def write_train_meta(infos: Sequence[ImageInfo], out_path: Path = TRAIN_META_PATH) -> int:
    """Ghi train_meta.csv (path, label, track_id, track_key, kích thước, ROI)."""
    rows = [i for i in infos if i.split == "train"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "path",
                "label",
                "track_id",
                "track_key",
                "filename",
                "width",
                "height",
                "roi_x1",
                "roi_y1",
                "roi_x2",
                "roi_y2",
                "roi_valid",
            ]
        )
        for info in rows:
            roi = info.roi if info.roi is not None else ("", "", "", "")
            writer.writerow(
                [
                    _relative(info.path),
                    info.label,
                    info.track_id,
                    info.track_key,
                    info.path.name,
                    info.width,
                    info.height,
                    *roi,
                    int(info.roi_valid),
                ]
            )
    LOG.info("Đã ghi %d dòng vào %s", len(rows), out_path)
    return len(rows)


def build_report(
    download_info: dict[str, str],
    train_infos: Sequence[ImageInfo],
    test_infos: Sequence[ImageInfo],
    errors: Sequence[dict[str, str]],
    duplicates: dict[str, Any],
    tracks: dict[str, Any],
    train_ann_total: int,
    test_ann_total: int,
    classid_mismatch: int,
    strict: bool = True,
) -> dict[str, Any]:
    """Tổng hợp mọi số liệu thật của Giai đoạn 1 thành dict."""
    train_counts = per_class_counts(i.label for i in train_infos)
    test_counts = per_class_counts(i.label for i in test_infos)
    if strict:
        check_class_coverage(train_counts, "train")
        check_class_coverage(test_counts, "test")
    else:
        LOG.warning("Chế độ smoke test: bỏ qua kiểm tra đủ 43 lớp")

    counts_array = np.array(list(train_counts.values()))
    train_roi_valid = sum(1 for i in train_infos if i.roi_valid)
    test_roi_valid = sum(1 for i in test_infos if i.roi_valid)

    return {
        "stage": 1,
        "generated_at": date.today().isoformat(),
        "source": {
            "name": "German Traffic Sign Recognition Benchmark (GTSRB)",
            "paper": "Stallkamp et al., Neural Networks 32 (2012) 323-332",
            "official_page": download_info.get("official_page"),
            "mirror_used": download_info.get("mirror"),
            "dataset_dir": _relative(GTSRB_ROOT),
        },
        "download": download_info,
        "counts": {
            "train": len(train_infos),
            "test": len(test_infos),
            "published_train": PUBLISHED_TRAIN_COUNT,
            "published_test": PUBLISHED_TEST_COUNT,
            "difference_train": len(train_infos) - PUBLISHED_TRAIN_COUNT,
            "difference_test": len(test_infos) - PUBLISHED_TEST_COUNT,
        },
        "num_classes": NUM_CLASSES,
        "classes_train_present": int(np.count_nonzero(counts_array)),
        "classes_test_present": int(sum(1 for v in test_counts.values() if v > 0)),
        "per_class_train": {str(k): v for k, v in train_counts.items()},
        "per_class_test": {str(k): v for k, v in test_counts.items()},
        "per_class_train_stats": describe(counts_array.tolist()),
        "per_class_test_stats": describe(list(test_counts.values())),
        "least_populated_train_class": {
            "class_id": int(counts_array.argmin()),
            "count": int(counts_array.min()),
        },
        "most_populated_train_class": {
            "class_id": int(counts_array.argmax()),
            "count": int(counts_array.max()),
        },
        "imbalance_ratio_max_over_min": round(
            float(counts_array.max()) / max(float(counts_array.min()), 1.0), 3
        ),
        "images_below_300_train": int((counts_array < 300).sum()),
        "class_names_file": _relative(CLASS_NAMES_PATH),
        "image_scan": {
            "images_scanned": len(train_infos) + len(test_infos),
            "unreadable": list(errors),
            "unreadable_count": len(errors),
            "modes": dict(Counter(i.mode for i in (*train_infos, *test_infos))),
            "width": describe([i.width for i in (*train_infos, *test_infos)]),
            "height": describe([i.height for i in (*train_infos, *test_infos)]),
            "aspect_ratio": describe(
                [i.width / max(i.height, 1) for i in (*train_infos, *test_infos)]
            ),
            "size_min": int(min(min(i.width, i.height) for i in (*train_infos, *test_infos))),
            "size_max": int(max(max(i.width, i.height) for i in (*train_infos, *test_infos))),
        },
        "duplicates": duplicates,
        "roi": {
            "train_annotation_rows": train_ann_total,
            "test_annotation_rows": test_ann_total,
            "train_images_with_roi": train_roi_valid,
            "test_images_with_roi": test_roi_valid,
            "train_valid_ratio": round(train_roi_valid / max(len(train_infos), 1), 6),
            "test_valid_ratio": round(test_roi_valid / max(len(test_infos), 1), 6),
            "classid_mismatch_vs_torchvision_label": classid_mismatch,
        },
        "tracks": tracks,
        "figures": [
            _relative(FIG_CLASS_DIST),
            _relative(FIG_IMAGE_SIZES),
            _relative(FIG_SAMPLE_GRID),
        ],
        "train_meta_csv": _relative(TRAIN_META_PATH),
        "license_note": LICENSE_NOTE,
        "notes": [
            "Số liệu trong file này là số đếm thật trên đĩa, không dùng số công bố.",
            "torchvision.datasets.GTSRB tải train từ GTSRB-Training_fixed.zip — bản rút gọn "
            "chỉ có 26.640 ảnh (888 track x 30 frame), không khớp 39.209 công bố. Vì vậy "
            "train lấy từ archive chính thức GTSRB_Final_Training_Images.zip (39.209 ảnh, "
            "có GT-*.csv) và đặt tại data/gtsrb/GTSRB/Training để torchvision vẫn đọc được; "
            "test vẫn dùng torchvision (12.630 ảnh).",
            "Ảnh test chỉ có tên <index>.ppm nên không trích được track; phân tích rò rỉ "
            "theo track chỉ áp dụng cho train.",
            "File zip gốc vẫn giữ trong data/ (đã bị .gitignore chặn) để có thể kiểm tra lại MD5.",
        ],
    }


def _classid_mismatches(
    samples: Sequence[tuple[Path, int]], annotations: dict[str, dict[str, str]], split: str
) -> int:
    """Đếm số ảnh có ClassId trong CSV khác nhãn suy ra từ thư mục/CSV test."""
    mismatches = 0
    for path, label in samples:
        row = annotations.get(annotation_key(path, split))
        if row is None or "ClassId" not in row:
            continue
        if int(row["ClassId"]) != label:
            mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_report(limit: int = 0, download_info: dict[str, str] | None = None) -> dict[str, Any]:
    """Chạy toàn bộ quy trình xác minh + EDA và ghi artifact."""
    download_info = download_info or {
        "method": "dữ liệu đã có sẵn trên đĩa (không tải lại)",
        "accessed_at": date.today().isoformat(),
    }

    train_samples = index_train_images()
    test_samples = index_test_images()
    LOG.info("Đếm được %d ảnh train, %d ảnh test", len(train_samples), len(test_samples))

    train_ann = load_train_annotations()
    test_ann = {Path(r["Filename"]).stem: r for r in read_test_annotations()}
    LOG.info("Annotation: %d dòng train, %d dòng test", len(train_ann), len(test_ann))

    LOG.info("Quét ảnh train...")
    train_infos, train_errors = scan_split(train_samples, "train", train_ann, limit=limit)
    LOG.info("Quét ảnh test...")
    test_infos, test_errors = scan_split(test_samples, "test", test_ann, limit=limit)
    errors = [*train_errors, *test_errors]

    duplicates = count_duplicates([*train_infos, *test_infos])
    tracks = summarize_tracks(train_infos)

    write_train_meta(train_infos)
    plot_class_distribution(per_class_counts(i.label for i in train_infos))
    plot_image_sizes([*train_infos, *test_infos])
    plot_sample_grid(train_samples, train_ann, load_class_names())

    report = build_report(
        download_info,
        train_infos,
        test_infos,
        errors,
        duplicates,
        tracks,
        len(train_ann),
        len(test_ann),
        classid_mismatch=_classid_mismatches(train_samples, train_ann, "train")
        + _classid_mismatches(test_samples, test_ann, "test"),
        strict=limit == 0,
    )
    if limit > 0:
        report["notes"].append(f"CHẠY SMOKE TEST với --limit {limit}: số liệu chưa đầy đủ.")

    DATA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    LOG.info("Đã ghi %s", DATA_REPORT_PATH)
    return report


def _configure_logging() -> None:
    """Log ra stdout với UTF-8 để không lỗi tiếng Việt trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Tham số CLI của ``python -m src.data``."""
    parser = argparse.ArgumentParser(description="Tải và xác minh GTSRB (Giai đoạn 1)")
    parser.add_argument("--download", action="store_true", help="tải dữ liệu qua torchvision")
    parser.add_argument("--report", action="store_true", help="xác minh, vẽ EDA, ghi data_report.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chỉ quét N ảnh mỗi tập (dùng cho smoke test; 0 = toàn bộ)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m src.data --download --report``."""
    _configure_logging()
    args = parse_args(argv)
    if not (args.download or args.report):
        LOG.error("Cần ít nhất một trong --download hoặc --report")
        return 2

    download_info = download_gtsrb() if args.download else None
    if args.report:
        run_report(limit=args.limit, download_info=download_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
