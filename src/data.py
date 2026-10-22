"""Tải GTSRB, xác minh dữ liệu, sinh báo cáo EDA, cache, chia tập và tiền xử lý.

Giai đoạn 1: tải, đếm, kiểm tra ảnh/trùng lặp/ROI/track, vẽ hình EDA, ghi
``outputs/results/data_report.json``.

Giai đoạn 2: cache ``.npy`` (uint8, 64×64), chia ``split_random``/``split_group``,
tính mean/std trên train, hàm :func:`preprocess` **duy nhất** dùng chung cho
train/eval/demo, dataloader có augmentation chỉ ở train.

Cách dùng::

    python -m src.data --download --report
    python -m src.data --prepare        # cache + split + mean/std + hình augmentation
    python -m src.data --verify         # chạy checklist Giai đoạn 2
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
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # không cần display, chạy được trên Windows không GUI

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import (
    CACHE_DIR,
    CLASS_NAMES_PATH,
    DATA_ROOT,
    DEFAULT_SEED,
    FIGURES_DIR,
    GTSRB_ROOT,
    IMG_SIZE,
    NUM_CLASSES,
    PROJECT_ROOT,
    PUBLISHED_TEST_COUNT,
    PUBLISHED_TRAIN_COUNT,
    RESULTS_DIR,
    SPLIT_FILES,
    STATS_PATH,
    TEST_GT_CSV,
    TEST_IMAGES_DIR,
    TRAIN_IMAGES_DIR,
    load_config,
    load_yaml,
    seed_worker,
)

LOG = logging.getLogger(__name__)

TRAIN_META_PATH = RESULTS_DIR / "train_meta.csv"
EXCLUDED_TRAIN_PATH = RESULTS_DIR / "excluded_train_duplicates.csv"
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
    excluded_from_train = sorted({p["train_path"] for p in cross_detail})

    return {
        "hash": "md5",
        "train_duplicate_groups": len(within_train),
        "train_extra_copies": sum(n - 1 for n in within_train),
        "train_duplicate_pairs": sum(math.comb(n, 2) for n in within_train),
        "test_duplicate_groups": len(within_test),
        "test_duplicate_pairs": sum(math.comb(n, 2) for n in within_test),
        "cross_split_groups": len({p["md5"] for p in cross_detail}),
        "cross_split_pairs": cross_pairs,
        "train_images_matching_test": len(excluded_from_train),
        "test_images_matching_train": len({p["test_path"] for p in cross_detail}),
        "cross_split_pairs_detail": cross_detail,
        "excluded_from_train": excluded_from_train,
        "excluded_train_count": len(excluded_from_train),
        "excluded_list_file": _relative(EXCLUDED_TRAIN_PATH),
        "train_pool_after_exclusion": None,  # điền trong build_report vì cần tổng số ảnh train
        "decision": (
            "Không có ảnh trùng MD5 giữa train và test nên không phải loại ảnh nào."
            if not cross_detail
            else (
                f"Có {cross_pairs} cặp ảnh trùng MD5 giữa train và test (đều cùng nhãn). "
                f"Quyết định: LOẠI {len(excluded_from_train)} ảnh train này khỏi tập huấn luyện "
                "để test hoàn toàn độc lập. Không xóa file trên đĩa (giữ nguyên 39.209 ảnh để "
                "số đếm khớp công bố); danh sách lưu ở outputs/results/"
                "excluded_train_duplicates.csv và đánh dấu usable_for_train=0 trong "
                "train_meta.csv. Giai đoạn 2 chỉ chia split trên các ảnh usable_for_train=1."
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


def write_train_meta(
    infos: Sequence[ImageInfo],
    excluded: Collection[str] = (),
    out_path: Path = TRAIN_META_PATH,
) -> int:
    """Ghi train_meta.csv (path, label, track, kích thước, ROI, usable_for_train)."""
    rows = [i for i in infos if i.split == "train"]
    excluded_set = set(excluded)
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
                "usable_for_train",
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
                    int(_relative(info.path) not in excluded_set),
                ]
            )
    LOG.info("Đã ghi %d dòng vào %s", len(rows), out_path)
    return len(rows)


def write_excluded_list(
    pairs: Sequence[dict[str, Any]], out_path: Path = EXCLUDED_TRAIN_PATH
) -> int:
    """Ghi danh sách ảnh train bị loại vì trùng MD5 với test (bằng chứng để tái lập)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["md5", "train_path", "train_label", "test_path", "test_label", "reason"])
        for pair in pairs:
            writer.writerow(
                [
                    pair["md5"],
                    pair["train_path"],
                    pair["train_label"],
                    pair["test_path"],
                    pair["test_label"],
                    "trùng MD5 với ảnh test",
                ]
            )
    LOG.info("Đã ghi %d cặp trùng bị loại khỏi train vào %s", len(pairs), out_path)
    return len(pairs)


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
    excluded_count = int(duplicates.get("excluded_train_count", 0))
    duplicates["train_pool_after_exclusion"] = len(train_infos) - excluded_count

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
            "train_pool_after_exclusion": len(train_infos) - excluded_count,
            "excluded_duplicates": excluded_count,
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
            f"Đã loại {excluded_count} ảnh train trùng MD5 với test khỏi tập huấn luyện; "
            "danh sách ở outputs/results/excluded_train_duplicates.csv, cột "
            "usable_for_train trong train_meta.csv đánh dấu 0. Số ảnh train dùng để chia "
            f"split là {len(train_infos) - excluded_count}.",
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
# Giai đoạn 2 — cache, chia tập, tiền xử lý, dataloader
# ---------------------------------------------------------------------------

CACHE_META_PATH = CACHE_DIR / "cache_meta.json"
FIG_AUGMENTATION = FIGURES_DIR / "04_augmentation.png"
PHASE2_CHECK_PATH = RESULTS_DIR / "phase2_check.json"

# 39.209 ảnh train trên đĩa trừ 8 ảnh trùng MD5 với test (Giai đoạn 1).
TRAIN_POOL_COUNT = PUBLISHED_TRAIN_COUNT - 8

CACHE_KEYS = {
    "train": ("X_train.npy", "y_train.npy", "roi_train.npy", "paths_train.txt"),
    "test": ("X_test.npy", "y_test.npy", "roi_test.npy", "paths_test.txt"),
}


@dataclass
class PoolMeta:
    """Mô tả một tập ảnh đã index: đường dẫn tương đối, nhãn, ROI, khóa track."""

    split: str
    paths: list[str]
    labels: np.ndarray
    rois: np.ndarray
    track_keys: list[str]


def _read_train_meta(path: Path = TRAIN_META_PATH) -> list[dict[str, str]]:
    """Đọc ``train_meta.csv`` do Giai đoạn 1 sinh ra."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không thấy {path}. Chạy trước: python -m src.data --report"
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _track_key(path: str, label: int) -> str:
    """Khóa nhóm track: ``<label>_<track>``; ảnh không khớp mẫu thì dùng path riêng."""
    stem = Path(path).stem
    track = parse_track_id(stem)
    return f"{label:05d}_{track}" if track else f"{label:05d}_path_{stem}"


def load_train_pool(meta_path: Path = TRAIN_META_PATH) -> PoolMeta:
    """Tập train dùng để chia split: chỉ các ảnh ``usable_for_train=1`` (39.201)."""
    rows = [r for r in _read_train_meta(meta_path) if int(r["usable_for_train"]) == 1]
    if not rows:
        raise RuntimeError(f"{meta_path}: không có dòng nào usable_for_train=1")
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    rois = np.array(
        [[int(r["roi_x1"]), int(r["roi_y1"]), int(r["roi_x2"]), int(r["roi_y2"])]
         if int(r["roi_valid"]) == 1 else [-1, -1, -1, -1] for r in rows],
        dtype=np.int32,
    )
    paths = [r["path"] for r in rows]
    return PoolMeta(
        split="train",
        paths=paths,
        labels=labels,
        rois=rois,
        track_keys=[_track_key(r["path"], int(r["label"])) for r in rows],
    )


def load_test_pool(images_dir: Path = TEST_IMAGES_DIR, gt_csv: Path = TEST_GT_CSV) -> PoolMeta:
    """Tập test chính thức (12.630 ảnh) + ROI từ ``GT-final_test.csv``."""
    annotations = {Path(r["Filename"]).stem: r for r in read_test_annotations(gt_csv)}
    samples = index_test_images(images_dir, gt_csv)
    paths = [_relative(p) for p, _ in samples]
    labels = np.array([label for _, label in samples], dtype=np.int64)
    rois = []
    for path, _ in samples:
        roi, valid = parse_roi(annotations.get(path.stem))
        rois.append(list(roi) if valid and roi is not None else [-1, -1, -1, -1])
    return PoolMeta(
        split="test",
        paths=paths,
        labels=labels,
        rois=np.array(rois, dtype=np.int32),
        track_keys=[],
    )


def _load_resized(path: Path, size: int) -> np.ndarray:
    """Đọc ảnh, chuyển RGB, resize bilinear về ``size×size``, trả ``uint8`` HWC."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def build_cache(size: int = IMG_SIZE, force: bool = False) -> dict[str, Any]:
    """Resize toàn bộ ảnh một lần và lưu ``.npy`` (uint8) để epoch chạy nhanh."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = CACHE_META_PATH
    if meta_file.is_file() and not force:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("image_size") == size and all(
            (CACHE_DIR / name).is_file() for names in CACHE_KEYS.values() for name in names
        ):
            LOG.info("Dùng lại cache sẵn có tại %s (image_size=%d)", CACHE_DIR, size)
            return meta

    started = time.perf_counter()
    pools = {"train": load_train_pool(), "test": load_test_pool()}
    meta: dict[str, Any] = {"image_size": size, "created_at": date.today().isoformat(), "splits": {}}

    for split, pool in pools.items():
        n = len(pool.paths)
        x = np.empty((n, size, size, 3), dtype=np.uint8)
        LOG.info("Dựng cache %s: %d ảnh...", split, n)
        for i, rel in enumerate(pool.paths):
            x[i] = _load_resized(PROJECT_ROOT / rel, size)
            if (i + 1) % 5000 == 0:
                LOG.info("  %s: %d/%d", split, i + 1, n)

        x_name, y_name, roi_name, path_name = CACHE_KEYS[split]
        np.save(CACHE_DIR / x_name, x)
        np.save(CACHE_DIR / y_name, pool.labels.astype(np.int16))
        np.save(CACHE_DIR / roi_name, pool.rois)
        (CACHE_DIR / path_name).write_text("\n".join(pool.paths), encoding="utf-8")
        meta["splits"][split] = {
            "count": n,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "files": {
                name: {
                    "size_mb": round((CACHE_DIR / name).stat().st_size / 1024**2, 2),
                }
                for name in (x_name, y_name, roi_name, path_name)
            },
        }

    meta["elapsed_s"] = round(time.perf_counter() - started, 2)
    meta["total_size_mb"] = round(
        sum(f.stat().st_size for names in CACHE_KEYS.values()
            for name in names for f in [CACHE_DIR / name]) / 1024**2,
        2,
    )
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info(
        "Xong cache trong %.1f s, tổng %.1f MB → %s",
        meta["elapsed_s"],
        meta["total_size_mb"],
        CACHE_DIR,
    )
    return meta


def load_cache(size: int = IMG_SIZE) -> dict[str, Any]:
    """Đọc cache đã dựng; raise nếu thiếu hoặc sai shape/dtype."""
    cache: dict[str, Any] = {}
    for split, (x_name, y_name, roi_name, path_name) in CACHE_KEYS.items():
        for name in (x_name, y_name, roi_name, path_name):
            if not (CACHE_DIR / name).is_file():
                raise FileNotFoundError(
                    f"Thiếu {CACHE_DIR / name}. Chạy trước: python -m src.data --prepare"
                )
        x = np.load(CACHE_DIR / x_name)
        y = np.load(CACHE_DIR / y_name)
        if x.ndim != 4 or x.shape[1:] != (size, size, 3) or x.dtype != np.uint8:
            raise ValueError(
                f"{x_name} có shape {x.shape}, dtype {x.dtype}; cần (N, {size}, {size}, 3) uint8"
            )
        if len(y) != len(x):
            raise ValueError(f"{y_name} có {len(y)} nhãn nhưng {x_name} có {len(x)} ảnh")
        cache[split] = {
            "x": x,
            "y": y.astype(np.int64),
            "roi": np.load(CACHE_DIR / roi_name),
            "paths": (CACHE_DIR / path_name).read_text(encoding="utf-8").splitlines(),
        }
    return cache


def make_splits(
    labels: np.ndarray,
    tracks: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
    mode: str = "random",
) -> dict[str, list[int]]:
    """Chia index train/val.

    ``mode='random'``: stratified theo nhãn (train_test_split).
    ``mode='group'``: GroupShuffleSplit theo ``track`` để val không chứa frame
    cùng track với train (đo mức lạc quan do rò rỉ frame).
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if mode == "random":
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, stratify=labels
        )
    elif mode == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, groups=np.asarray(tracks)))
    else:
        raise ValueError(f"mode phải là 'random' hoặc 'group', nhận được {mode!r}")
    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
    }


def _split_summary(labels: np.ndarray, tracks: Sequence[str], split: dict[str, list[int]]) -> dict[str, Any]:
    """Số liệu kiểm chứng một bộ split: số ảnh, đủ lớp, lệch tỷ lệ, track rời nhau."""
    train_idx = np.asarray(split["train"])
    val_idx = np.asarray(split["val"])
    train_counts = per_class_counts(labels[train_idx])
    val_counts = per_class_counts(labels[val_idx])
    ratios = {
        class_id: {
            "train_ratio": round(train_counts[class_id] / max(len(train_idx), 1), 6),
            "val_ratio": round(val_counts[class_id] / max(len(val_idx), 1), 6),
        }
        for class_id in range(NUM_CLASSES)
    }
    max_ratio_gap = max(
        abs(v["train_ratio"] - v["val_ratio"]) for v in ratios.values()
    )
    track_array = np.asarray(tracks)
    train_tracks = set(track_array[train_idx]) if len(track_array) else set()
    val_tracks = set(track_array[val_idx]) if len(track_array) else set()
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "classes_train": len([c for c in train_counts.values() if c > 0]),
        "classes_val": len([c for c in val_counts.values() if c > 0]),
        "classes_missing_val": [c for c, v in val_counts.items() if v == 0],
        "min_class_val": {
            "class_id": int(min(val_counts, key=lambda k: val_counts[k])),
            "count": int(min(val_counts.values())),
        },
        "max_class_ratio_gap": round(float(max_ratio_gap), 6),
        "overlap_index": int(len(set(train_idx.tolist()) & set(val_idx.tolist()))),
        "n_tracks_train": len(train_tracks),
        "n_tracks_val": len(val_tracks),
        "overlap_track": len(train_tracks & val_tracks),
        "per_class": {
            str(c): {"train": train_counts[c], "val": val_counts[c]} for c in range(NUM_CLASSES)
        },
    }


def build_splits(
    labels: np.ndarray,
    tracks: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> dict[str, Any]:
    """Sinh và lưu cả hai bộ split (random, group) ra JSON để tái sử dụng."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"seed": seed, "val_ratio": val_ratio, "num_images": int(len(labels)), "splits": {}}
    for mode, out_path in SPLIT_FILES.items():
        if out_path.is_file() and not force:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            LOG.info("Dùng lại %s (%d train / %d val)", out_path.name, len(payload["train"]), len(payload["val"]))
        else:
            indices = make_splits(labels, tracks, val_ratio=val_ratio, seed=seed, mode=mode)
            payload = {
                "mode": mode,
                "seed": seed,
                "val_ratio": val_ratio,
                "num_images": int(len(labels)),
                "train": indices["train"],
                "val": indices["val"],
                "summary": _split_summary(labels, tracks, indices),
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            LOG.info(
                "Đã ghi %s: %d train / %d val", out_path.name, len(payload["train"]), len(payload["val"])
            )
        result["splits"][mode] = {
            "file": _relative(out_path),
            "n_train": len(payload["train"]),
            "n_val": len(payload["val"]),
            "summary": payload["summary"],
        }
    return result


def load_split(mode: str = "random") -> dict[str, Any]:
    """Đọc lại bộ split đã lưu (không sinh lại để mọi run so sánh được)."""
    if mode not in SPLIT_FILES:
        raise ValueError(f"split phải thuộc {list(SPLIT_FILES)}, nhận được {mode!r}")
    path = SPLIT_FILES[mode]
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python -m src.data --prepare")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_stats(
    images: np.ndarray, indices: Sequence[int] | None = None
) -> dict[str, Any]:
    """mean/std từng kênh (thang [0,1]) tính **chỉ trên train split**."""
    idx = np.arange(len(images)) if indices is None else np.asarray(indices)
    total = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    for chunk in np.array_split(idx, max(1, len(idx) // 5000 or 1)):
        block = images[chunk].astype(np.float32) / 255.0
        flat = block.reshape(-1, 3).astype(np.float64)
        channel_sum += flat.sum(axis=0)
        channel_sq += (flat * flat).sum(axis=0)
        total += flat.shape[0]
    mean = channel_sum / total
    var = channel_sq / total - mean * mean
    std = np.sqrt(np.clip(var, 1e-12, None))
    return {
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "num_images": int(len(idx)),
        "num_pixels": int(total),
    }


def load_stats(path: Path = STATS_PATH) -> dict[str, Any]:
    """Đọc mean/std đã lưu; raise nếu chưa chạy ``--prepare``."""
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python -m src.data --prepare")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_rgb_array(image: Image.Image | np.ndarray | str | Path, size: int) -> np.ndarray:
    """Chuẩn hóa đầu vào về mảng RGB uint8 kích thước ``size×size``."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[:2] != (size, size):
            pil = Image.fromarray(array).convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(pil, dtype=np.uint8)
        return array
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    raise TypeError(f"Không hỗ trợ kiểu ảnh {type(image)!r}")


def _array_to_tensor(array: np.ndarray, mean: Sequence[float], std: Sequence[float]):
    """uint8 HWC → tensor (3,H,W) đã chuẩn hóa; dùng chung cho train/eval/demo."""
    import torch

    # astype tạo bản sao writable (mảng do PIL trả về là read-only) và đổi sang float32.
    tensor = torch.from_numpy(array.astype(np.float32)).div_(255.0).permute(2, 0, 1)
    mean_t = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def preprocess(
    image: Image.Image | np.ndarray | str | Path,
    size: int = IMG_SIZE,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    """Hàm tiền xử lý **duy nhất** dùng cho train, eval và demo.

    Args:
        image: PIL image, ndarray HWC, hoặc đường dẫn file.
        size: cạnh ảnh đầu ra (mặc định 64).
        mean/std: nếu None thì lấy từ ``outputs/results/data_stats.json``.

    Returns:
        Float tensor shape ``(3, size, size)``, chuẩn hóa theo mean/std của train.
    """
    if mean is None or std is None:
        stats = load_stats()
        mean = mean or stats["mean"]
        std = std or stats["std"]
    return _array_to_tensor(_to_rgb_array(image, size), mean, std)


def build_train_transform(cfg: Mapping[str, Any]):
    """Augmentation **chỉ cho train**; cấm lật ảnh (đổi nghĩa biển báo)."""
    from torchvision import transforms

    aug = cfg["augmentation"]
    data = cfg["data"]
    operations: list[Any] = []
    if aug.get("enabled"):
        operations.append(
            transforms.RandomAffine(
                degrees=float(aug["rotation"]),
                translate=(float(aug["translate"]), float(aug["translate"])),
                scale=(float(aug["scale_min"]), float(aug["scale_max"])),
                shear=float(aug["shear"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
        operations.append(
            transforms.ColorJitter(
                brightness=float(aug["brightness"]),
                contrast=float(aug["contrast"]),
                saturation=float(aug["saturation"]),
                hue=float(aug["hue"]),
            )
        )
        operations.append(
            transforms.RandomPerspective(
                distortion_scale=float(aug["perspective"]),
                p=float(aug["perspective_p"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(data["mean"]), std=list(data["std"])),
        ]
    )
    transform = transforms.Compose(operations)
    _assert_no_flip(transform)
    return transform


def _assert_no_flip(transform: Any) -> None:
    """Chặn cấu hình augmentation có lật ảnh hoặc xoay 90/180 độ."""
    text = repr(transform)
    forbidden = [name for name in ("Flip", "RandomRotation") if name in text]
    if forbidden:
        raise ValueError(
            f"Augmentation chứa phép biến đổi bị cấm {forbidden}: làm đổi nghĩa biển báo. "
            f"Transform hiện tại: {text}"
        )


class GTSRBTensorDataset:
    """Dataset đọc từ cache uint8, trả tensor đã chuẩn hóa (và augmentation nếu có)."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Sequence[int],
        transform: Any = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        import torch

        idx = int(self.indices[position])
        array = self.images[idx]
        if self.transform is not None:
            from PIL import Image as PILImage

            tensor = self.transform(PILImage.fromarray(array))
        else:
            if self.mean is None or self.std is None:
                raise ValueError("Dataset eval cần mean/std để chuẩn hóa")
            tensor = _array_to_tensor(array, self.mean, self.std)
        return tensor, int(self.labels[idx])


def make_loaders(
    cfg: Mapping[str, Any],
    split: str | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Tạo ``train_loader``/``val_loader`` từ cache + split đã lưu."""
    import torch
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    split_mode = split or data_cfg["split"]
    seed = int(cfg["experiment"].get("seed", DEFAULT_SEED) if seed is None else seed)
    batch_size = int(cfg["train"]["batch_size"]) if batch_size is None else int(batch_size)
    workers = int(data_cfg.get("num_workers", 0) if num_workers is None else num_workers)
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    cache = load_cache(int(data_cfg["image_size"]))
    split_payload = load_split(split_mode)
    mean = list(data_cfg["mean"])
    std = list(data_cfg["std"])

    train_ds = GTSRBTensorDataset(
        cache["train"]["x"],
        cache["train"]["y"],
        split_payload["train"],
        transform=build_train_transform(cfg),
    )
    val_ds = GTSRBTensorDataset(
        cache["train"]["x"],
        cache["train"]["y"],
        split_payload["val"],
        transform=None,
        mean=mean,
        std=std,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=generator, **common)
    return {
        "train": train_loader,
        "val": val_loader,
        "meta": {
            "split": split_mode,
            "seed": seed,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "batch_size": batch_size,
            "num_workers": workers,
            "train_transform": repr(train_ds.transform),
            "augmentation_enabled": bool(cfg["augmentation"].get("enabled")),
            "mean": mean,
            "std": std,
        },
    }


def plot_augmentation_grid(cfg: Mapping[str, Any], columns: int = 8, seed: int = DEFAULT_SEED) -> Path:
    """Vẽ lưới 8 ảnh gốc / 8 ảnh sau augmentation để kiểm tra bằng mắt."""
    import torch

    cache = load_cache(int(cfg["data"]["image_size"]))
    labels = cache["train"]["y"]
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for class_id in range(NUM_CLASSES):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates):
            chosen.append(int(rng.choice(candidates)))
        if len(chosen) == columns:
            break
    while len(chosen) < columns:
        chosen.append(int(rng.integers(0, len(labels))))

    transform = build_train_transform(cfg)
    from PIL import Image as PILImage

    mean = np.array(cfg["data"]["mean"])
    std = np.array(cfg["data"]["std"])
    fig, axes = plt.subplots(2, columns, figsize=(columns * 1.7, 4.2))
    for col, idx in enumerate(chosen):
        array = cache["train"]["x"][idx]
        axes[0, col].imshow(array)
        axes[0, col].set_title(f"gốc · lớp {labels[idx]}", fontsize=8)
        axes[0, col].axis("off")
        tensor = transform(PILImage.fromarray(array))
        shown = tensor.permute(1, 2, 0).numpy() * std + mean
        axes[1, col].imshow(np.clip(shown, 0, 1))
        axes[1, col].set_title("sau aug", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle(
        "Kiểm tra augmentation: KHÔNG lật ngang/dọc, KHÔNG xoay 90/180 (giữ nghĩa biển báo)",
        fontsize=10,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_AUGMENTATION, dpi=130)
    plt.close(fig)
    LOG.info("Đã ghi %s", FIG_AUGMENTATION)
    return FIG_AUGMENTATION


def prepare(force: bool = False, size: int = IMG_SIZE) -> dict[str, Any]:
    """Chạy trọn Giai đoạn 2: cache → split → mean/std → hình augmentation."""
    cache_meta = build_cache(size=size, force=force)
    pool = load_train_pool()
    splits = build_splits(
        pool.labels, pool.track_keys, val_ratio=0.2, seed=DEFAULT_SEED, force=force
    )
    cache = load_cache(size)
    stats = compute_stats(cache["train"]["x"], load_split("random")["train"])
    stats.update(
        {
            "split": "random",
            "seed": DEFAULT_SEED,
            "image_size": size,
            "note": (
                "mean/std tính trên train split (không dùng val/test) theo đơn vị [0,1]; "
                "dùng chung cho train, eval và demo qua src.data.preprocess."
            ),
        }
    )
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("mean=%s std=%s → %s", stats["mean"], stats["std"], STATS_PATH)

    cfg = load_config(require_stats=False)
    cfg["data"]["mean"] = stats["mean"]
    cfg["data"]["std"] = stats["std"]
    plot_augmentation_grid(cfg)

    payload = {"cache": cache_meta, "splits": splits, "stats": stats}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase2_prepare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def verify_phase2(size: int = IMG_SIZE) -> list[dict[str, Any]]:
    """Chạy checklist Giai đoạn 2 bằng số liệu thật; ghi ``phase2_check.json``."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        LOG.info("%s %s — %s", "PASS" if passed else "FAIL", name, detail)

    train_pool = load_train_pool()
    random_split = load_split("random")
    group_split = load_split("group")
    cache = load_cache(size)
    x_train, y_train = cache["train"]["x"], cache["train"]["y"]
    x_test = cache["test"]["x"]

    add(
        "split_random_tong",
        len(random_split["train"]) + len(random_split["val"]) == TRAIN_POOL_COUNT,
        f"{len(random_split['train'])} + {len(random_split['val'])} = "
        f"{len(random_split['train']) + len(random_split['val'])} (kỳ vọng {TRAIN_POOL_COUNT})",
    )
    overlap = len(set(random_split["train"]) & set(random_split["val"]))
    add("split_random_giao", overlap == 0, f"giao train/val = {overlap}")

    for name, idx in (("train", random_split["train"]), ("val", random_split["val"])):
        unique = int(len(np.unique(y_train[idx])))
        add(f"du_lop_{name}", unique == NUM_CLASSES, f"{unique}/{NUM_CLASSES} lớp")

    ratios = random_split["summary"]
    add(
        "lech_ty_le_lop",
        ratios["max_class_ratio_gap"] <= 0.01,
        f"lệch tỷ lệ lớn nhất = {ratios['max_class_ratio_gap']:.6f} (ngưỡng ≤ 0,01)",
    )

    track_array = np.asarray(train_pool.track_keys)
    group_overlap = len(
        set(track_array[group_split["train"]]) & set(track_array[group_split["val"]])
    )
    add(
        "split_group_track_roi",
        group_overlap == 0,
        f"giao track = {group_overlap}; track train/val = "
        f"{len(set(track_array[group_split['train']]))}/{len(set(track_array[group_split['val']]))}; "
        f"lớp vắng ở val = {group_split['summary']['classes_missing_val']}",
    )
    add(
        "split_group_kich_thuoc",
        len(group_split["train"]) + len(group_split["val"]) == TRAIN_POOL_COUNT,
        f"{len(group_split['train'])} + {len(group_split['val'])} = {TRAIN_POOL_COUNT}",
    )

    add(
        "cache_shape_dtype",
        x_train.shape == (TRAIN_POOL_COUNT, size, size, 3) and x_train.dtype == np.uint8,
        f"X_train {x_train.shape} {x_train.dtype}; X_test {x_test.shape} {x_test.dtype}",
    )

    sample_path = PROJECT_ROOT / train_pool.paths[0]
    mean_std = load_stats()
    tensor = preprocess(sample_path, size=size)
    # Khoảng giá trị lý thuyết của ảnh đã chuẩn hóa: [(0-mean)/std, (1-mean)/std].
    bound = max(
        max(abs((0.0 - m) / s), abs((1.0 - m) / s))
        for m, s in zip(mean_std["mean"], mean_std["std"])
    )
    add(
        "preprocess_shape_range",
        tuple(tensor.shape) == (3, size, size)
        and bool(np.isfinite(tensor.numpy()).all())
        and float(tensor.min()) >= -bound - 0.05
        and float(tensor.max()) <= bound + 0.05,
        f"shape={tuple(tensor.shape)} min={float(tensor.min()):.3f} max={float(tensor.max()):.3f} "
        f"(biên lý thuyết ±{bound:.2f} ứng với mean/std đã tính; tài liệu ghi 'xấp xỉ [-2,2]')",
    )

    transform_text = repr(build_train_transform(load_config()))
    add(
        "aug_khong_lat",
        all(name not in transform_text for name in ("Flip", "RandomRotation"))
        and FIG_AUGMENTATION.is_file(),
        f"không có Flip/RandomRotation trong transform; hình {FIG_AUGMENTATION.name} "
        f"{'tồn tại' if FIG_AUGMENTATION.is_file() else 'THIẾU'}",
    )

    again = load_split("random")
    add(
        "split_doc_lai",
        again["train"] == random_split["train"] and again["val"] == random_split["val"],
        f"đọc lại {SPLIT_FILES['random'].name} cho cùng index",
    )

    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.data import preprocess;"
            "import torch;"
            "print(tuple(preprocess('" + train_pool.paths[0] + "').shape))",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    add(
        "import_preprocess_tu_ngoai",
        probe.returncode == 0 and "(3, 64, 64)" in probe.stdout,
        f"subprocess: {probe.stdout.strip() or probe.stderr.strip()[:120]}",
    )

    summary = {
        "phase": 2,
        "generated_at": date.today().isoformat(),
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_CHECK_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Kết quả: %d/%d PASS → %s", summary["passed"], summary["total"], PHASE2_CHECK_PATH)
    return checks


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

    write_train_meta(train_infos, excluded=duplicates["excluded_from_train"])
    write_excluded_list(duplicates["cross_split_pairs_detail"])
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
    parser = argparse.ArgumentParser(description="Dữ liệu GTSRB: tải, xác minh, cache, chia tập (GĐ1–2)")
    parser.add_argument("--download", action="store_true", help="tải dữ liệu qua torchvision")
    parser.add_argument("--report", action="store_true", help="xác minh, vẽ EDA, ghi data_report.json")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="dựng cache .npy + split random/group + mean/std + hình augmentation",
    )
    parser.add_argument("--verify", action="store_true", help="chạy checklist Giai đoạn 2")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="chỉ quét N ảnh mỗi tập (dùng cho smoke test; 0 = toàn bộ)",
    )
    parser.add_argument(
        "--image-size", type=int, default=IMG_SIZE, help="cạnh ảnh cache (mặc định 64)"
    )
    parser.add_argument("--force", action="store_true", help="dựng lại cache/split dù đã có")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: ``python -m src.data --download --report | --prepare | --verify``."""
    _configure_logging()
    args = parse_args(argv)
    actions = (args.download, args.report, args.prepare, args.verify)
    if not any(actions):
        LOG.error("Cần ít nhất một trong --download, --report, --prepare, --verify")
        return 2

    download_info = download_gtsrb() if args.download else None
    if args.report:
        run_report(limit=args.limit, download_info=download_info)
    if args.prepare:
        prepare(force=args.force, size=args.image_size)
    if args.verify:
        checks = verify_phase2(size=args.image_size)
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            LOG.error("Checklist Giai đoạn 2 FAIL: %s", failed)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
