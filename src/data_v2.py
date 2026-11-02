"""Dữ liệu cho split v2: đọc manifest, cache theo kích thước, dataloader, augmentation.

Không sửa hành vi cũ trong ``src/data.py``; module này chỉ thêm chế độ v2.
Tiền xử lý dùng lại đúng hàm ``_array_to_tensor`` của ``src.data.preprocess``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import CACHE_DIR, PROJECT_ROOT, RESULTS_DIR
from .data import (
    TRAIN_META_PATH,
    _array_to_tensor,
    _load_resized,
    _track_key,
    load_cache,
    load_train_pool,
    seed_worker,
)

LOG = logging.getLogger(__name__)

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits" / "v2"
MANIFEST_PATH = SPLIT_DIR / "manifest.json"
V2_RESULTS = RESULTS_DIR / "v2"
_CACHE: dict[int, dict[str, np.ndarray]] = {}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Đọc ``manifest.json`` của split v2; raise nếu chưa sinh."""
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python -m src.split_v2 --build")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_sha256(path: Path = MANIFEST_PATH) -> str:
    """SHA-256 của file manifest (ghi vào checkpoint/meta của mỗi run)."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sha256(name: str, manifest: Mapping[str, Any] | None = None) -> str:
    """SHA-256 của file split tương ứng (screening/fold_k/dev/lockbox)."""
    manifest = manifest or load_manifest()
    if name.startswith("fold_"):
        fold = int(name.split("_")[1])
        return str(manifest["folds"][fold]["sha256"])
    key = {"screening": "screening", "dev_all": "dev", "lockbox": "lockbox"}[name]
    return str(manifest[key]["sha256"])


def get_indices(name: str, fold: int | None = None) -> dict[str, list[int]]:
    """Trả ``{"train": [...], "val": [...]}`` (chỉ số trong pool 39.201).

    ``name``: ``screening`` | ``dev_all`` | ``lockbox`` | ``fold_0..2``.
    """
    if name == "screening":
        payload = json.loads((SPLIT_DIR / "screening.json").read_text(encoding="utf-8"))
        return {"train": payload["train"], "val": payload["val"]}
    if name == "dev_all":
        payload = json.loads((SPLIT_DIR / "dev.json").read_text(encoding="utf-8"))
        return {"train": payload["indices"], "val": []}
    if name == "lockbox":
        payload = json.loads((SPLIT_DIR / "lockbox.json").read_text(encoding="utf-8"))
        return {"train": [], "val": payload["indices"]}
    if name.startswith("fold_"):
        index = int(name.split("_")[1]) if fold is None else int(fold)
        payload = json.loads((SPLIT_DIR / f"fold_{index}.json").read_text(encoding="utf-8"))
        return {"train": payload["train"], "val": payload["val"]}
    raise ValueError(f"split v2 không hợp lệ: {name!r}")


def stats_key(name: str) -> str:
    """Khóa mean/std trong manifest cho train của split."""
    return "dev_all" if name == "dev_all" else (f"{name}_train" if name.startswith("fold_") else "screening_train")


def get_mean_std(name: str, manifest: Mapping[str, Any] | None = None) -> tuple[list[float], list[float]]:
    """mean/std tính trên train của đúng split (không dùng toàn cục)."""
    manifest = manifest or load_manifest()
    stats = manifest["mean_std"][stats_key(name)]
    return list(stats["mean"]), list(stats["std"])


# ---------------------------------------------------------------------------
# Cache theo kích thước (64 dùng cache cũ; 96/48 dựng riêng, không ghi đè)
# ---------------------------------------------------------------------------


def cache_dir(size: int) -> Path:
    """Thư mục cache cho kích thước ảnh; 64 dùng ``data/cache`` hiện có."""
    return CACHE_DIR if int(size) == 64 else CACHE_DIR.parent / f"cache_{int(size)}"


def load_images(size: int = 64) -> dict[str, np.ndarray]:
    """Nạp ``X/y`` của pool train ở kích thước mong muốn (dựng cache nếu thiếu)."""
    size = int(size)
    if size in _CACHE:
        return _CACHE[size]
    if size == 64:
        cache = load_cache(64)
        payload = {"x": cache["train"]["x"], "y": cache["train"]["y"]}
    else:
        folder = cache_dir(size)
        x_path, y_path = folder / "X_train.npy", folder / "y_train.npy"
        if not (x_path.is_file() and y_path.is_file()):
            build_cache_size(size)
        payload = {"x": np.load(x_path), "y": np.load(y_path)}
    if len(payload["x"]) != 39_201:
        raise RuntimeError(f"Cache {size} có {len(payload['x'])} ảnh, kỳ vọng 39.201")
    _CACHE[size] = payload
    return payload


def build_cache_size(size: int) -> Path:
    """Dựng cache uint8 cho kích thước 48/96 từ ảnh gốc (không đụng cache 64)."""
    if int(size) == 64:
        raise ValueError("Cache 64 đã có sẵn; không dựng lại")
    pool = load_train_pool()
    folder = cache_dir(size)
    folder.mkdir(parents=True, exist_ok=True)
    x = np.empty((len(pool.paths), int(size), int(size), 3), dtype=np.uint8)
    LOG.info("Dựng cache %d: %d ảnh...", size, len(pool.paths))
    for i, rel in enumerate(pool.paths):
        x[i] = _load_resized(PROJECT_ROOT / rel, int(size))
        if (i + 1) % 5000 == 0:
            LOG.info("  %d/%d", i + 1, len(pool.paths))
    np.save(folder / "X_train.npy", x)
    np.save(folder / "y_train.npy", pool.labels.astype(np.int16))
    LOG.info("Đã ghi cache %d → %s", size, folder)
    return folder


def track_map() -> list[dict[str, Any]]:
    """Danh sách ``{index, path, label, track_key}`` theo đúng thứ tự pool/cache."""
    rows = [r for r in _read_meta() if int(r["usable_for_train"]) == 1]
    return [
        {
            "index": i,
            "path": r["path"],
            "label": int(r["label"]),
            "track_key": _track_key(r["path"], int(r["label"])),
        }
        for i, r in enumerate(rows)
    ]


def _read_meta() -> list[dict[str, str]]:
    import csv

    with TRAIN_META_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Augmentation theo profile
# ---------------------------------------------------------------------------


def build_augmentation(
    profile: str,
    size: int = 64,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    """Transform train theo profile: none/affine/color/perspective/all (+ biến thể exploratory).

    Luôn kết thúc bằng ``ToTensor`` + ``Normalize`` để trả tensor, không trả PIL.
    Cấm tuyệt đối flip và xoay 90/180 (đổi nghĩa biển báo).
    """
    from torchvision import transforms

    profile = (profile or "none").lower()
    ops: list[Any] = []
    affine = transforms.RandomAffine(
        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    color_strong = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03)
    perspective = transforms.RandomPerspective(
        distortion_scale=0.1, p=0.3, interpolation=transforms.InterpolationMode.BILINEAR
    )
    mapping = {
        "affine": [affine],
        "color": [color],
        "color_strong": [color_strong],
        "perspective": [perspective],
        "all": [affine, color, perspective],
        "affine_color": [affine, color],
        "affine_perspective": [affine, perspective],
        "color_perspective": [color, perspective],
    }
    if profile not in ("none", "") and profile not in mapping:
        raise ValueError(f"augmentation profile không hợp lệ: {profile!r}")
    if profile in mapping:
        ops = mapping[profile]
    _assert_no_flip(ops)
    normalize_mean = list(mean) if mean is not None else [0.0, 0.0, 0.0]
    normalize_std = list(std) if std is not None else [1.0, 1.0, 1.0]
    return transforms.Compose(
        [*ops, transforms.ToTensor(), transforms.Normalize(mean=normalize_mean, std=normalize_std)]
    )


def _assert_no_flip(ops: Sequence[Any]) -> None:
    text = repr(list(ops))
    if "Flip" in text or "RandomRotation" in text:
        raise ValueError(f"Augmentation chứa phép biến đổi bị cấm: {text}")


class V2Dataset:
    """Dataset uint8 → tensor chuẩn hóa; augmentation chỉ ở train."""

    def __init__(self, images, labels, indices, transform=None, mean=None, std=None):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        idx = int(self.indices[position])
        array = self.images[idx]
        if self.transform is not None:
            from PIL import Image as PILImage

            tensor = self.transform(PILImage.fromarray(array))
        else:
            tensor = _array_to_tensor(array, self.mean, self.std)
        return tensor, int(self.labels[idx])


def make_loaders_v2(
    cfg: Mapping[str, Any],
    split_name: str,
    seed: int,
    batch_size: int | None = None,
    num_workers: int | None = None,
    augmentation: bool | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Tạo train/val loader cho split v2 (mean/std lấy từ manifest của split)."""
    import torch
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    size = int(data_cfg.get("image_size", 64))
    indices = get_indices(split_name, data_cfg.get("fold"))
    mean, std = get_mean_std(split_name)
    payload = load_images(size)
    aug_cfg = cfg.get("augmentation", {})
    use_aug = bool(aug_cfg.get("enabled", False)) if augmentation is None else augmentation
    used_profile = str(aug_cfg.get("profile", "none")) if profile is None else profile
    transform = build_augmentation(used_profile, size, mean, std) if use_aug else None
    workers = int(data_cfg.get("num_workers", 4) if num_workers is None else num_workers)
    # Số worker là hạ tầng, không phải yếu tố mô hình; cho phép override bằng biến
    # môi trường để tăng tốc mà KHÔNG đổi config_sha256.
    import os

    workers = int(os.environ.get("V2_NUM_WORKERS", workers))
    batch = int(cfg["train"].get("batch_size", 128)) if batch_size is None else int(batch_size)

    train_ds = V2Dataset(payload["x"], payload["y"], indices["train"], transform=transform, mean=mean, std=std)
    val_ds = V2Dataset(payload["x"], payload["y"], indices["val"], mean=mean, std=std)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(val_ds, batch_size=max(batch, 256), shuffle=False, **common)
    return {
        "train": train_loader,
        "val": val_loader,
        "meta": {
            "split": split_name,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "batch_size": batch,
            "num_workers": workers,
            "mean": mean,
            "std": std,
            "augmentation_enabled": bool(use_aug),
            "augmentation_profile": used_profile,
            "transform": repr(transform),
        },
        "indices": indices,
    }
