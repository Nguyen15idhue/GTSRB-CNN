"""Test split v2, dataloader và các bất biến metric (Bước 0.4).

Chạy::

    pytest -q tests/test_split_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import NUM_CLASSES, PROJECT_ROOT
from src.data import _read_train_meta, _track_key, compute_stats, load_cache
from src.data_v2 import (
    build_augmentation,
    get_indices,
    get_mean_std,
    load_manifest,
    make_loaders_v2,
    split_sha256,
)
from src.models import build_model
from src.train_v2 import evaluate_epoch

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits" / "v2"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest()


@pytest.fixture(scope="module")
def meta() -> dict:
    rows = [r for r in _read_train_meta() if int(r["usable_for_train"]) == 1]
    labels = np.array([int(r["label"]) for r in rows])
    tracks = np.array([_track_key(r["path"], int(r["label"])) for r in rows], dtype=object)
    return {"rows": rows, "labels": labels, "tracks": tracks}


def test_manifest_checks_all_pass(manifest):
    checks = manifest["checks"]
    assert checks["C1_overlap_track_lockbox_dev"] == 0
    assert checks["C2_overlap_sign_group_lockbox_dev"] == 0
    assert checks["C3_classes_lockbox"] == NUM_CLASSES
    assert checks["C4_classes_dev"] == NUM_CLASSES
    assert checks["C5_classes_screening_val"] == NUM_CLASSES
    assert all(v == NUM_CLASSES for v in checks["C6_folds_classes_val"])
    assert checks["C7_overlap_screening"] == 0
    assert all(v == 0 for v in checks["C8_fold_overlap_track"])
    assert checks["C9_test_in_train"] == 0
    assert checks["C10_lockbox_near_duplicate_mad_lt_5"] == 0
    assert checks["C11_md5_duplicates_in_pool"] == 0
    assert checks["C12_min_tracks_per_class_dev"] >= 2


def test_split_files_match_manifest_hash(manifest):
    import hashlib

    names = {
        "screening": manifest["screening"]["sha256"],
        "dev": manifest["dev"]["sha256"],
        "lockbox": manifest["lockbox"]["sha256"],
    }
    for i in range(3):
        names[f"fold_{i}"] = manifest["folds"][i]["sha256"]
    for name, expected in names.items():
        digest = hashlib.sha256((SPLIT_DIR / f"{name}.json").read_bytes()).hexdigest()
        assert digest == expected, name
        assert split_sha256("dev_all" if name == "dev" else name) == expected


def test_splits_disjoint_by_track(manifest, meta):
    tracks = meta["tracks"]
    lock = set(tracks[get_indices("lockbox")["val"]].tolist())
    dev = set(tracks[get_indices("dev_all")["train"]].tolist())
    assert not (lock & dev)
    screening = get_indices("screening")
    assert not (set(tracks[screening["train"]].tolist()) & set(tracks[screening["val"]].tolist()))
    for i in range(3):
        fold = get_indices(f"fold_{i}")
        assert not (set(tracks[fold["train"]].tolist()) & set(tracks[fold["val"]].tolist()))


def test_every_eval_set_has_43_classes(meta):
    labels = meta["labels"]
    for name in ("screening", "fold_0", "fold_1", "fold_2", "lockbox"):
        idx = get_indices(name)["val"] if name != "lockbox" else get_indices("lockbox")["val"]
        support = np.bincount(labels[idx], minlength=NUM_CLASSES)
        assert (support > 0).sum() == NUM_CLASSES, name


def test_mean_std_in_manifest_matches_recompute(manifest):
    mean, std = get_mean_std("screening")
    stored = manifest["mean_std"]["screening_train"]
    assert np.allclose(mean, stored["mean"], atol=1e-6)
    assert np.allclose(std, stored["std"], atol=1e-6)
    cache = load_cache(64)
    recomputed = compute_stats(cache["train"]["x"], get_indices("screening")["train"])
    assert np.allclose(recomputed["mean"], mean, atol=1e-5)
    assert np.allclose(recomputed["std"], std, atol=1e-5)


def test_forward_shapes_for_a_b_c():
    import yaml

    for config_id, expected_blocks in (("f00", 3), ("f07", 3), ("d3", 4)):
        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "v2" / f"{config_id}.yaml").read_text(encoding="utf-8"))
        model, params = build_model(cfg)
        assert params > 0
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 3, cfg["data"]["image_size"], cfg["data"]["image_size"]))
        assert tuple(logits.shape) == (2, NUM_CLASSES)
        assert len(cfg["model"]["widths"]) == expected_blocks


def test_macro_f1_matches_sklearn():
    import torch
    from sklearn.metrics import f1_score
    from src.data_v2 import V2Dataset

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(64, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, size=64).astype(np.int64)
    dataset = V2Dataset(images, labels, np.arange(64), mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    cfg = {"model": {"name": "smallcnn_3blk", "widths": [8, 16], "use_bn": True, "dropout": 0.0,
                     "gap": True, "fc_hidden": 16},
           "data": {"num_classes": NUM_CLASSES, "image_size": 32}}
    model, _ = build_model(cfg)
    _, _, macro = evaluate_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    with torch.no_grad():
        pred = torch.cat([model(x).argmax(1) for x, _ in loader]).numpy()
    expected = f1_score(labels, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    assert abs(macro - float(expected)) < 1e-9


def test_no_flip_in_any_profile():
    for profile in ("affine", "color", "perspective", "all", "color_strong"):
        transform = build_augmentation(profile)
        assert "Flip" not in repr(transform)
        assert "RandomRotation" not in repr(transform)


def test_num_workers_2_does_not_hang():
    """Windows: DataLoader 2 worker phải chạy được (dùng dataset nhỏ để spawn nhanh)."""
    from torch.utils.data import DataLoader
    from src.data_v2 import V2Dataset

    rng = np.random.default_rng(1)
    images = rng.integers(0, 255, size=(128, 64, 64, 3), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, size=128).astype(np.int64)
    dataset = V2Dataset(images, labels, np.arange(128), mean=[0.5] * 3, std=[0.25] * 3)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    assert tuple(batch[0].shape) == (16, 3, 64, 64)


def test_predictions_cover_dev_exactly_once(meta):
    """3 fold ghép lại phải phủ đúng toàn bộ dev, không trùng."""
    dev = set(get_indices("dev_all")["train"])
    covered: list[int] = []
    for i in range(3):
        covered.extend(get_indices(f"fold_{i}")["val"])
    assert len(covered) == len(set(covered))
    assert set(covered) == dev
