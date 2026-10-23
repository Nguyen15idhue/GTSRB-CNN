"""Bộ test tự động tối thiểu cho pipeline GTSRB (không cần GPU, không train).

Chạy::

    python -m pytest tests -q
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from src.config import CLASS_NAMES_PATH, NUM_CLASSES, PROJECT_ROOT, RESULTS_DIR, RUNS_DIR
from src.data import load_split, load_train_pool, preprocess
from src.predict import validate_image_bytes

DATA_READY = (PROJECT_ROOT / "data" / "cache" / "X_test.npy").is_file()
needs_data = pytest.mark.skipif(not DATA_READY, reason="Chưa có cache dữ liệu (data/cache)")


@needs_data
def test_preprocess_shape_range() -> None:
    """preprocess trả (3,64,64) float, giá trị hữu hạn, đúng biên lý thuyết."""
    stats = json.loads((RESULTS_DIR / "data_stats.json").read_text(encoding="utf-8"))
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    tensor = preprocess(image, size=64, mean=stats["mean"], std=stats["std"])
    assert tuple(tensor.shape) == (3, 64, 64)
    assert torch.isfinite(tensor).all()
    bound = max(max(abs((0 - m) / s), abs((1 - m) / s)) for m, s in zip(stats["mean"], stats["std"]))
    assert float(tensor.min()) >= -bound - 0.05
    assert float(tensor.max()) <= bound + 0.05


def test_split_random_disjoint() -> None:
    split = load_split("random")
    assert len(set(split["train"]) & set(split["val"])) == 0
    assert len(split["train"]) + len(split["val"]) == 39201


def test_split_stratified_group_tracks_disjoint_and_all_classes() -> None:
    pool = load_train_pool()
    split = load_split("stratified_group")
    tracks = np.asarray(pool.track_keys)
    train_idx, val_idx = np.asarray(split["train"]), np.asarray(split["val"])
    assert len(set(tracks[train_idx]) & set(tracks[val_idx])) == 0
    assert len(np.unique(pool.labels[val_idx])) == NUM_CLASSES


def test_macro_f1_counts_all_43_classes() -> None:
    """Thiếu lớp trong y_true thì macro-F1 phải tính cả 43 lớp, không bỏ qua."""
    y_true = np.array([0] * 5 + [1] * 5)
    y_pred = np.array([0] * 4 + [1] + [1] * 4 + [2])
    f1_default = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_43 = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    assert f1_43 < f1_default


def test_checkpoint_loads_and_forward_43() -> None:
    from src.models import build_model

    path = RUNS_DIR / "final_s42" / "best.pt"
    if not path.is_file():
        pytest.skip("Chưa có checkpoint final_s42")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "config", "class_names", "epoch", "val_macro_f1", "git_commit", "torch_version"):
        assert key in checkpoint
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, NUM_CLASSES)
    assert len(checkpoint["class_names"]) == NUM_CLASSES


def test_validate_image_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="hỏng|hợp lệ"):
        validate_image_bytes("fake.jpg", b"not an image")
    with pytest.raises(ValueError, match="JPG"):
        validate_image_bytes("note.txt", b"abc")
    with pytest.raises(ValueError, match="quá lớn"):
        validate_image_bytes("big.jpg", b"\xff" * (21 * 1024 * 1024))


def test_predictions_and_report_consistent() -> None:
    report_path = RESULTS_DIR / "test_report.json"
    predictions_path = RUNS_DIR / "final_s42" / "eval_test" / "predictions.csv"
    if not report_path.is_file() or not predictions_path.is_file():
        pytest.skip("Chưa có test report/predictions")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(predictions_path.open(encoding="utf-8-sig")))
    assert len(rows) == 12630
    assert len({r["path"] for r in rows}) == 12630
    y_true = np.array([int(r["true_id"]) for r in rows])
    y_pred = np.array([int(r["pred_id"]) for r in rows])
    assert abs(float((y_true == y_pred).mean()) - report["accuracy"]) < 1e-12
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    assert abs(macro - report["macro_f1_43"]) < 1e-12


def test_test_evaluated_once() -> None:
    log = RESULTS_DIR / "test_runs.log"
    if not log.is_file():
        pytest.skip("Chưa có log test")
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_threshold_chosen_on_val() -> None:
    path = RESULTS_DIR / "threshold.json"
    if not path.is_file():
        pytest.skip("Chưa có threshold.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_val"] == 7830
    assert "val" in payload["chosen_on"]
    assert 0.0 < payload["tau"] < 1.0


def test_manifest_matches_images() -> None:
    folder = PROJECT_ROOT / "external_test"
    manifest = folder / "manifest.csv"
    if not manifest.is_file():
        pytest.skip("Chưa có external_test/manifest.csv")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    images = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    assert len(rows) == len(images) == 11
    for row in rows:
        assert (folder / row["filename"]).is_file()
        assert row["license"].strip()
        assert row["anonymized"].strip()
    reference = [r for r in rows if r["group"] == "reference_gtsrb_test"]
    assert len(reference) == 1
    assert reference[0]["license"].startswith("GTSRB") and "KHÔNG phải CC0" in reference[0]["license"]


def test_class_names_cover_43() -> None:
    names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    assert len(names) == NUM_CLASSES
    assert all("vi" in value and "en" in value for value in names.values())
