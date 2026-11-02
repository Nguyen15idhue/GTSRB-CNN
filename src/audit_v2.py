"""Audit độc lập split v2 (C1–C12) — Giai đoạn 0.3.

Không tin ``manifest.checks``: đọc lại file split, tính lại overlap/đếm lớp/hash,
so với manifest, rồi ghi ``outputs/results/v2/audit/audit_report.json`` và
``checks.csv``. Nếu bất kỳ mục nào fail → exit 1 (dừng khẩn cấp, không sang Phase 1).

Cách dùng::

    python -m src.audit_v2
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import NUM_CLASSES, PROJECT_ROOT, RESULTS_DIR
from .data import _read_train_meta, _track_key, compute_stats, load_cache

LOG = logging.getLogger(__name__)

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits" / "v2"
AUDIT_DIR = RESULTS_DIR / "v2" / "audit"
MANIFEST = SPLIT_DIR / "manifest.json"
AUDIT_REPORT = AUDIT_DIR / "audit_report.json"
CHECKS_CSV = AUDIT_DIR / "checks.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, Any]:
    return json.loads((SPLIT_DIR / name).read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    """Tính lại C1–C12 từ file split và ghi báo cáo."""
    manifest = _load("manifest.json")
    rows = [r for r in _read_train_meta() if int(r["usable_for_train"]) == 1]
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    tracks = np.array([_track_key(r["path"], int(r["label"])) for r in rows], dtype=object)

    lock = _load("lockbox.json")["indices"]
    dev = _load("dev.json")["indices"]
    screening = _load("screening.json")
    folds = [_load(f"fold_{i}.json") for i in range(3)]

    lock_t, dev_t = set(tracks[lock].tolist()), set(tracks[dev].tolist())
    # sign-group lấy từ audit CSV (đã sinh ở bước 0.2).
    group_of: dict[str, str] = {}
    with (AUDIT_DIR / "sign_groups.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for track in row["tracks"].split("|"):
                group_of[track] = row["sign_group"]
    groups = np.array([group_of.get(t, "") for t in tracks], dtype=object)

    def support(indices: Sequence[int]) -> list[int]:
        return np.bincount(labels[list(indices)], minlength=NUM_CLASSES).astype(int).tolist()

    checks: list[dict[str, Any]] = []

    def add(code: str, name: str, value: Any, passed: bool, criterion: str) -> None:
        checks.append({"check": code, "name": name, "value": value, "criterion": criterion,
                       "passed": bool(passed)})
        LOG.info("%s %s — %s (%s)", "PASS" if passed else "FAIL", code, name, value)

    add("C1", "Track overlap lockbox/dev", len(lock_t & dev_t), len(lock_t & dev_t) == 0, "= 0")
    add("C2", "Sign-group overlap lockbox/dev",
        len(set(groups[lock].tolist()) & set(groups[dev].tolist())),
        len(set(groups[lock].tolist()) & set(groups[dev].tolist())) == 0, "= 0")
    add("C3", "Lockbox đủ lớp", sum(1 for v in support(lock) if v > 0),
        sum(1 for v in support(lock) if v > 0) == NUM_CLASSES, "= 43")
    add("C4", "Dev đủ lớp", sum(1 for v in support(dev) if v > 0),
        sum(1 for v in support(dev) if v > 0) == NUM_CLASSES, "= 43")
    add("C5", "Screening val đủ lớp", sum(1 for v in support(screening["val"]) if v > 0),
        sum(1 for v in support(screening["val"]) if v > 0) == NUM_CLASSES, "= 43")
    fold_classes = [sum(1 for v in support(f["val"]) if v > 0) for f in folds]
    add("C6", "Mỗi fold đủ lớp", fold_classes, all(v == NUM_CLASSES for v in fold_classes), "= 43")
    add("C7", "Overlap screening train/val",
        len(set(tracks[screening["train"]].tolist()) & set(tracks[screening["val"]].tolist())),
        len(set(tracks[screening["train"]].tolist()) & set(tracks[screening["val"]].tolist())) == 0, "= 0")
    fold_overlap = [len(set(tracks[f["train"]].tolist()) & set(tracks[f["val"]].tolist())) for f in folds]
    add("C8", "Overlap track từng fold", fold_overlap, all(v == 0 for v in fold_overlap), "= 0")

    summary = json.loads((AUDIT_DIR / "near_duplicate_summary.json").read_text(encoding="utf-8"))
    add("C9", "Test nằm trong train/lockbox", summary["test_images_matching_train"],
        summary["test_images_matching_train"] == 0, "= 0")
    add("C10", "Lockbox gần trùng train (MAD < 5)", summary["n_cross_lockbox_dev"],
        summary["n_cross_lockbox_dev"] == 0, "= 0")
    md5_rows = sum(1 for _ in csv.DictReader((AUDIT_DIR / "md5_duplicates_train.csv").open(encoding="utf-8")))
    add("C11", "Hash trùng trong pool", md5_rows, md5_rows == 0, "= 0")
    dev_track_per_class = [
        len({tracks[i] for i in np.flatnonzero(labels == c) if i in set(dev)})
        for c in range(NUM_CLASSES)
    ]
    add("C12", "Track/lớp nhỏ nhất trong dev", int(min(dev_track_per_class)),
        min(dev_track_per_class) >= 2, ">= 2")

    # Hash từng file khớp manifest.
    hash_checks = {
        "lockbox.json": manifest["lockbox"]["sha256"],
        "dev.json": manifest["dev"]["sha256"],
        "screening.json": manifest["screening"]["sha256"],
    }
    for i in range(3):
        hash_checks[f"fold_{i}.json"] = manifest["folds"][i]["sha256"]
    hash_mismatch = [name for name, expected in hash_checks.items()
                     if _sha256(SPLIT_DIR / name) != expected]
    add("HASH", "Hash file split khớp manifest", hash_mismatch or "khớp", not hash_mismatch, "khớp toàn bộ")

    # mean/std khớp manifest (tính lại screening train).
    cache = load_cache(64)
    recomputed = compute_stats(cache["train"]["x"], screening["train"])
    stored = manifest["mean_std"]["screening_train"]
    mean_diff = float(np.max(np.abs(np.array(recomputed["mean"]) - np.array(stored["mean"]))))
    std_diff = float(np.max(np.abs(np.array(recomputed["std"]) - np.array(stored["std"]))))
    add("MEANSTD", "mean/std khớp manifest", {"mean_abs_diff": mean_diff, "std_abs_diff": std_diff},
        mean_diff < 1e-5 and std_diff < 1e-5, "< 1e-5")

    # Số ảnh khớp giữa manifest và file.
    counts_ok = (
        manifest["lockbox"]["n"] == len(lock)
        and manifest["dev"]["n"] == len(dev)
        and manifest["screening"]["train"] == len(screening["train"])
        and manifest["screening"]["val"] == len(screening["val"])
        and all(manifest["folds"][i]["train"] == len(folds[i]["train"])
                and manifest["folds"][i]["val"] == len(folds[i]["val"]) for i in range(3))
    )
    add("COUNT", "Số ảnh manifest khớp file split", counts_ok, counts_ok, "khớp")

    passed = all(c["passed"] for c in checks)
    report = {
        "protocol": "v2_clean",
        "manifest_sha256": _sha256(MANIFEST),
        "passed": passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "split_sizes": {
            "lockbox": len(lock), "dev": len(dev),
            "screening_train": len(screening["train"]), "screening_val": len(screening["val"]),
            "folds": [{"train": len(f["train"]), "val": len(f["val"])} for f in folds],
        },
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with CHECKS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["check", "name", "value", "criterion", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    LOG.info("Audit: %d/%d PASS → %s", report["n_passed"], report["n_checks"], AUDIT_REPORT)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    report = audit()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
