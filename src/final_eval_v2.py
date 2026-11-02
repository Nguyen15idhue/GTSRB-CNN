"""Đánh giá cuối v2: lockbox one-shot và official test raw/clean.

Quy tắc:
* Lockbox chỉ mở **một lần** sau freeze. Nếu file kết quả đã tồn tại, từ chối
  trừ khi ``allow_rerun`` + lý do (sinh bản ``_v2``, không ghi đè).
* Mọi lần mở ghi ``outputs/results/v2/lockbox_access.log`` (thời gian, lệnh,
  người chạy, hash checkpoint).
* Official test chỉ là benchmark phụ; báo cả raw và clean (bỏ ảnh gần trùng
  train theo audit MAD < 5 của chính repo này).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import CLASS_NAMES_PATH, NUM_CLASSES, PROJECT_ROOT, RESULTS_DIR, get_device
from .data import load_cache
from .data_v2 import get_indices, get_mean_std, load_images, track_map
from .evaluate import classification_metrics, sha256_file
from .models import build_model

LOG = logging.getLogger(__name__)

V2_RESULTS = RESULTS_DIR / "v2"
AUDIT_DIR = V2_RESULTS / "audit"
LOCKBOX_RESULT = V2_RESULTS / "lockbox_result.json"
LOCKBOX_LOG = V2_RESULTS / "lockbox_access.log"
TEST_RESULT = V2_RESULTS / "official_test_v2.json"
FREEZE_PATH = V2_RESULTS / "FREEZE.json"


def _ece_brier(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict[str, float]:
    """ECE 15 bin + Brier đa lớp."""
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    return {"ece_15bin": float(ece), "brier": brier}


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Đánh giá một checkpoint trên mảng uint8; trả metric đầy đủ."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mean_t = torch.tensor(list(mean)).view(3, 1, 1)
    std_t = torch.tensor(list(std)).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    probs = []
    for start in range(0, len(tensor), batch_size):
        logits = model(tensor[start:start + batch_size].to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(axis=1)
    report = classification_metrics(labels, pred)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    report["top3_accuracy"] = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))
    report.update(_ece_brier(probs, labels))
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
            "config_sha256": checkpoint.get("config_sha256", ""),
            "split_sha256": checkpoint.get("split_sha256", ""),
        }
    )
    return report


def _summarize_seeds(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean ± sd qua seed cho các metric chính."""
    metrics = ("accuracy", "macro_f1_43", "top3_accuracy", "ece_15bin", "brier")
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = np.array([float(r[metric]) for r in reports])
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def _guard_freeze(final_runs: Sequence[Path]) -> dict[str, Any]:
    """Bắt buộc đã freeze trước khi mở lockbox; config của run phải khớp freeze."""
    if not FREEZE_PATH.is_file():
        raise RuntimeError(
            f"Chưa có {FREEZE_PATH}. Phải freeze (bước 3.2) trước khi mở lockbox."
        )
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    expected = freeze.get("final_config_sha256")
    for run in final_runs:
        run = Path(run)
        meta_path = run / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if expected and meta.get("config_sha256") != expected:
            raise RuntimeError(
                f"{run.name}: config_sha256 {str(meta.get('config_sha256'))[:12]} khác freeze {expected[:12]}"
            )
        if not (run / "best.pt").is_file():
            raise FileNotFoundError(run / "best.pt")
    return freeze


def run_lockbox(
    final_runs: Sequence[Path],
    allow_rerun: bool = False,
    reason: str = "",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Mở lockbox đúng một lần cho các run final; ghi kết quả + log."""
    if LOCKBOX_RESULT.is_file() and not allow_rerun:
        raise RuntimeError(
            f"{LOCKBOX_RESULT} đã tồn tại — lockbox chỉ mở một lần. "
            "Nếu có bug: --allow-rerun --reason '...' (sinh bản _v2)."
        )
    if allow_rerun and not reason.strip():
        raise RuntimeError("--allow-rerun bắt buộc kèm --reason")
    device = device or get_device()
    freeze = _guard_freeze([Path(r) for r in final_runs])

    indices = get_indices("lockbox")["val"]
    payload = load_images(64)
    images, labels = payload["x"][indices], payload["y"][indices]
    mean, std = get_mean_std("dev_all")
    reports = []
    for run in final_runs:
        report = evaluate_checkpoint(Path(run) / "best.pt", images, labels, mean, std, device)
        report["run_id"] = Path(run).name
        report["checkpoint_sha256"] = sha256_file(Path(run) / "best.pt")
        reports.append(report)
        LOG.info("Lockbox %s: acc %.6f | macro-F1 %.6f", Path(run).name, report["accuracy"], report["macro_f1_43"])

    result = {
        "protocol": "v2_clean",
        "evaluated_on": "lockbox",
        "n_images": int(len(labels)),
        "n_tracks": len({track_map()[int(i)]["track_key"] for i in indices}),
        "split_sha256": freeze["split_sha256"],
        "freeze": freeze,
        "reports": reports,
        "summary": _summarize_seeds(reports),
        "confusion_matrix": reports[0]["confusion_matrix"] if len(reports) == 1 else None,
        "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "user": os.environ.get("USERNAME", "unknown"),
        "rerun": bool(allow_rerun),
        "rerun_reason": reason,
    }
    path = LOCKBOX_RESULT if not allow_rerun else V2_RESULTS / "lockbox_result_v2.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with LOCKBOX_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{result['opened_at']} | user={result['user']} | host={result['host']} | "
            f"runs={[Path(r).name for r in final_runs]} | n={len(labels)} | "
            f"acc={result['summary']['accuracy']['mean']:.6f} | "
            f"f1={result['summary']['macro_f1_43']['mean']:.6f} | rerun={allow_rerun}\n"
        )
    LOG.info("Đã ghi %s", path)
    return result


def compute_test_near_duplicates(force: bool = False) -> dict[str, Any]:
    """Audit test vs train (MAD < 5) bằng thumbnail + kiểm chứng full 64×64."""
    out_pairs = AUDIT_DIR / "near_duplicate_test.csv"
    out_clean = AUDIT_DIR / "clean_test.json"
    if out_clean.is_file() and not force:
        return json.loads(out_clean.read_text(encoding="utf-8"))

    train = load_images(64)
    test = load_cache(64)["test"]
    x_train, y_train = train["x"], train["y"]
    x_test, y_test = test["x"], test["y"]

    def thumb(x: np.ndarray) -> np.ndarray:
        n = len(x)
        gray = x.astype(np.float32).mean(axis=3)
        return gray.reshape(n, 16, 4, 16, 4).mean(axis=(2, 4)).reshape(n, 256)

    t_train, t_test = thumb(x_train), thumb(x_test)
    flat_train = x_train.reshape(len(x_train), -1).astype(np.int16)
    flat_test = x_test.reshape(len(x_test), -1).astype(np.int16)
    pairs: list[dict[str, Any]] = []
    for class_id in range(NUM_CLASSES):
        train_idx = np.flatnonzero(y_train == class_id)
        test_idx = np.flatnonzero(y_test == class_id)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        block = t_train[train_idx]
        for start in range(0, len(test_idx), 64):
            stop = min(start + 64, len(test_idx))
            distance = np.abs(t_test[test_idx[start:stop], None, :] - block[None, :, :]).mean(axis=2)
            rows, cols = np.nonzero(distance < 8.0)
            for r, c in zip(rows, cols):
                ti, ri = int(test_idx[start + r]), int(train_idx[c])
                mad = float(np.abs(flat_test[ti] - flat_train[ri]).mean())
                if mad < 5.0:
                    pairs.append({"test_index": ti, "train_index": ri, "label": class_id, "mad": round(mad, 4)})
    unique_test = sorted({p["test_index"] for p in pairs})
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test_index", "train_index", "label", "mad"])
        writer.writeheader(); writer.writerows(pairs)
    payload = {
        "criterion": "MAD < 5 trên ảnh 64x64 uint8, cùng lớp",
        "n_pairs": len(pairs),
        "n_test_images": len(unique_test),
        "n_test_total": int(len(y_test)),
        "n_clean": int(len(y_test) - len(unique_test)),
        "clean_indices": [i for i in range(len(y_test)) if i not in set(unique_test)],
        "excluded_indices": unique_test,
        "note": "Official test không track-disjoint; clean chỉ là kiểm tra độ nhạy.",
    }
    out_clean.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Test gần trùng: %d cặp / %d ảnh test; clean %d", len(pairs), len(unique_test), payload["n_clean"])
    return payload


def run_official_test(
    final_runs: Sequence[Path],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Official test raw + clean cho các run final (benchmark phụ, không tune)."""
    device = device or get_device()
    test = load_cache(64)["test"]
    clean = compute_test_near_duplicates()
    mean, std = get_mean_std("dev_all")
    reports_raw, reports_clean = [], []
    for run in final_runs:
        report = evaluate_checkpoint(Path(run) / "best.pt", test["x"], test["y"], mean, std, device)
        report["run_id"] = Path(run).name
        reports_raw.append(report)
        idx = np.asarray(clean["clean_indices"], dtype=np.int64)
        clean_report = evaluate_checkpoint(
            Path(run) / "best.pt", test["x"][idx], test["y"][idx], mean, std, device
        )
        clean_report["run_id"] = Path(run).name
        reports_clean.append(clean_report)
        LOG.info(
            "Official %s: raw acc %.6f | clean acc %.6f",
            Path(run).name, report["accuracy"], clean_report["accuracy"],
        )
    result = {
        "protocol": "v2_clean",
        "n_raw": int(len(test["y"])),
        "n_clean": int(len(clean["clean_indices"])),
        "n_near_duplicate_test_images": int(clean["n_test_images"]),
        "note": "Official test không track-disjoint; chỉ báo cáo như benchmark phụ.",
        "raw": {"reports": reports_raw, "summary": _summarize_seeds(reports_raw)},
        "clean": {"reports": reports_clean, "summary": _summarize_seeds(reports_clean)},
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    TEST_RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Đã ghi %s", TEST_RESULT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    parser = argparse.ArgumentParser(description="Đánh giá lockbox / official test v2")
    parser.add_argument("--mode", choices=("lockbox", "test"), required=True)
    parser.add_argument("--runs", required=True, help="danh sách thư mục run, phân tách bằng dấu phẩy")
    parser.add_argument("--allow-rerun", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    runs = [Path(r) for r in args.runs.split(",") if r.strip()]
    if args.mode == "lockbox":
        run_lockbox(runs, allow_rerun=args.allow_rerun, reason=args.reason)
    else:
        run_official_test(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
