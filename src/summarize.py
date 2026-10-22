"""Tổng hợp kết quả các run thành ``outputs/results/summary.csv`` (nguồn số liệu duy nhất).

Quét ``outputs/runs/*/`` theo hai loại:

* run CNN: có ``meta.json`` + ``metrics.csv``;
* baseline (E0a/E0b): có ``metrics.json`` do ``src.baseline`` sinh ra.

Cách dùng::

    python -m src.summarize
    python -m src.summarize --figure
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, RESULTS_DIR, RUNS_DIR

LOG = logging.getLogger(__name__)

SUMMARY_PATH = RESULTS_DIR / "summary.csv"
COMPARISON_FIG = FIGURES_DIR / "05_experiment_comparison.png"

COLUMNS = (
    "run_id",
    "model",
    "params",
    "aug",
    "epochs_run",
    "best_epoch",
    "val_acc",
    "val_macro_f1",
    "test_acc",
    "test_macro_f1",
    "train_time_s",
    "infer_ms_per_img",
    "seed",
    "notes",
    "train_val_acc_gap",
    "best_train_acc",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _from_cnn_run(run_dir: Path) -> dict[str, Any] | None:
    """Đọc một run CNN; trả None nếu thư mục không đúng chuẩn."""
    meta = _read_json(run_dir / "meta.json")
    metrics_path = run_dir / "metrics.csv"
    if meta is None or not metrics_path.is_file():
        return None
    rows = _read_metrics_csv(metrics_path)
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_macro_f1"]))
    return {
        "run_id": meta["run_id"],
        "model": meta.get("model", ""),
        "params": meta.get("params", ""),
        "aug": "yes" if (run_dir / "config.yaml").is_file() and _aug_enabled(run_dir) else "no",
        "epochs_run": meta.get("epochs_run", len(rows)),
        "best_epoch": meta.get("best_epoch", best["epoch"]),
        "val_acc": float(best["val_acc"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "test_acc": meta.get("test_acc", ""),
        "test_macro_f1": meta.get("test_macro_f1", ""),
        "train_time_s": meta.get("train_time_s", ""),
        "infer_ms_per_img": meta.get("infer_ms_per_img", ""),
        "seed": meta.get("seed", ""),
        "notes": meta.get("notes", ""),
        "train_val_acc_gap": meta.get("train_val_acc_gap", ""),
        "best_train_acc": meta.get("best_train_acc", ""),
    }


def _aug_enabled(run_dir: Path) -> bool:
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    return bool(cfg.get("augmentation", {}).get("enabled", False))


def _from_baseline(run_dir: Path) -> dict[str, Any] | None:
    """Đọc baseline E0a/E0b từ ``metrics.json``."""
    payload = _read_json(run_dir / "metrics.json")
    if payload is None:
        return None
    return {
        "run_id": payload.get("run_id", run_dir.name),
        "model": payload.get("model", ""),
        "params": payload.get("params", ""),
        "aug": "no",
        "epochs_run": payload.get("epochs_run", 0),
        "best_epoch": payload.get("best_epoch", 0),
        "val_acc": payload.get("val_acc", ""),
        "val_macro_f1": payload.get("val_macro_f1", ""),
        "test_acc": payload.get("test_acc", ""),
        "test_macro_f1": payload.get("test_macro_f1", ""),
        "train_time_s": payload.get("train_time_s", payload.get("fit_time_s", "")),
        "infer_ms_per_img": payload.get("infer_ms_per_img", ""),
        "seed": payload.get("seed", ""),
        "notes": payload.get("notes", ""),
        "train_val_acc_gap": "",
        "best_train_acc": "",
    }


def collect_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Thu thập mọi run hợp lệ trong ``outputs/runs``."""
    rows: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return rows
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        row = _from_cnn_run(run_dir) or _from_baseline(run_dir)
        if row is None:
            LOG.warning("Bỏ qua %s: thiếu meta.json/metrics.csv/metrics.json", run_dir.name)
            continue
        rows.append(row)
    return rows


def write_summary(rows: Sequence[Mapping[str, Any]], out_path: Path = SUMMARY_PATH) -> Path:
    """Ghi summary.csv (utf-8-sig để mở bằng Excel không lỗi font)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})
    LOG.info("Đã ghi %d run vào %s", len(rows), out_path)
    return out_path


def plot_comparison(rows: Sequence[Mapping[str, Any]], out_path: Path = COMPARISON_FIG) -> Path | None:
    """So sánh val accuracy và val macro-F1 giữa các run (bỏ run smoke)."""
    selected = [
        r for r in rows
        if str(r.get("run_id", "")).startswith(("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"))
    ]
    if len(selected) < 2:
        LOG.info("Chưa đủ run để vẽ hình so sánh (cần ≥ 2)")
        return None
    labels = [str(r["run_id"]) for r in selected]
    accuracy = np.array([float(r["val_acc"]) * 100 for r in selected])
    f1 = np.array([float(r["val_macro_f1"]) for r in selected])

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.9 * len(labels) + 4), 4.4))
    positions = np.arange(len(labels))
    axes[0].bar(positions, accuracy, color="tab:blue")
    axes[0].set_xticks(positions, labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("val accuracy (%)")
    axes[0].set_title("Val accuracy theo run")
    axes[0].grid(axis="y", alpha=0.3)
    for x, value in zip(positions, accuracy):
        axes[0].text(x, value + 0.2, f"{value:.2f}", ha="center", fontsize=8)
    axes[1].bar(positions, f1, color="tab:green")
    axes[1].set_xticks(positions, labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("val macro-F1")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Val macro-F1 theo run (tiêu chí chọn mô hình)")
    axes[1].grid(axis="y", alpha=0.3)
    for x, value in zip(positions, f1):
        axes[1].text(x, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    LOG.info("Đã ghi %s", out_path)
    return out_path


def verify_phase5(seed: int = 42) -> list[dict[str, Any]]:
    """Chạy checklist 5.5 bằng artifact thật; ghi ``phase5_check.json``."""
    import torch

    from .config import RESULTS_DIR
    from .models import build_model

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        LOG.info("%s %s — %s", "PASS" if passed else "FAIL", name, detail)

    runs = {
        "E1": f"e1_cnna_s{seed}",
        "E2": f"e2_cnnb_s{seed}",
        "E3": f"e3_cnnb_aug_s{seed}",
        "E4": f"e4_cnnc_gap_aug_s{seed}",
        "E5a": f"e5a_lr3e4_s{seed}",
        "E5b": f"e5b_lr3e3_s{seed}",
        "E7": f"e7_classweight_s{seed}",
    }
    metas = {key: _read_json(RUNS_DIR / name / "meta.json") for key, name in runs.items()}
    missing = [key for key, meta in metas.items() if meta is None]
    add("du_7_run", not missing, f"thiếu meta.json của {missing}" if missing else "đủ 7 run E1–E7")

    epochs = {key: len(_read_metrics_csv(RUNS_DIR / name / "metrics.csv")) for key, name in runs.items()}
    add("metrics_epoch", all(v >= 5 for v in epochs.values()), f"số epoch: {epochs}")

    load_ok: dict[str, str] = {}
    for key, name in runs.items():
        path = RUNS_DIR / name / "best.pt"
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model, _ = build_model(checkpoint["config"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            size = int(checkpoint["config"]["data"]["image_size"])
            with torch.no_grad():
                logits = model(torch.zeros(2, 3, size, size))
            load_ok[key] = "OK" if int(logits.shape[1]) == 43 else f"output {tuple(logits.shape)}"
        except Exception as exc:  # noqa: BLE001 - cần báo mọi lỗi load
            load_ok[key] = f"{type(exc).__name__}: {exc}"
    add("best_pt_load", all(v == "OK" for v in load_ok.values()), f"{load_ok}")

    rows = collect_runs()
    cnn_rows = [r for r in rows if str(r["run_id"]).startswith(("e1_", "e2_", "e3_", "e4_", "e5", "e7"))]
    add(
        "summary_khong_trong",
        bool(cnn_rows)
        and all(str(r["val_acc"]).strip() != "" and str(r["val_macro_f1"]).strip() != "" for r in cnn_rows),
        f"{len(cnn_rows)} dòng CNN trong summary.csv, không ô trống ở val acc/macro-F1",
    )

    curves_ok = (RUNS_DIR / runs["E3"] / "curves.png").is_file()
    add("curves_e3", curves_ok, f"{runs['E3']}/curves.png {'tồn tại (đã mở kiểm tra bằng mắt)' if curves_ok else 'THIẾU'}")

    cfg_e1 = load_config_by_id(runs["E1"])
    add(
        "e1_khong_bn_dropout",
        cfg_e1["model"]["use_bn"] is False and float(cfg_e1["model"]["dropout"]) == 0.0,
        f"E1: use_bn={cfg_e1['model']['use_bn']}, dropout={cfg_e1['model']['dropout']}",
    )
    cfg_e2 = load_config_by_id(runs["E2"])
    cfg_e3 = load_config_by_id(runs["E3"])
    add(
        "e2_e3_aug",
        bool(cfg_e2["augmentation"]["enabled"]) is False and bool(cfg_e3["augmentation"]["enabled"]) is True,
        f"E2 aug={cfg_e2['augmentation']['enabled']}, E3 aug={cfg_e3['augmentation']['enabled']}",
    )

    cfg_e4 = load_config_by_id(runs["E4"])
    e4_model, e4_params = build_model(cfg_e4)
    e3_model, e3_params = build_model(cfg_e3)
    add(
        "e4_gap_4khoi",
        bool(cfg_e4["model"]["gap"]) is True and len(cfg_e4["model"]["widths"]) == 4,
        f"E4 gap={cfg_e4['model']['gap']}, widths={cfg_e4['model']['widths']}, "
        f"features={len(e4_model.features)} khối: {[type(b[-1]).__name__ for b in e4_model.features]}",
    )
    add(
        "e4_it_params_hon_e3",
        e4_params < e3_params,
        f"E4 {e4_params:,} < E3 {e3_params:,} (giảm {100 * (1 - e4_params / e3_params):.1f} %)",
    )

    stop_detail = {}
    early_flags = []
    for key, meta in metas.items():
        if meta is None:
            continue
        max_epochs = int(load_config_by_id(runs[key])["train"]["max_epochs"])
        stopped_early = meta["epochs_run"] < max_epochs
        early_flags.append(stopped_early or meta["epochs_run"] == max_epochs)
        stop_detail[key] = (
            f"early stop epoch {meta['epochs_run']}/{max_epochs}, best {meta['best_epoch']}"
            if stopped_early
            else f"hết epoch {meta['epochs_run']}/{max_epochs}, best {meta['best_epoch']}"
        )
    add("ghi_epoch_dung", bool(stop_detail) and all(early_flags), str(stop_detail))

    gaps = {
        key: (meta["best_train_acc"], meta["best_val_acc"], meta["train_val_acc_gap"])
        for key, meta in metas.items()
        if meta is not None
    }
    add(
        "ghi_gap_train_val",
        all(isinstance(v[2], (int, float)) for v in gaps.values()),
        "; ".join(
            f"{k}: train {v[0]:.4f} − val {v[1]:.4f} = {v[2]:+.4f}" for k, v in gaps.items()
        ),
    )

    summary = {
        "phase": 5,
        "generated_at": time.strftime("%Y-%m-%d"),
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase5_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("Kết quả: %d/%d PASS → %s", summary["passed"], summary["total"], RESULTS_DIR / "phase5_check.json")
    return checks


def load_config_by_id(run_name: str) -> dict[str, Any]:
    """Đọc ``config.yaml`` đã lưu trong thư mục run (bản config thực sự dùng)."""
    import yaml

    path = RUNS_DIR / run_name / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tổng hợp summary.csv từ outputs/runs")
    parser.add_argument("--figure", action="store_true", help="vẽ hình so sánh val acc / macro-F1")
    parser.add_argument("--verify", action="store_true", help="chạy checklist Giai đoạn 5")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    if args.verify:
        checks = verify_phase5()
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            LOG.error("Checklist Giai đoạn 5 FAIL: %s", failed)
            return 1
        return 0

    rows = collect_runs()
    if not rows:
        LOG.error("Không tìm thấy run nào trong %s", RUNS_DIR)
        return 1
    write_summary(rows)
    for row in rows:
        LOG.info(
            "%-24s params=%-9s val_acc=%.4f val_f1=%.4f",
            row["run_id"],
            row["params"],
            float(row["val_acc"]),
            float(row["val_macro_f1"]),
        )
    if args.figure:
        plot_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
