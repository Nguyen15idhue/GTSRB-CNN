"""Tổng hợp chi phí huấn luyện v2 từ artifact (không nhập tay số nào).

Đọc:
* ``outputs/results/v2/summary.csv`` (111 run CNN)
* ``outputs/runs/v2_*/metrics.csv`` (epoch time, GPU mem)
* ``outputs/results/v2/baseline.json`` (B1)

Ghi:
* ``outputs/results/v2/training_cost.csv`` (từng run)
* ``outputs/results/v2/training_cost_summary.json``
* ``outputs/results/v2/tables/training_cost.md``

Cách dùng::

    python scripts/training_cost_v2.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "results" / "v2"
RUNS = ROOT / "outputs" / "runs"


def group_of(row: dict) -> str:
    split = row["split"]
    if split == "screening":
        return row["block"] or "screening"
    if split.startswith("fold"):
        return "X_confirmation"
    if split == "dev_all":
        return "FINAL"
    return split


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader((OUT / "summary.csv").open(encoding="utf-8")))
    per_run = []
    for row in rows:
        run_dir = RUNS / row["run_id"]
        metrics_path = run_dir / "metrics.csv"
        epoch_times, gpu = [], []
        if metrics_path.is_file():
            with metrics_path.open(newline="", encoding="utf-8") as fh:
                for metric in csv.DictReader(fh):
                    if metric.get("epoch_time_s"):
                        epoch_times.append(float(metric["epoch_time_s"]))
                    if metric.get("gpu_mem_mb"):
                        gpu.append(float(metric["gpu_mem_mb"]))
        per_run.append({
            "run_id": row["run_id"],
            "config_id": row["config_id"],
            "seed": int(row["seed"]),
            "fold": row["fold"],
            "split": row["split"],
            "group": group_of(row),
            "n_train": int(json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["n_train"]),
            "params": int(row["params"]),
            "epochs_run": int(row["epochs_run"]),
            "best_epoch": int(row["best_epoch"]),
            "train_time_s": float(row["train_time_s"]),
            "mean_epoch_s": statistics.mean(epoch_times) if epoch_times else 0.0,
            "max_gpu_mem_mb": max(gpu) if gpu else 0.0,
            "infer_ms_per_img": float(row["infer_ms_per_img"]),
        })

    with (OUT / "training_cost.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)

    def aggregate(items: list[dict]) -> dict:
        times = [r["train_time_s"] for r in items]
        epochs = sum(r["epochs_run"] for r in items)
        return {
            "n_runs": len(items),
            "total_epochs": epochs,
            "total_train_s": sum(times),
            "total_train_min": sum(times) / 60,
            "mean_train_s": statistics.mean(times),
            "median_train_s": statistics.median(times),
            "min_train_s": min(times),
            "max_train_s": max(times),
            "mean_epoch_s_weighted": sum(times) / epochs if epochs else 0.0,
            "max_gpu_mem_mb": max(r["max_gpu_mem_mb"] for r in items),
        }

    by_group: dict[str, list[dict]] = defaultdict(list)
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in per_run:
        by_group[row["group"]].append(row)
        if row["split"] == "screening":
            by_config[row["config_id"]].append(row)

    summary = {
        "total": aggregate(per_run),
        "by_group": {name: aggregate(items) for name, items in sorted(by_group.items())},
        "by_config": {name: aggregate(items) for name, items in sorted(by_config.items())},
    }

    baseline = json.loads((OUT / "baseline.json").read_text(encoding="utf-8"))
    summary["baseline_b1"] = {
        "feature_time_s": baseline["feature_time_s"],
        "fit_time_s": baseline["fit_time_s"],
        "infer_ms_per_img": baseline["infer_ms_per_img"],
        "total_s": baseline["feature_time_s"] + baseline["fit_time_s"],
    }
    summary["grand_total_s"] = summary["total"]["total_train_s"] + summary["baseline_b1"]["total_s"]
    (OUT / "training_cost_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "| Nhóm | Run | Tổng epoch | Tổng thời gian (phút) | TB/run (s) | Trung vị (s) | Min–Max (s) | TB/epoch (s) | GPU mem max (MB) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name, agg in summary["by_group"].items():
        lines.append(
            f"| {name} | {agg['n_runs']} | {agg['total_epochs']} | {agg['total_train_min']:.1f} | "
            f"{agg['mean_train_s']:.1f} | {agg['median_train_s']:.1f} | "
            f"{agg['min_train_s']:.0f}–{agg['max_train_s']:.0f} | {agg['mean_epoch_s_weighted']:.2f} | "
            f"{agg['max_gpu_mem_mb']:.0f} |"
        )
    total = summary["total"]
    lines.append(
        f"| **Tổng CNN** | **{total['n_runs']}** | **{total['total_epochs']}** | **{total['total_train_min']:.1f}** | "
        f"{total['mean_train_s']:.1f} | {total['median_train_s']:.1f} | "
        f"{total['min_train_s']:.0f}–{total['max_train_s']:.0f} | {total['mean_epoch_s_weighted']:.2f} | "
        f"{total['max_gpu_mem_mb']:.0f} |"
    )
    lines.append(
        f"| B1 HOG+SVC | 1 | – | {summary['baseline_b1']['total_s']/60:.1f} | "
        f"{summary['baseline_b1']['total_s']:.1f} | – | – | – | – |"
    )
    (OUT / "tables" / "training_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nTổng thời gian epoch loop: {total['total_train_s']/3600:.2f} h")
    print(f"Tổng gồm B1: {summary['grand_total_s']/3600:.2f} h")
    print("Theo cấu hình:")
    for name, agg in summary["by_config"].items():
        print(f"  {name}: {agg['mean_train_s']:.1f}s/run, {agg['mean_epoch_s_weighted']:.2f}s/epoch, "
              f"best_epoch TB {sum(r['best_epoch'] for r in by_config[name])/agg['n_runs']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
