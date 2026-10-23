"""Đánh giá sâu và phân tích lỗi trên tập test (Giai đoạn 7).

Đầu vào: ``predictions.csv`` của lần đánh giá test duy nhất + ``test_report.json``.
Đầu ra (``outputs/``):

- ``results/per_class_test.csv`` — 43 lớp: precision, recall, F1, support
- ``results/top_confusions_test.csv`` — 10 cặp nhầm nhiều nhất
- ``results/error_rates_tertiles.csv`` — error rate theo tertile 3 biến
- ``results/ood_results.csv`` — dự đoán trên ảnh ngoài dataset
- ``results/threshold.json`` — ngưỡng unknown τ chọn trên **val**
- ``results/error_report.json`` — tổng hợp số liệu cho Chương 5
- ``figures/07_*.png`` — 6 hình

Cách dùng::

    python -m src.error_analysis --run outputs/runs/final_s42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, get_device
from .data import read_test_annotations
from .predict import Predictor, load_tau

LOG = logging.getLogger(__name__)

ANNOTATIONS = {Path(r["Filename"]).stem: r for r in read_test_annotations()}


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, str]]:
    """Đọc CSV UTF-8 (có/không BOM)."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def roi_crop(image: Image.Image, row: Mapping[str, str] | None, margin: float = 0.12) -> Image.Image:
    """Cắt quanh ROI (nếu có) và phóng to để hiển thị; ghi rõ đã phóng to."""
    rgb = image.convert("RGB")
    if row is None:
        return rgb
    width, height = int(row["Width"]), int(row["Height"])
    x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
    x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
    pad_x, pad_y = int((x2 - x1) * margin), int((y2 - y1) * margin)
    box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    crop = rgb.crop(box)
    scale = max(1, int(round(112 / max(crop.width, 1))))
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    return crop


def tertile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Chia 3 nhóm theo phân vị 33/67; trả nhãn 0/1/2 và 2 ngưỡng."""
    finite = values[np.isfinite(values)]
    q1, q2 = np.percentile(finite, [100 / 3, 200 / 3])
    labels = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
    return labels, [float(q1), float(q2)]


# ---------------------------------------------------------------------------
# Các phần phân tích
# ---------------------------------------------------------------------------


def per_class_outputs(report: Mapping[str, Any]) -> dict[str, Any]:
    """Bảng per-class + hình F1 sắp tăng dần (chú thích support)."""
    per_class = report["per_class"]
    rows = [
        {
            "class_id": int(c),
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1"],
            "support": v["support"],
        }
        for c, v in per_class.items()
    ]
    rows.sort(key=lambda r: r["f1"])
    out_csv = RESULTS_DIR / "per_class_test.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_id", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(13, 9))
    positions = np.arange(len(rows))
    colors = ["tab:red" if r["f1"] < 0.9 else "tab:blue" for r in rows]
    ax.barh(positions, [r["f1"] for r in rows], color=colors)
    ax.set_yticks(positions, [f"{r['class_id']} (n={r['support']})" for r in rows], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 trên test")
    ax.set_title("F1 theo lớp (sắp tăng dần) — đỏ: F1 < 0,90; nhãn kèm support")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_per_class_f1.png", dpi=140)
    plt.close(fig)
    weakest = rows[:15]
    return {"per_class_rows": rows, "weakest15": weakest, "n_below_090": sum(r["f1"] < 0.9 for r in rows)}


def confusion_zoom(report: Mapping[str, Any], weakest: Sequence[Mapping[str, Any]]) -> Path:
    """Zoom confusion matrix cho 15 lớp F1 thấp nhất."""
    ids = [r["class_id"] for r in weakest]
    matrix = np.asarray(report["confusion_matrix"], dtype=float)[np.ix_(ids, ids)]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="magma")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="tỷ lệ theo hàng")
    ax.set_xticks(range(len(ids)), ids, rotation=90)
    ax.set_yticks(range(len(ids)), ids)
    ax.set_xlabel("nhãn dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.set_title("Zoom 15 lớp F1 thấp nhất trên test (chuẩn hóa theo hàng)")
    for i in range(len(ids)):
        for j in range(len(ids)):
            if normalized[i, j] >= 0.05:
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if normalized[i, j] < 0.6 else "black")
    fig.tight_layout()
    out = FIGURES_DIR / "07_confusion_zoom15.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def top_pairs(predictions: Sequence[Mapping[str, str]], pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Lưu 10 cặp nhầm + chọn tối đa 3 ảnh ví dụ mỗi cặp, vẽ lưới 10×3."""
    rows_out, grid = [], []
    for pair in pairs:
        members = [
            r for r in predictions
            if int(r["true_id"]) == pair["true"] and int(r["pred_id"]) == pair["pred"]
        ][:3]
        rows_out.append({**pair, "n_examples": len(members)})
        grid.append((pair, members))

    with (RESULTS_DIR / "top_confusions_test.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["true", "pred", "count", "n_examples"])
        writer.writeheader()
        writer.writerows(rows_out)

    fig, axes = plt.subplots(len(grid), 3, figsize=(7.5, 2.1 * len(grid)))
    for row_index, (pair, members) in enumerate(grid):
        for col in range(3):
            ax = axes[row_index, col]
            ax.axis("off")
            if col < len(members):
                path = PROJECT_ROOT / members[col]["path"]
                with Image.open(path) as image:
                    crop = roi_crop(image, ANNOTATIONS.get(path.stem))
                ax.imshow(crop)
                ax.set_title(f"conf {float(members[col]['prob_pred']):.2f}", fontsize=7)
            if col == 0:
                ax.text(-0.05, 0.5, f"{pair['true']}→{pair['pred']}\n(n={pair['count']})",
                        transform=ax.transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Top 10 cặp nhầm trên test (mỗi cặp tối đa 3 ảnh, crop theo ROI, đã phóng to)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_top_pairs.png", dpi=140)
    plt.close(fig)
    return rows_out


def error_rate_tertiles(predictions: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Error rate theo tertile của blur / brightness / ROI ratio + bảng CSV + hình."""
    variables = {
        "blur_laplacian": "độ mờ (Laplacian, cao = nét)",
        "brightness_mean": "độ sáng (mean V)",
        "roi_area_ratio": "diện tích biển / ảnh",
    }
    labels = {r["path"]: r for r in predictions}
    arrays = {name: np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in predictions])
              for name in variables}
    correct = np.array([int(r["is_correct"]) for r in predictions])

    rows_csv, summary = [], {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, title) in zip(axes, variables.items()):
        groups, bounds = tertile_labels(arrays[name])
        rates = []
        for group in (0, 1, 2):
            mask = groups == group
            rate = float(1 - correct[mask].mean()) if mask.any() else float("nan")
            rates.append(rate)
            rows_csv.append({
                "variable": name,
                "group": group,
                "n": int(mask.sum()),
                "error_rate": round(rate, 6) if np.isfinite(rate) else "",
                "bound_low": round(bounds[0], 6),
                "bound_high": round(bounds[1], 6),
            })
        summary[name] = {
            "bounds": bounds,
            "n": [int((groups == g).sum()) for g in (0, 1, 2)],
            "error_rate": rates,
            "ratio_worst_best": float(max(rates) / max(min(rates), 1e-9)),
        }
        ax.bar(["nhóm 0", "nhóm 1", "nhóm 2"], rates, color=["tab:orange", "tab:blue", "tab:green"])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("error rate")
        ax.grid(axis="y", alpha=0.3)
        for index, rate in enumerate(rates):
            ax.text(index, rate, f"{rate*100:.2f}%", ha="center", va="bottom", fontsize=8)

    with (RESULTS_DIR / "error_rates_tertiles.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)

    # Tương quan point-biserial giữa lỗi và từng biến
    from scipy.stats import pointbiserialr

    correlations = {}
    for name in variables:
        values = arrays[name]
        mask = np.isfinite(values)
        if mask.sum() > 10:
            r, p = pointbiserialr(correct[mask], values[mask])
            correlations[name] = {"r": round(float(r), 4), "p_value": float(f"{p:.3g}")}
    fig.suptitle("Error rate theo tertile: mờ / sáng / diện tích biển (test, n=12.630)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_error_rates.png", dpi=140)
    plt.close(fig)
    return {"table": rows_csv, "summary": summary, "correlations": correlations}


def select_error_samples(predictions: Sequence[Mapping[str, str]], count: int = 12) -> list[dict[str, Any]]:
    """Chọn mẫu sai đa dạng: ưu tiên confidence cao, trải đều 4 nhóm nguyên nhân."""
    errors = [r for r in predictions if r["is_correct"] == "0"]
    errors.sort(key=lambda r: float(r["prob_pred"]), reverse=True)
    blur = np.array([float(r["blur_laplacian"]) for r in errors])
    bright = np.array([float(r["brightness_mean"]) for r in errors])
    roi = np.array([float(r["roi_area_ratio"]) if r["roi_area_ratio"] != "nan" else np.nan for r in errors])
    blur_groups, _ = tertile_labels(blur)
    bright_groups, _ = tertile_labels(bright)
    roi_groups, _ = tertile_labels(roi)

    def bucket(index: int) -> str:
        if blur_groups[index] == 0:
            return "E-A mờ"
        if bright_groups[index] == 0:
            return "E-B tối"
        if np.isfinite(roi[index]) and roi_groups[index] == 0:
            return "E-C biển nhỏ"
        return "E-E lớp giống nhau"

    # Chia theo 4 nhóm nguyên nhân rồi lấy vòng tròn (round-robin) để mẫu đa dạng,
    # trong mỗi nhóm ưu tiên confidence cao nhất (lỗi "tự tin sai").
    buckets: dict[str, list[tuple[int, Mapping[str, str]]]] = {}
    for index, row in enumerate(errors):
        buckets.setdefault(bucket(index), []).append((index, row))
    order = ["E-A mờ", "E-B tối", "E-C biển nhỏ", "E-E lớp giống nhau"]
    chosen: list[dict[str, Any]] = []
    for round_index in range(count):
        added = False
        for name in order:
            if len(chosen) >= count:
                break
            items = buckets.get(name, [])
            if round_index < len(items):
                index, row = items[round_index]
                chosen.append({"row": row, "bucket": name, "index": index})
                added = True
        if not added:
            break
    return chosen[:count]


def error_samples_grid(predictions: Sequence[Mapping[str, str]], chosen: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Vẽ lưới mẫu sai (4 cột) + trả metadata để viết nhận xét."""
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows))
    axes = np.atleast_2d(axes)
    metadata = []
    for position, item in enumerate(chosen):
        row = item["row"]
        ax = axes[position // cols, position % cols]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            crop = roi_crop(image, ANNOTATIONS.get(path.stem))
        ax.imshow(crop)
        ax.axis("off")
        ax.set_title(
            f"{item['bucket']}\n{row['true_name'][:24]} → {row['pred_name'][:24]}\n"
            f"conf {float(row['prob_pred']):.2f} | blur {float(row['blur_laplacian']):.3f} | "
            f"bright {float(row['brightness_mean']):.2f}",
            fontsize=7,
        )
        metadata.append(
            {
                "path": row["path"],
                "true_id": int(row["true_id"]),
                "true_name": row["true_name"],
                "pred_id": int(row["pred_id"]),
                "pred_name": row["pred_name"],
                "prob_pred": float(row["prob_pred"]),
                "margin": float(row["margin"]),
                "blur_laplacian": float(row["blur_laplacian"]),
                "brightness_mean": float(row["brightness_mean"]),
                "roi_area_ratio": float(row["roi_area_ratio"]) if row["roi_area_ratio"] != "nan" else float("nan"),
                "bucket": item["bucket"],
            }
        )
    for position in range(len(chosen), rows * cols):
        axes[position // cols, position % cols].axis("off")
    fig.suptitle("Mẫu dự đoán sai trên test — crop theo ROI, đã phóng to", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_error_samples.png", dpi=140)
    plt.close(fig)
    return metadata


def choose_threshold(val_predictions: Sequence[Mapping[str, str]], percentile: float = 5.0) -> dict[str, Any]:
    """Chọn τ trên **val**: phân vị ``percentile`` của max-prob → giữ ~95 % in-distribution."""
    max_prob = np.array([float(r["prob_pred"]) for r in val_predictions])
    tau = float(np.percentile(max_prob, percentile))
    payload = {
        "tau": tau,
        "percentile": percentile,
        "acceptance_in_distribution": float((max_prob >= tau).mean()),
        "n_val": int(len(max_prob)),
        "chosen_on": "val stratified_group (final_s42), KHÔNG dùng test/demo",
        "method": "τ = phân vị 5% của max-prob trên val; reject nếu max-prob < τ",
    }
    (RESULTS_DIR / "threshold.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ood_evaluation(manifest: Path, predictor: Predictor, tau: float) -> dict[str, Any]:
    """Chạy model trên ảnh ngoài dataset; lưu CSV + hình + thống kê rejection."""
    rows = read_rows(manifest)
    results = []
    for row in rows:
        path = PROJECT_ROOT / "external_test" / row["filename"]
        prediction = predictor.predict(Image.open(path), k=3, tau=tau)
        expected = row["expected_label"].strip()
        results.append(
            {
                "filename": row["filename"],
                "scenario": row["scenario"],
                "expected_id": expected,
                "pred_id": prediction["topk"][0]["class_id"],
                "pred_name": prediction["topk"][0]["name_vi"],
                "prob": round(prediction["max_prob"], 6),
                "top3": " | ".join(f"{t['class_id']}:{t['prob']:.3f}" for t in prediction["topk"]),
                "correct": (
                    "yes" if expected and int(expected) == prediction["topk"][0]["class_id"]
                    else ("no" if expected else "n/a")
                ),
                "rejected": prediction["unknown"],
                "source": row["source"],
                "license": row["license"],
            }
        )
    with (RESULTS_DIR / "ood_results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, row, item in zip(axes.ravel(), rows, results):
        path = PROJECT_ROOT / "external_test" / row["filename"]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.axis("off")
        status = "KHÔNG XÁC ĐỊNH" if item["rejected"] else item["pred_name"]
        ax.set_title(f"{row['filename'][:22]}\n{status} ({item['prob']:.2f})", fontsize=7)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"Ảnh ngoài dataset (CC0/PD) — τ = {tau:.3f} (chọn trên val)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_ood.png", dpi=140)
    plt.close(fig)

    no_sign = [r for r in results if r["scenario"] == "4"]
    with_expected = [r for r in results if r["correct"] in ("yes", "no")]
    return {
        "n_images": len(results),
        "n_no_sign": len(no_sign),
        "n_with_expected_label": len(with_expected),
        "n_correct": sum(r["correct"] == "yes" for r in results),
        "rejection_rate_no_sign": (
            float(np.mean([r["rejected"] for r in no_sign])) if no_sign else None
        ),
        "acceptance_rate_on_signs": (
            float(np.mean([not r["rejected"] for r in results if r["scenario"] != "4"]))
        ),
        "rows": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(run_dir: Path) -> dict[str, Any]:
    """Chạy toàn bộ Giai đoạn 7 trên artifact test của ``run_dir``."""
    report = json.loads((RESULTS_DIR / "test_report.json").read_text(encoding="utf-8"))
    predictions = read_rows(run_dir / "eval_test" / "predictions.csv")
    val_predictions = read_rows(run_dir / "eval_val" / "predictions.csv")

    threshold = choose_threshold(val_predictions)
    per_class = per_class_outputs(report)
    confusion_zoom(report, per_class["weakest15"])
    pairs = top_pairs(predictions, report["top_confusions"])
    tertiles = error_rate_tertiles(predictions)
    chosen = select_error_samples(predictions, count=12)
    samples = error_samples_grid(predictions, chosen)

    predictor = Predictor(run_dir, tau=threshold["tau"])
    ood = ood_evaluation(PROJECT_ROOT / "external_test" / "manifest.csv", predictor, threshold["tau"])

    payload = {
        "run_id": run_dir.name,
        "threshold": threshold,
        "n_test": report["n_samples"],
        "accuracy": report["accuracy"],
        "macro_f1_43": report["macro_f1_43"],
        "top3_accuracy": report["top3_accuracy"],
        "n_classes_f1_below_090": per_class["n_below_090"],
        "weakest15": per_class["weakest15"],
        "top_confusions": pairs,
        "tertiles": tertiles,
        "samples": samples,
        "ood": ood,
    }
    (RESULTS_DIR / "error_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info(
        "GĐ7 xong: %d lớp F1<0,90 | top cặp %s→%s (%d) | τ=%.3f | OOD %d ảnh, reject no-sign %.2f",
        per_class["n_below_090"],
        pairs[0]["true"], pairs[0]["pred"], pairs[0]["count"],
        threshold["tau"], ood["n_images"],
        ood["rejection_rate_no_sign"] or 0.0,
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích lỗi trên test (Giai đoạn 7)")
    parser.add_argument("--run", type=str, default="outputs/runs/final_s42")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
