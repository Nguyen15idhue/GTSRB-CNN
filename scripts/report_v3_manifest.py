"""Sinh manifest (sha256 + kích thước + mục đích) cho report pack v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results" / "v3"
FIGURES = ROOT / "outputs" / "figures" / "v3"

PURPOSE = {
    "00_corrections.csv": "Các số cũ sai/lệch và số đúng kèm bằng chứng",
    "01_class_catalog.csv": "Danh mục 43 lớp: tên, số ảnh pool/test/lockbox, %",
    "02_split_summary.csv": "Bảng các tầng dữ liệu: ảnh, track, sign-group, seed, vai trò",
    "03_data_quality_per_class.csv": "Chất lượng ảnh theo lớp: blur, sáng, tương phản, ROI",
    "04_data_quality_summary.json": "Tổng hợp toàn vẹn/chất lượng dữ liệu pool và test",
    "05_model_comparison.csv": "Bảng so sánh một chỗ: baseline, factorial, kiến trúc, hyperparam, aug, OOF",
    "06_eval_sets_summary.csv": "Bốn tập đánh giá: n, accuracy, macro-F1, CI, top-3, ECE, Brier",
    "06b_lockbox_uncertainty.csv": "Bất định lockbox: sd 3 seed, binomial CI cho accuracy",
    "06c_oof_accuracy_check.json": "Đính chính OOF accuracy 98,004% → 97,924%",
    "07_per_class_lockbox.csv": "Precision/Recall/F1 43 lớp trên lockbox (mean 3 seed)",
    "08_per_class_oof.csv": "Precision/Recall/F1 43 lớp trên OOF (mean 3 seed)",
    "09_top_confusions_lockbox.csv": "Các cặp nhầm nhiều nhất trên lockbox",
    "10_confusion_lockbox_counts.csv": "Confusion matrix 43×43 (số đếm, mean 3 seed)",
    "10_confusion_lockbox_normalized.csv": "Confusion matrix 43×43 (chuẩn hóa hàng)",
    "11_oof_by_fold_seed.csv": "OOF từng fold × seed × cấu hình",
    "12_calibration_confidence.csv": "ECE, Brier, confidence đúng/sai, lỗi tự tin cao",
    "13_inference_benchmark.csv": "Latency/throughput CPU-GPU theo batch, VRAM, RAM",
    "13_inference_benchmark.json": "Metadata benchmark + kích thước checkpoint/params",
    "14_hardware_software.json": "Phần cứng, phần mềm, ổ đĩa, phiên bản thư viện",
    "15_deployment_status.csv": "Đã có/chưa có của demo và khoảng trống triển khai",
    "16_gradcam_roi_mass.csv": "Grad-CAM: tỷ lệ mass trong ROI từng ảnh",
    "16_gradcam_summary.json": "Tổng hợp Grad-CAM đúng/sai so với mức ngẫu nhiên",
    "17_manual_error_review.csv": "Toàn bộ lỗi screening val của D2 + cột trống để người rà soát",
    "17b_manual_error_review_diverse.csv": "Bản rút gọn: tối đa 2 ảnh/track để rà soát thủ công",
    "18_metric_worked_example.csv": "Ví dụ tính Accuracy vs Macro-F1 trên 43 lớp",
    "19_training_cost_by_group.csv": "Thời gian/epoch/GPU mem theo nhóm thực nghiệm",
    "20_experiment_index.csv": "Danh sách 111 run CNN + baseline, đọc được",
    "21_lockbox_by_seed.csv": "Kết quả lockbox từng seed",
    "22_test_per_class_raw_clean.csv": "Precision/Recall/F1 43 lớp trên official test raw/clean",
    "23_architecture_spec.csv": "Thông số từng lớp của 8 cấu hình CNN",
    "24_results_evidence.csv": "Kết quả nào tốt/không đủ bằng chứng/kém + bằng chứng",
    "25_corruption_summary.csv": "Accuracy/F1 trung bình 3 mức corruption theo cấu hình",
    "26_evaluation_time.json": "Thời gian đánh giá 6.180 ảnh CPU/GPU",
    "28_glossary.csv": "Thuật ngữ cho người không chuyên",
    "threshold_v2_oof.json": "Ngưỡng unknown v2 chọn trên OOF (τ=0,9590)",
    "training_cost_totals.json": "Tổng thời gian/epoch của 111 run + baseline",
    "metric_worked_example.json": "Tóm tắt ví dụ Accuracy vs Macro-F1",
    "class_catalog_summary.json": "Tóm tắt phân bố 43 lớp",
    "confusion_lockbox_summary.json": "Tóm tắt confusion lockbox + top cặp",
}

FIG_PURPOSE = {
    "01_class_distribution.png": "Phân bố 43 lớp + mức mất cân bằng",
    "02_split_diagram.png": "Sơ đồ chia tập 3 tầng",
    "03_architecture_d2.png": "Kiến trúc D2, params từng khối",
    "04_learning_curves.png": "Learning curve train vs validation (D2, F07)",
    "05_model_comparison.png": "So sánh Accuracy và Macro-F1 các cấu hình",
    "06_confusion_matrix.png": "Confusion matrix lockbox + top cặp nhầm",
    "07_per_class_f1.png": "F1 từng lớp trên lockbox",
    "08_factor_effects.png": "Hiệu ứng factorial + tương tác BN×Aug",
    "09_oof_fold_seed.png": "OOF theo fold × seed + phân rã phương sai",
    "10_robustness.png": "Robustness 6 corruption × 3 mức",
    "11_gradcam.png": "Grad-CAM đúng/sai/thấp confidence",
    "12_calibration.png": "Reliability, confidence histogram, selective risk",
    "13_precision_recall.png": "Precision–Recall và AP từng lớp",
    "14_inference_perf.png": "Latency/throughput CPU vs GPU",
    "15_prediction_examples.png": "Ví dụ dự đoán đúng/sai theo confidence",
    "16_data_quality.png": "Phân bố độ nét/sáng/tương phản",
    "16b_data_quality_examples.png": "Ảnh cực trị chất lượng",
    "17_class_samples.png": "Mỗi lớp một ảnh mẫu rõ",
    "contact_sheets/error_sheet_01.png": "Phiếu rà soát lỗi (trang 1)",
    "contact_sheets/error_sheet_02.png": "Phiếu rà soát lỗi (trang 2)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(base: Path, purpose: dict[str, str], skip_data: bool = False) -> list[dict]:
    rows = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(base).as_posix()
        if skip_data and rel.startswith("data/"):
            continue
        rows.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "purpose": purpose.get(rel, ""),
            }
        )
    return rows


def main() -> int:
    manifest = {
        "pack": "report_v3",
        "created_at": "2026-11-04",
        "description": "Tài nguyên cho bản báo cáo làm lại; không chạy lại train, không mở lockbox.",
        "results_root": "outputs/results/v3",
        "figures_root": "outputs/figures/v3",
        "results": collect(RESULTS, PURPOSE, skip_data=True),
        "figures": collect(FIGURES, FIG_PURPOSE),
        "data_local_only": collect(RESULTS / "data", {}),
        "data_note": "data/ chỉ dùng cục bộ, không commit (theo .gitignore *.npy/*.npz); sinh lại bằng scripts/report_v3_data.py.",
        "source_artifacts": [
            "outputs/results/v2/",
            "outputs/splits/v2/",
            "outputs/runs/v2_*",
            "outputs/figures/v2/",
        ],
        "scripts": [
            "scripts/report_v3_data.py",
            "scripts/report_v3_figures.py",
            "scripts/report_v3_meta.py",
            "scripts/report_v3_manifest.py",
        ],
    }
    path = RESULTS / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "ghi", path.relative_to(ROOT),
        "results", len(manifest["results"]),
        "figures", len(manifest["figures"]),
        "data_local", len(manifest["data_local_only"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
