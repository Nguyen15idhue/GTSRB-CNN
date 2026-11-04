"""Bảng metadata cho báo cáo v3: phần cứng/phần mềm, trạng thái triển khai, Grad-CAM."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "results" / "v3"
TABLES = OUT / "tables"
V2 = ROOT / "outputs" / "results" / "v2"
ENV = ROOT / "outputs" / "results" / "env.json"


def write_json(name: str, payload: Any) -> None:
    path = TABLES / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ghi", path.relative_to(ROOT))


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = TABLES / name
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("ghi", path.relative_to(ROOT), len(rows), "dong")


def hardware_software() -> None:
    env = json.loads(ENV.read_text(encoding="utf-8"))
    import matplotlib
    import PIL
    import sklearn
    import torch
    import torchvision

    payload = {
        "do_do_tren": "máy train của dự án (không phải máy demo cấu hình thấp)",
        "os": {
            "caption": "Microsoft Windows 11 Pro",
            "version": "10.0.26200",
            "build": "26200",
            "arch": "64-bit",
            "python_platform": platform.platform(),
        },
        "cpu": {
            "name": "Intel Core i9-14900KF",
            "cores": 24,
            "logical_processors": 32,
            "ram_gb": 63.84,
        },
        "gpu": {
            "name": env.get("gpu_name", "NVIDIA GeForce RTX 5070"),
            "vram_gb": env.get("gpu_total_memory_gb", 11.94),
            "compute_capability": env.get("gpu_capability", [12, 0]),
            "driver": env.get("driver_version", "610.88"),
            "cuda_build": env.get("cuda_build", "12.8"),
            "cudnn": env.get("cudnn", "91900"),
        },
        "disk": {
            "model": "Lexar SSD NM710 2TB",
            "type": "SSD NVMe",
            "capacity_gb": round(__import__("shutil").disk_usage("C:/").total / 1024**3, 1),
            "free_gb": round(__import__("shutil").disk_usage("C:/").free / 1024**3, 1),
            "note": "ổ chứa data/cache/checkpoint; không phải yếu tố được đo trong benchmark",
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "repo_commit_at_env_capture": env.get("repo_commit", ""),
        "env_json": "outputs/results/env.json",
        "note": "Số liệu phần cứng lấy từ env.json + truy vấn hệ thống ngày 04/11/2026.",
    }
    write_json("14_hardware_software.json", payload)


def deployment_status() -> None:
    e2e = json.loads((ROOT / "outputs/results/e2e_ui.json").read_text(encoding="utf-8"))
    golden = json.loads((ROOT / "outputs/results/golden_compare.json").read_text(encoding="utf-8"))
    rows = [
        {
            "feature": "Giao diện web (Streamlit)",
            "status": "đã có",
            "evidence": "app.py; outputs/figures/11_ui_*.png",
            "gap": "không có",
        },
        {
            "feature": "Tải 1 ảnh lên và dự đoán top-3",
            "status": "đã có",
            "evidence": f"e2e_ui.json {e2e['passed']}/{e2e['total']} check pass",
            "gap": "chỉ 1 ảnh mỗi lần",
        },
        {
            "feature": "Hiển thị tên tiếng Việt + confidence",
            "status": "đã có",
            "evidence": "app.py render_prediction; e2e_ui.json",
            "gap": "không có",
        },
        {
            "feature": "Cảnh báo 'không xác định' theo ngưỡng τ",
            "status": "đã có (τ v2)",
            "evidence": "outputs/results/v3/tables/threshold_v2_oof.json τ=0.9590 chọn trên OOF X2",
            "gap": "không có",
        },
        {
            "feature": "Xử lý file lỗi / quá 20 MB",
            "status": "đã có",
            "evidence": "e2e_ui.json check loi_file_hong, loi_qua_20mb",
            "gap": "không có",
        },
        {
            "feature": "Chọn ảnh mẫu",
            "status": "đã có",
            "evidence": "e2e_ui.json check chon_anh_mau",
            "gap": "không có",
        },
        {
            "feature": "Nhiều ảnh / batch trên giao diện",
            "status": "chưa có",
            "evidence": "app.py dùng st.file_uploader đơn",
            "gap": "cần phát triển nếu muốn demo batch",
        },
        {
            "feature": "API cho hệ thống khác",
            "status": "chưa có",
            "evidence": "không có endpoint trong repo",
            "gap": "ngoài phạm vi bài tập",
        },
        {
            "feature": "Lưu lịch sử dự đoán",
            "status": "chưa có",
            "evidence": "không có DB/session state lưu lịch sử",
            "gap": "ngoài phạm vi bài tập",
        },
        {
            "feature": "Checkpoint D2/v2 (X2)",
            "status": "đã tích hợp",
            "evidence": "app.py mặc định outputs/runs/v2_final_s42; sidebar đọc app_summary_v2.json",
            "gap": "không có",
        },
        {
            "feature": "Golden test 20 ảnh giữa 2 môi trường",
            "status": "đã có (legacy)",
            "evidence": f"golden_compare.json max_prob_diff={golden['max_probability_difference']}, pass={golden['passed']}",
            "gap": "chưa chạy lại cho checkpoint v2",
        },
        {
            "feature": "Kiểm thử E2E Playwright",
            "status": "đã có",
            "evidence": f"e2e_ui.json {e2e['passed']}/{e2e['total']} pass; outputs/figures/11_ui_*.png",
            "gap": "không có",
        },
        {
            "feature": "Kiểm thử trên máy cấu hình thấp",
            "status": "chưa có",
            "evidence": "chỉ đo CPU trên máy train (i9-14900KF)",
            "gap": "cần máy khác hoặc giới hạn thread/RAM",
        },
        {
            "feature": "Kiểm thử camera thật / video",
            "status": "chưa có",
            "evidence": "external_test chỉ có ảnh tĩnh đã ẩn danh",
            "gap": "ngoài phạm vi và cần thu thập dữ liệu",
        },
        {
            "feature": "Tuyên bố an toàn cho giao thông thực tế",
            "status": "không",
            "evidence": "footer app.py và báo cáo đều ghi cảnh báo",
            "gap": "không được tuyên bố",
        },
    ]
    write_csv(
        "15_deployment_status.csv",
        rows,
        ["feature", "status", "evidence", "gap"],
    )


def gradcam_summary() -> None:
    df = pd.read_csv(TABLES / "16_gradcam_roi_mass.csv")
    by_outcome = {}
    for outcome, part in df.groupby("outcome"):
        by_outcome[outcome] = {
            "n": int(len(part)),
            "mass_in_roi_mean": float(part["cam_mass_in_roi"].mean()),
            "mass_in_roi_sd": float(part["cam_mass_in_roi"].std(ddof=1)),
            "mass_in_roi_median": float(part["cam_mass_in_roi"].median()),
        }
    payload = {
        "model": "D2 seed 42 trên screening val",
        "n_images": int(len(df)),
        "roi_area_ratio_mean": float(df["roi_area_ratio"].mean()),
        "by_outcome": by_outcome,
        "chance_level_note": "Nếu CAM phân bố đều, mass trong ROI xấp xỉ diện tích ROI (~0,56).",
        "interpretation": (
            "CAM tập trung trong ROI nhiều hơn mức ngẫu nhiên ở cả ca đúng và sai; "
            "ca sai vẫn nhìn vào biển nên lỗi chủ yếu là phân biệt lớp, không phải nhìn nền."
        ),
    }
    write_json("16_gradcam_summary.json", payload)


def lockbox_uncertainty() -> None:
    lock = json.loads((V2 / "lockbox_result.json").read_text(encoding="utf-8"))
    acc = lock["summary"]["accuracy"]
    f1 = lock["summary"]["macro_f1_43"]
    n = int(lock["n_images"])
    p = float(acc["mean"])
    se = (p * (1 - p) / n) ** 0.5
    rows = [
        {
            "metric": "accuracy",
            "mean": p,
            "sd_across_seeds": float(acc["sd"]),
            "seed_min": float(min(acc["values"])),
            "seed_max": float(max(acc["values"])),
            "ci95_low": p - 1.96 * se,
            "ci95_high": p + 1.96 * se,
            "ci_method": "binomial normal approx, n=8302 ảnh",
            "note": "CI này chỉ cho accuracy; không dùng cho macro-F1.",
        },
        {
            "metric": "macro_f1_43",
            "mean": float(f1["mean"]),
            "sd_across_seeds": float(f1["sd"]),
            "seed_min": float(min(f1["values"])),
            "seed_max": float(max(f1["values"])),
            "ci95_low": "",
            "ci95_high": "",
            "ci_method": "không có CI track-bootstrap trong artifact one-shot",
            "note": "Chỉ báo mean ± sd và range 3 seed; không tái sử dụng CI của OOF.",
        },
    ]
    write_csv(
        "06b_lockbox_uncertainty.csv",
        rows,
        ["metric", "mean", "sd_across_seeds", "seed_min", "seed_max", "ci95_low", "ci95_high", "ci_method", "note"],
    )


def oof_accuracy_check() -> None:
    df = pd.read_csv(V2 / "oof_predictions.csv")
    x2 = df[df["config_id"] == "x2"]
    per_seed = {
        int(seed): float((part["true_label"] == part["pred_label"]).mean())
        for seed, part in x2.groupby("seed")
    }
    payload = {
        "old_report_value_pct": 98.004,
        "old_report_claim": "OOF accuracy 98,004%",
        "recomputed_oof_accuracy_pct": float(np.mean(list(per_seed.values())) * 100),
        "per_seed_pct": {str(k): v * 100 for k, v in per_seed.items()},
        "screening_d2_accuracy_pct": 98.0043,
        "conclusion": (
            "98,004% là accuracy screening val của D2, không phải OOF. "
            "OOF accuracy đúng là 97,924% (pooled 30.899 ảnh, trung bình 3 seed)."
        ),
        "evidence": "outputs/results/v2/oof_predictions.csv; outputs/results/v2/summary.csv",
    }
    write_json("06c_oof_accuracy_check.json", payload)


def corrections() -> None:
    rows = [
        {
            "item": "OOF accuracy",
            "old_value": "98,004%",
            "new_value": "97,924%",
            "evidence": "06c_oof_accuracy_check.json",
            "note": "98,004% là accuracy screening của D2; OOF pooled là 97,924%.",
        },
        {
            "item": "Lockbox CI của macro-F1",
            "old_value": "[95,0511; 97,9422]",
            "new_value": "mean 97,5981; sd 0,6246; range 3 seed [96,8770; 97,9706]",
            "evidence": "06b_lockbox_uncertainty.csv",
            "note": "CI cũ là CI của OOF, không phải lockbox.",
        },
        {
            "item": "Số lỗi screening val",
            "old_value": "190 lỗi / 6.180 ảnh",
            "new_value": "F15: 190 lỗi; D2: 144 lỗi / 23 track",
            "evidence": "17_manual_error_review_summary.json",
            "note": "190 là của F15 (control), không phải D2 (winner).",
        },
        {
            "item": "Latency",
            "old_value": "0,93 ms/ảnh (không nêu batch/thiết bị)",
            "new_value": "GPU batch 1: 0,86 ms model-only; 1,24 ms end-to-end Python",
            "evidence": "13_inference_benchmark.csv",
            "note": "Phải nêu thiết bị, batch, FP32 và phạm vi đo.",
        },
        {
            "item": "Ngưỡng unknown của demo",
            "old_value": "τ = 0,9462 trên val split_random",
            "new_value": "τ v2 = 0,9590 trên OOF X2, đã dùng trong app.py",
            "evidence": "threshold_v2_oof.json",
            "note": "Demo đã cập nhật; không còn dùng τ 0,9462.",
        },
        {
            "item": "Checkpoint demo",
            "old_value": "final_s42 (legacy)",
            "new_value": "v2_final_s42 / X2 (D2, 1.448.779 params), đã tích hợp",
            "evidence": "15_deployment_status.csv; app.py",
            "note": "app.py mặc định v2_final_s42; sidebar đọc app_summary_v2.json.",
        },
        {
            "item": "Thời gian huấn luyện",
            "old_value": "4,13 giờ",
            "new_value": "3,99 giờ epoch-loop (111 run) + 8,2 phút baseline",
            "evidence": "training_cost_totals.json",
            "note": "Không gồm nạp cache, dự đoán cuối, phân tích, hình, báo cáo.",
        },
        {
            "item": "Depth: CI và Holm",
            "old_value": "+3,212 [+0,036; +4,440], p Holm 0,373",
            "new_value": "giữ nguyên; báo 'có bằng chứng vừa, chưa đủ sau Holm'",
            "evidence": "main_comparisons.csv",
            "note": "Không được viết 'depth tốt hơn' như kết luận chắc chắn.",
        },
    ]
    write_csv(
        "00_corrections.csv",
        rows,
        ["item", "old_value", "new_value", "evidence", "note"],
    )


def corruption_summary() -> None:
    df = pd.read_csv(V2 / "corruption.csv")
    table = df.groupby(["config_id", "corruption"]).agg(
        accuracy_mean=("accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
    ).reset_index()
    clean = {"f07": 96.501, "f15": 93.946, "g1": 92.251, "g2": 97.417, "g3": 95.264, "d2": 97.158}
    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "config_id": row["config_id"],
                "corruption": row["corruption"],
                "accuracy_mean_pct": row["accuracy_mean"] * 100,
                "macro_f1_mean_pct": row["macro_f1_mean"] * 100,
                "clean_macro_f1_pct": clean.get(row["config_id"], ""),
                "drop_vs_clean_pct": (clean.get(row["config_id"], np.nan) - row["macro_f1_mean"] * 100)
                if row["config_id"] in clean
                else "",
            }
        )
    write_csv(
        "25_corruption_summary.csv",
        rows,
        ["config_id", "corruption", "accuracy_mean_pct", "macro_f1_mean_pct", "clean_macro_f1_pct", "drop_vs_clean_pct"],
    )


def results_evidence() -> None:
    rows = [
        {"item": "BN", "status": "tot_hon", "effect_pct": 3.724, "ci": "[+1.767; +5.005]", "evidence": "factorial_effects.json; main_comparisons.csv", "note": "p Holm 0.000"},
        {"item": "Dropout 0,3", "status": "tot_hon", "effect_pct": 5.963, "ci": "[+2.873; +7.147]", "evidence": "factorial_effects.json", "note": "p Holm 0.000"},
        {"item": "Weight decay 1e-4", "status": "khong_du_bang_chung", "effect_pct": 0.058, "ci": "[-0.506; +0.551]", "evidence": "factorial_effects.json", "note": "CI chứa 0"},
        {"item": "Augmentation all", "status": "khong_du_bang_chung", "effect_pct": 0.529, "ci": "[-1.204; +2.303]", "evidence": "factorial_effects.json", "note": "F15-F07 = -2.555; không được nói aug giúp"},
        {"item": "GAP 3 khối (D1)", "status": "kem_hon", "effect_pct": -34.758, "ci": "[-36.181; -22.916]", "evidence": "tables/architecture.md", "note": "underfit nặng"},
        {"item": "Depth 4 vs 3 khối", "status": "chua_ket_luan_sau_holm", "effect_pct": 3.212, "ci": "[+0.036; +4.440]", "evidence": "main_comparisons.csv", "note": "CI không chứa 0 nhưng p Holm 0.373"},
        {"item": "Width lớn (W2)", "status": "khong_du_bang_chung", "effect_pct": 1.268, "ci": "[-0.585; +2.499]", "evidence": "tables/architecture.md", "note": "CI chứa 0, params 3,37M"},
        {"item": "LR 1e-4 (L1)", "status": "kem_hon", "effect_pct": -7.139, "ci": "[-9.372; -3.353]", "evidence": "tables/hyperparam.md", "note": "không hội tụ trong 40 epoch"},
        {"item": "Class weight inv_sqrt", "status": "khong_du_bang_chung", "effect_pct": -1.004, "ci": "[-2.195; +0.261]", "evidence": "tables/hyperparam.md", "note": "CI chứa 0"},
        {"item": "Input 96 (I2)", "status": "kem_hon", "effect_pct": -3.448, "ci": "[-5.463; -0.984]", "evidence": "tables/hyperparam.md", "note": "params 2,2x"},
        {"item": "Aug affine (G1)", "status": "kem_hon", "effect_pct": -4.250, "ci": "[-6.972; -0.084]", "evidence": "tables/augmentation.md", "note": "thành phần hại"},
        {"item": "Aug color (G2)", "status": "khong_du_bang_chung", "effect_pct": 0.917, "ci": "[-0.141; +1.842]", "evidence": "tables/augmentation.md", "note": "CI chứa 0; corruption +0,46 dưới ngưỡng +0,5"},
        {"item": "Aug perspective (G3)", "status": "khong_du_bang_chung", "effect_pct": -1.237, "ci": "[-2.706; +0.530]", "evidence": "tables/augmentation.md", "note": "CI chứa 0"},
        {"item": "X2 vs X3 (OOF)", "status": "tot_hon", "effect_pct": 1.981, "ci": "[+1.053; +2.961]", "evidence": "oof_comparisons.csv", "note": "chọn X2 làm winner"},
        {"item": "X1 vs X2 (OOF)", "status": "chua_ket_luan", "effect_pct": -0.743, "ci": "[-1.590; +0.274]", "evidence": "oof_comparisons.csv", "note": "tie-break ít params: 1,45M < 2,20M"},
        {"item": "Lockbox X2", "status": "xac_nhan", "effect_pct": "", "ci": "", "evidence": "lockbox_result.json", "note": "accuracy 98,876; macro-F1 97,598"},
        {"item": "Occlusion robustness", "status": "kem_hon", "effect_pct": "", "ci": "", "evidence": "corruption.csv", "note": "accuracy trung bình 17,6%"},
        {"item": "Calibration lockbox", "status": "tot", "effect_pct": "", "ci": "", "evidence": "lockbox_result.json", "note": "ECE 0,0053; Brier 0,0179"},
        {"item": "Grad-CAM trong ROI", "status": "tot", "effect_pct": "", "ci": "", "evidence": "16_gradcam_summary.json", "note": "0,853 khi đúng vs 0,543 nếu phân bố đều"},
        {"item": "Lỗi tập trung theo track", "status": "canh_bao", "effect_pct": "", "ci": "", "evidence": "17_manual_error_review_summary.json", "note": "144 lỗi nhưng chỉ 23 track; không đếm ảnh lỗi như ca độc lập"},
    ]
    write_csv(
        "24_results_evidence.csv",
        rows,
        ["item", "status", "effect_pct", "ci", "evidence", "note"],
    )


def evaluation_time() -> None:
    import time

    import torch

    from src.data import _array_to_tensor
    from src.data_v2 import get_indices
    from src.models import build_model

    checkpoint = torch.load(ROOT / "outputs/runs/v2_final_s42/best.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    mean, std = cfg["_resolved"]["mean"], cfg["_resolved"]["std"]
    x = np.load(ROOT / "data/cache/X_train.npy", mmap_mode="r")
    indices = np.asarray(get_indices("screening")["val"], dtype=np.int64)
    rows = []
    for device_name in ["cpu", "cuda"]:
        if device_name == "cuda" and not torch.cuda.is_available():
            continue
        model, _ = build_model(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        device = torch.device(device_name)
        model.to(device).eval()
        # warmup
        with torch.no_grad():
            batch = torch.stack([_array_to_tensor(np.asarray(x[indices[i]]), mean, std) for i in range(32)]).to(device)
            for _ in range(3):
                _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for offset in range(0, len(indices), 256):
                chunk = indices[offset : offset + 256]
                tensors = torch.stack([_array_to_tensor(np.asarray(x[i]), mean, std) for i in chunk]).to(device)
                _ = model(tensors)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        model_only_s = None
        if device_name == "cuda":
            model_only_s = 6180 * (0.809936 / 64) / 1000.0
        else:
            model_only_s = 6180 * (40.16325 / 64) / 1000.0
        rows.append(
            {
                "device": device_name,
                "n_images": int(len(indices)),
                "batch_size": 256,
                "seconds": seconds,
                "images_per_s": len(indices) / seconds,
                "model_only_estimate_s": model_only_s,
                "preprocess_estimate_ms_per_img": (seconds - model_only_s) / len(indices) * 1000,
                "scope": "eval screening val: normalize uint8 -> forward; không gồm đọc file/ppm",
            }
        )
        if device.type == "cuda":
            del model
            torch.cuda.empty_cache()
    write_json(
        "26_evaluation_time.json",
        {
            "split": "screening val",
            "n": int(len(indices)),
            "rows": rows,
            "note": "Đo ngày 04/11/2026 trên máy train; dùng để ước lượng thời gian đánh giá, không phải thời gian UI.",
        },
    )


def glossary() -> None:
    rows = [
        {"term": "Accuracy", "definition": "Tỷ lệ ảnh dự đoán đúng trên tổng số ảnh.", "formula_or_analogy": "số đúng / tổng; dễ bị lớp nhiều ảnh chi phối"},
        {"term": "Precision", "definition": "Trong các ảnh được dự đoán là lớp X, bao nhiêu ảnh thực sự là X.", "formula_or_analogy": "TP / (TP + FP)"},
        {"term": "Recall", "definition": "Trong các ảnh thực sự là lớp X, mô hình nhận ra bao nhiêu.", "formula_or_analogy": "TP / (TP + FN)"},
        {"term": "F1-score", "definition": "Trung bình điều hòa của precision và recall.", "formula_or_analogy": "2PR / (P + R)"},
        {"term": "Macro-F1/43", "definition": "Trung bình F1 của 43 lớp, mỗi lớp trọng số 1/43.", "formula_or_analogy": "lớp hiếm quan trọng ngang lớp nhiều ảnh"},
        {"term": "Weighted-F1", "definition": "F1 trung bình có trọng số theo số ảnh từng lớp.", "formula_or_analogy": "giống accuracy ở chỗ lớp nhiều ảnh chi phối hơn"},
        {"term": "Top-3 accuracy", "definition": "Tỷ lệ ảnh mà nhãn đúng nằm trong 3 dự đoán cao nhất.", "formula_or_analogy": "dùng khi cần gợi ý thay vì chốt 1 nhãn"},
        {"term": "Track", "definition": "Một đoạn video ngắn (~30 frame) của cùng một biển.", "formula_or_analogy": "các frame gần như trùng nhau; không được chia khác tập"},
        {"term": "Sign-group", "definition": "Nhóm các track được gộp vì ảnh gần trùng (MAD < 5).", "formula_or_analogy": "đơn vị chia split để tránh rò rỉ"},
        {"term": "OOF (out-of-fold)", "definition": "Mỗi ảnh dev được dự đoán bởi mô hình không train trên fold chứa ảnh đó.", "formula_or_analogy": "giống thi thử mà đề không có trong bài ôn"},
        {"term": "Lockbox", "definition": "Tập giữ riêng, chỉ mở một lần sau khi chốt mô hình.", "formula_or_analogy": "đề thi cuối kỳ chưa từng xem"},
        {"term": "Official test", "definition": "Tập test công bố của GTSRB, không có track ID.", "formula_or_analogy": "benchmark phụ, không track-disjoint"},
        {"term": "CI 95%", "definition": "Khoảng bất định của một ước lượng.", "formula_or_analogy": "nếu lặp lại thí nghiệm nhiều lần, ~95% khoảng chứa giá trị thật"},
        {"term": "Holm–Bonferroni", "definition": "Hiệu chỉnh p-value khi kiểm định nhiều giả thuyết.", "formula_or_analogy": "giảm xác suất 'may mắn' khi so nhiều cặp"},
        {"term": "ECE", "definition": "Sai lệch trung bình giữa confidence và accuracy theo bin.", "formula_or_analogy": "0 = calibrated hoàn hảo"},
        {"term": "Brier score", "definition": "Sai số bình phương của xác suất dự đoán so với one-hot.", "formula_or_analogy": "càng nhỏ càng tốt"},
        {"term": "Grad-CAM", "definition": "Bản đồ nhiệt cho biết vùng ảnh ảnh hưởng tới dự đoán.", "formula_or_analogy": "mô hình 'nhìn' vào đâu"},
        {"term": "Corruption", "definition": "Biến đổi ảnh để kiểm tra độ bền: blur, noise, brightness, contrast, JPEG, occlusion.", "formula_or_analogy": "điều kiện xấu ngoài đời"},
        {"term": "Occlusion", "definition": "Che một phần trung tâm ảnh.", "formula_or_analogy": "biển bị vật cản che"},
        {"term": "Throughput", "definition": "Số ảnh xử lý được mỗi giây.", "formula_or_analogy": "batch / latency"},
    ]
    write_csv("28_glossary.csv", rows, ["term", "definition", "formula_or_analogy"])


def data_inventory() -> None:
    rows = [
        (1, "Mục tiêu và phạm vi", "Đã có", "Đã có", "BAO_CAO_v2.md ch.1", ""),
        (2, "Nguồn và quy mô dữ liệu", "Đã có", "Đã có", "data_report.json; 01_class_catalog.csv", ""),
        (3, "Phân bố dữ liệu", "Cần hoàn thiện", "Đã bổ sung", "01_class_catalog.csv; 01_class_distribution.png", ""),
        (4, "Chất lượng dữ liệu", "Cần hoàn thiện", "Đã bổ sung", "03_data_quality_per_class.csv; 04_data_quality_summary.json; 16_data_quality.png", ""),
        (5, "Quy trình chia tập", "Đã có", "Đã có", "02_split_summary.csv; 02_split_diagram.png", ""),
        (6, "Audit rò rỉ dữ liệu", "Đã có", "Đã có", "audit_report.json; leak_delta.json", ""),
        (7, "Kiến trúc mô hình", "Đã có", "Đã có", "23_architecture_spec.csv; 03_architecture_d2.png", ""),
        (8, "Cấu hình huấn luyện", "Đã có", "Đã có", "configs/v2/*.yaml", ""),
        (9, "Cấu hình phần cứng", "Cần bổ sung", "Đã bổ sung", "14_hardware_software.json", ""),
        (10, "Thời gian toàn bộ thực nghiệm", "Cần bổ sung", "Đã bổ sung", "19_training_cost_by_group.csv; training_cost_totals.json", ""),
        (11, "Baseline", "Đã có", "Đã có", "05_model_comparison.csv; baseline.json", ""),
        (12, "Factorial 2^4", "Đã có", "Đã có", "05_model_comparison.csv; factorial_effects.json", ""),
        (13, "So sánh kiến trúc", "Đã có", "Đã có", "05_model_comparison.csv; 23_architecture_spec.csv", ""),
        (14, "Hyperparameter", "Đã có", "Đã có", "05_model_comparison.csv", ""),
        (15, "Augmentation", "Đã có", "Đã có", "05_model_comparison.csv; 25_corruption_summary.csv", ""),
        (16, "OOF", "Đã có", "Đã bổ sung bảng", "11_oof_by_fold_seed.csv; 06_eval_sets_summary.csv", ""),
        (17, "Lockbox", "Đã có", "Đã bổ sung bất định", "21_lockbox_by_seed.csv; 06b_lockbox_uncertainty.csv", ""),
        (18, "Official Test", "Đã có", "Đã bổ sung per-class", "22_test_per_class_raw_clean.csv; 06_eval_sets_summary.csv", ""),
        (19, "Precision/Recall từng lớp", "Cần bổ sung", "Đã bổ sung", "07_per_class_lockbox.csv; 08_per_class_oof.csv; 22_test_per_class_raw_clean.csv", ""),
        (20, "F1 từng lớp", "Cần hoàn thiện", "Đã bổ sung", "07_per_class_lockbox.csv; 07_per_class_f1.png", ""),
        (21, "Confusion Matrix", "Cần hoàn thiện", "Đã bổ sung", "09_top_confusions_lockbox.csv; 10_confusion_lockbox_*.csv; 06_confusion_matrix.png", ""),
        (22, "Phân loại lỗi", "Cần hoàn thiện", "Một phần", "17_manual_error_review.csv; 17b_manual_error_review_diverse.csv; contact_sheets/", "verdict thủ công còn trống"),
        (23, "Robustness", "Đã có", "Đã có", "25_corruption_summary.csv; 10_robustness.png", ""),
        (24, "Grad-CAM", "Cần hoàn thiện", "Đã bổ sung", "16_gradcam_roi_mass.csv; 16_gradcam_summary.json; 11_gradcam.png", ""),
        (25, "Calibration", "Cần hoàn thiện", "Đã bổ sung", "12_calibration_confidence.csv; 12_calibration.png", ""),
        (26, "Latency", "Cần hoàn thiện", "Đã bổ sung", "13_inference_benchmark.csv; 14_inference_perf.png", ""),
        (27, "Throughput", "Cần bổ sung", "Đã bổ sung", "13_inference_benchmark.csv", ""),
        (28, "Kích thước mô hình", "Cần bổ sung", "Đã bổ sung", "13_inference_benchmark.json", ""),
        (29, "Bộ nhớ khi suy luận", "Cần bổ sung", "Đã bổ sung", "13_inference_benchmark.csv", ""),
        (30, "Triển khai thực tế", "Cần bổ sung", "Một phần", "15_deployment_status.csv; app.py; e2e_ui.json", "demo đã dùng D2/v2, e2e 9/9; còn API/batch/lịch sử/camera"),
        (31, "Tái lập", "Đã có", "Đã có", "reproducibility.json; tests/", ""),
        (32, "Nghiệm thu", "Đã có", "Đã có", "ACCEPTANCE.json; acceptance.md", ""),
        (33, "Hạn chế", "Đã có", "Đã có", "24_results_evidence.csv; 15_deployment_status.csv", ""),
        (34, "Kế hoạch tương lai", "Đã có", "Đã có", "BAO_CAO_v2.md ch.6.4", ""),
    ]
    write_csv(
        "29_data_inventory.csv",
        [
            {
                "item_no": no,
                "item": item,
                "status_before": before,
                "status_after": after,
                "artifact": artifact,
                "note": note,
            }
            for no, item, before, after, artifact, note in rows
        ],
        ["item_no", "item", "status_before", "status_after", "artifact", "note"],
    )


def dataset_summary() -> None:
    import numpy as np

    y_train = np.load(ROOT / "data/cache/y_train.npy")
    y_test = np.load(ROOT / "data/cache/y_test.npy")
    pool = np.bincount(y_train, minlength=43)
    test = np.bincount(y_test, minlength=43)
    rows = [
        {"metric": "Tổng ảnh train ban đầu", "value": 39209, "note": "GTSRB Final_Training_Images"},
        {"metric": "Ảnh train trùng MD5 với test", "value": 8, "note": "đều lớp 14, track 00023"},
        {"metric": "Pool dùng chia split", "value": 39201, "note": "39.209 − 8"},
        {"metric": "Số lớp", "value": 43, "note": "0–42"},
        {"metric": "Ảnh/lớp nhỏ nhất (pool)", "value": int(pool.min()), "note": f"lớp {int(pool.argmin())}"},
        {"metric": "Ảnh/lớp trung vị (pool)", "value": int(np.median(pool)), "note": ""},
        {"metric": "Ảnh/lớp lớn nhất (pool)", "value": int(pool.max()), "note": f"lớp {int(pool.argmax())}"},
        {"metric": "Tỷ lệ mất cân bằng max/min", "value": round(float(pool.max() / pool.min()), 3), "note": "10,714 lần"},
        {"metric": "Số lớp dưới 300 ảnh", "value": int((pool < 300).sum()), "note": "0, 19, 24, 27, 29, 32, 37, 41, 42"},
        {"metric": "Ảnh test chính thức", "value": int(len(y_test)), "note": "12.630"},
        {"metric": "Ảnh test gần trùng train", "value": 751, "note": "MAD < 5, cùng lớp"},
        {"metric": "Ảnh test clean", "value": 11879, "note": "12.630 − 751"},
        {"metric": "Ảnh lockbox", "value": 8302, "note": "277 track / 238 sign-group"},
        {"metric": "Ảnh dev", "value": 30899, "note": "1.030 track / 945 sign-group"},
        {"metric": "Ảnh screening train", "value": 24719, "note": "824 track"},
        {"metric": "Ảnh screening val", "value": 6180, "note": "206 track"},
        {"metric": "Track toàn pool", "value": 1307, "note": "29–30 frame/track"},
        {"metric": "Sign-group toàn pool", "value": 1183, "note": "gộp 39.603 cặp gần trùng"},
        {"metric": "Ảnh trùng MD5 trong pool", "value": 0, "note": "SHA-256/MD5"},
    ]
    write_csv("02b_dataset_summary.csv", rows, ["metric", "value", "note"])


def class_extremes() -> None:
    from scipy import stats

    per = pd.read_csv(TABLES / "07_per_class_lockbox.csv")
    names = {int(r["class_id"]): r["name_vi"] for r in per.to_dict("records")}
    per = per.sort_values("f1_mean")
    bottom = per.head(5)
    top = per.tail(5).sort_values("f1_mean", ascending=False)
    # Lỗi ước lượng = support * (1 - recall)
    per["errors_est"] = per["support"] * (1 - per["recall_mean"])
    most_errors = per.sort_values("errors_est", ascending=False).head(5)
    least_support = per.sort_values("support").head(5)
    rho, p_rho = stats.spearmanr(per["support"], per["f1_mean"])
    r, p_r = stats.pearsonr(per["support"], per["f1_mean"])
    payload = {
        "top5_f1": [
            {"class_id": int(r["class_id"]), "name_vi": r["name_vi"], "f1": float(r["f1_mean"]), "support": int(r["support"])}
            for r in top.to_dict("records")
        ],
        "bottom5_f1": [
            {"class_id": int(r["class_id"]), "name_vi": r["name_vi"], "f1": float(r["f1_mean"]), "support": int(r["support"])}
            for r in bottom.to_dict("records")
        ],
        "most_errors_est": [
            {"class_id": int(r["class_id"]), "name_vi": r["name_vi"], "errors_est": float(r["errors_est"]), "support": int(r["support"])}
            for r in most_errors.to_dict("records")
        ],
        "least_support": [
            {"class_id": int(r["class_id"]), "name_vi": r["name_vi"], "support": int(r["support"])}
            for r in least_support.to_dict("records")
        ],
        "support_vs_f1": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
            "n_classes": 43,
            "note": "Tương quan yếu/không rõ; lớp hiếm không phải lúc nào cũng kém.",
        },
        "source": "07_per_class_lockbox.csv",
    }
    write_json("07b_class_extremes.json", payload)


def app_summary() -> None:
    lock = json.loads((V2 / "lockbox_result.json").read_text(encoding="utf-8"))
    report = lock["reports"][0]
    summary = lock["summary"]
    payload = {
        "model": "D2 (X2) — 4 khối 32/64/128/256, flatten, BN, dropout 0,3",
        "checkpoint": "outputs/runs/v2_final_s42/best.pt",
        "checkpoint_sha256": report.get("checkpoint_sha256", ""),
        "n_samples": int(lock["n_images"]),
        "accuracy": float(summary["accuracy"]["mean"]),
        "accuracy_sd": float(summary["accuracy"]["sd"]),
        "macro_f1_43": float(summary["macro_f1_43"]["mean"]),
        "macro_f1_sd": float(summary["macro_f1_43"]["sd"]),
        "top3_accuracy": float(summary["top3_accuracy"]["mean"]),
        "ece_15bin": float(summary["ece_15bin"]["mean"]),
        "brier": float(summary["brier"]["mean"]),
        "seed_accuracy": summary["accuracy"]["values"],
        "seed_macro_f1": summary["macro_f1_43"]["values"],
        "eval_set": "lockbox 8.302 ảnh (277 track), mở một lần sau freeze",
        "note": "Dùng cho sidebar demo v2; không dùng để chọn mô hình/ngưỡng.",
    }
    write_json("app_summary_v2.json", payload)


def main() -> int:
    hardware_software()
    deployment_status()
    gradcam_summary()
    lockbox_uncertainty()
    oof_accuracy_check()
    corrections()
    corruption_summary()
    results_evidence()
    evaluation_time()
    glossary()
    data_inventory()
    dataset_summary()
    class_extremes()
    app_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
