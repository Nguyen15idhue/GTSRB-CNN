# Report pack v3 — tài nguyên cho báo cáo làm lại

Thư mục này chứa **bảng, số liệu, hình và manifest** phục vụ bản báo cáo làm lại
sau khi báo cáo v2 bị đánh giá chưa đạt. Pack **không chạy lại huấn luyện** và
**không mở lại lockbox**; mọi số liệu lấy từ artifact đã có hoặc đo mới trong
phạm vi suy luận.

Báo cáo hoàn chỉnh: `docs/bao_cao/BAO_CAO_v3.md` và `docs/bao_cao/BAO_CAO_v3.docx`.
Phân tích đánh giá: `docs/bao_cao/PHAN_TICH_DANH_GIA_VA_KE_HOACH_TAI_TAO.md`.
Kiểm kê dữ liệu: `docs/bao_cao/REPORT_DATA_INVENTORY.md`.

## 1. Nguyên tắc

- Không có số liệu minh họa. Mỗi con số đều truy được về file trong `tables/`
  hoặc artifact nguồn trong `outputs/results/v2/`.
- Bốn tập đánh giá được tách rõ: **screening**, **OOF**, **lockbox**,
  **official test**. Không so sánh trực tiếp các con số giữa bốn tập.
- Kết quả "không đủ bằng chứng" được giữ nguyên, không làm tròn thành "tốt hơn".
- Các số cũ sai/lệch được ghi trong `tables/00_corrections.csv`.

## 2. Cấu trúc

```
outputs/results/v3/
├── README.md
├── manifest.json              # sha256 + kích thước + mục đích; data/ ở mục data_local_only
├── tables/                    # 43 bảng/số liệu máy đọc
│   ├── 00_corrections.csv
│   ├── 01_class_catalog.csv
│   ├── ...
│   └── 29_data_inventory.csv
└── data/                      # mảng numpy tái dùng cho hình (KHÔNG commit; sinh lại)
    ├── screening_val_probs_d2_s42.npz
    ├── calibration_screening.npz
    ├── confusion_lockbox_mean.npy
    ├── gradcam_cams.npz
    ├── selective_risk_oof.npz
    └── train_*.npy / test_*.npy

outputs/figures/v3/
├── 01_class_distribution.png
├── ...
├── 17_class_samples.png
└── contact_sheets/error_sheet_01..02.png
```

## 3. Sinh lại

```powershell
.\.venv\Scripts\python.exe scripts\report_v3_data.py --only all
.\.venv\Scripts\python.exe scripts\report_v3_figures.py --only all
.\.venv\Scripts\python.exe scripts\report_v3_meta.py
.\.venv\Scripts\python.exe scripts\report_v3_manifest.py

# 5. Báo cáo Markdown -> DOCX
.\.venv\Scripts\python.exe docs\bao_cao\md_to_docx_v3.py
```

Thời gian chạy lại toàn bộ khoảng 3–5 phút trên máy train (chủ yếu là đọc cache
và đo benchmark/Grad-CAM), không train lại.

## 4. Số liệu chốt (dùng cho báo cáo)

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Pool chia split | 39.201 ảnh, 43 lớp | `tables/02_split_summary.csv` |
| Lockbox | 8.302 ảnh / 277 track | `tables/02_split_summary.csv` |
| OOF macro-F1/43 (X2) | 97,0400 ± 0,1869 | `tables/06_eval_sets_summary.csv` |
| OOF accuracy (X2) | **97,9244** (không phải 98,004) | `tables/06c_oof_accuracy_check.json` |
| Lockbox accuracy | 98,8758 ± 0,1010 | `tables/06b_lockbox_uncertainty.csv` |
| Lockbox macro-F1/43 | 97,5981 ± 0,6246; range 3 seed [96,8770; 97,9706] | `tables/06b_lockbox_uncertainty.csv` |
| Official test raw | 98,6989 / 98,0398 (acc/F1) | `tables/06_eval_sets_summary.csv` |
| Official test clean | 98,8692 / 98,2499 | `tables/06_eval_sets_summary.csv` |
| D2 params | 1.448.779 | `tables/13_inference_benchmark.json` |
| Checkpoint D2 | 5,55 MB (5.815.209 byte) | `tables/13_inference_benchmark.json` |
| GPU model-only batch 1 | 0,86 ms; end-to-end Python 1,24 ms | `tables/13_inference_benchmark.csv` |
| GPU throughput batch 64 | ~79.000 ảnh/s (model-only) | `tables/13_inference_benchmark.csv` |
| Tổng thời gian 111 run | 3,99 giờ epoch-loop + 8,2 phút baseline | `tables/training_cost_totals.json` |

## 5. Cảnh báo khi dùng

1. **Không** dùng 98,004% làm OOF accuracy. Đó là accuracy screening của D2.
2. **Không** dùng CI [95,0511; 97,9422] làm CI lockbox. Đó là CI OOF.
3. **Không** nói augmentation "all" giúp D2 thắng. Main effect +0,529 với CI chứa 0;
   F15 còn kém F07 2,555 điểm trên clean.
4. **Không** so sánh lockbox 8.302 ảnh với official test 12.630 ảnh như cùng một phép thử.
5. **Không** gọi latency model-only là thời gian phản hồi của app.
6. **Không** đếm 144 ảnh lỗi như 144 ca độc lập: chúng chỉ thuộc 23 track.
7. Demo đã dùng D2/v2 (`v2_final_s42`, τ = 0,9590); golden test 20 ảnh vẫn là checkpoint legacy, chưa chạy lại cho v2.

## 6. Việc còn cần con người

- Điền `verdict`, `cause`, `note` trong `tables/17b_manual_error_review_diverse.csv`
  sau khi xem 2 contact sheet (41 ảnh, 23 track). Đây là bước kiểm tra thủ công
  duy nhất chưa làm được bằng máy.
- Demo đã cập nhật; chạy lại e2e bằng `python tests/e2e_app.py` (9/9 pass ngày 04/11/2026).
- Nếu muốn tuyên bố hiệu năng trên máy yếu: đo lại trên laptop cấu hình thấp,
  không suy từ i9-14900KF.
