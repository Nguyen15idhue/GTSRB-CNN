# REPORT DATA INVENTORY — BÁO CÁO LÀM LẠI (PACK V3)

| Thông tin | Giá trị |
|---|---|
| Mục đích | Kiểm kê toàn bộ dữ liệu cần cho báo cáo mới: đã có gì, ở đâu, còn thiếu gì, lệnh lấy |
| Ngày lập | 04/11/2026 |
| Phạm vi | Báo cáo mới thay thế `BAO_CAO_v2.md`; không chạy lại 111 run, không mở lockbox |
| Nguồn sự thật | `outputs/results/v2/`, `outputs/splits/v2/`, `outputs/runs/`, `outputs/results/v3/` |
| Trạng thái | **Sẵn sàng 32/34 mục dữ liệu; 2 mục một phần (kiểm tra lỗi thủ công, tính năng triển khai nâng cao); 2 hạng mục ngoài cần thiết bị/dữ liệu thật. Báo cáo v3 đã tạo: docs/bao_cao/BAO_CAO_v3.md + .docx** |

> Quy ước: **Đã có** = artifact đã tồn tại và dùng được; **Đã bổ sung** = pack v3
> vừa sinh; **Một phần** = có tài nguyên nhưng cần người/thiết bị; **Chưa thể** =
> không thể tạo từ repo hiện tại.

---

## 1. Tóm tắt kiểm kê

| Nhóm | Số mục | Trạng thái |
|---|---:|---|
| Kết quả thực nghiệm đã có | 19 | Đã có, chỉ cần trình bày lại |
| Trình bày lại / xuất bảng | 8 | Đã bổ sung bảng/hình |
| Phép đo vận hành mới | 7 | Đã đo mới (trừ utilization lịch sử) |
| **Tổng 34 mục** | **34** | **32 đủ tài nguyên; 2 mục một phần (lỗi thủ công, triển khai nâng cao)** |

Tài nguyên mới:

- **43 file bảng/số liệu** trong `outputs/results/v3/tables/`.
- **20 hình** trong `outputs/figures/v3/` (gồm 2 contact sheet).
- **11 file dữ liệu numpy** trong `outputs/results/v3/data/`.
- **4 script** sinh lại.
- **1 manifest** sha256.

---

## 2. Đối chiếu 34 mục dữ liệu

> Đường dẫn `tables/...`, `data/...`, `figures/...` tính từ pack root
> `outputs/results/v3/` và `outputs/figures/v3/`. Đường dẫn bắt đầu bằng
> `outputs/...` là đường dẫn đầy đủ từ repo.

| # | Mục theo tài liệu đánh giá | Trước | Sau | File / nguồn | Ghi chú |
|---:|---|---|---|---|---|
| 1 | Mục tiêu và phạm vi | Đã có | Đã có | `BAO_CAO_v2.md` ch.1 | Viết lại cho người không chuyên |
| 2 | Nguồn và quy mô dữ liệu | Đã có | Đã có | `outputs/results/data_report.json`; `tables/01_class_catalog.csv`, `tables/02b_dataset_summary.csv` | 39.209 → 39.201 sau loại 8 ảnh trùng |
| 3 | Phân bố dữ liệu | Cần hoàn thiện | **Đã bổ sung** | `tables/01_class_catalog.csv`, `02b_dataset_summary.csv`; hình `01` | 43 dòng, có % và support lockbox |
| 4 | Chất lượng dữ liệu | Cần hoàn thiện | **Đã bổ sung** | `tables/03_data_quality_per_class.csv`, `04_data_quality_summary.json`; hình `16`, `16b` | Blur/brightness/contrast/ROI, 0 ảnh hỏng |
| 5 | Quy trình chia tập | Đã có | Đã có | `tables/02_split_summary.csv`; hình `02` | Sơ đồ mới 3 tầng |
| 6 | Audit rò rỉ dữ liệu | Đã có | Đã có | `outputs/results/v2/audit/audit_report.json`, `leak_delta.json` | 15/15 PASS; 95,22% frame ±1 ở random |
| 7 | Kiến trúc mô hình | Đã có | Đã có | `tables/23_architecture_spec.csv`; hình `03` | Params từng lớp, 8 cấu hình |
| 8 | Cấu hình huấn luyện | Đã có | Đã có | `configs/v2/*.yaml` | AdamW, LR 1e-3, batch 128, 40 epoch |
| 9 | Cấu hình phần cứng | Cần bổ sung | **Đã bổ sung** | `tables/14_hardware_software.json` | CPU/GPU/VRAM/RAM/SSD/OS |
| 10 | Thời gian toàn bộ thực nghiệm | Cần bổ sung | **Đã bổ sung** | `tables/19_training_cost_by_group.csv`, `training_cost_totals.json` | 3,99 h epoch-loop + 8,2 phút baseline |
| 11 | Baseline | Đã có | Đã có | `tables/05_model_comparison.csv`; `baseline.json` | B0 majority, B1 HOG+SVC |
| 12 | Factorial 2⁴ | Đã có | Đã có | `tables/05_model_comparison.csv`; `factorial_effects.json` | 16 cấu hình × 3 seed |
| 13 | So sánh kiến trúc | Đã có | Đã có | `tables/05`, `23`; `tables/architecture.md` | D1/D2/D3/W1/W2 |
| 14 | Hyperparameter | Đã có | Đã có | `tables/05`; `tables/hyperparam.md` | LR, class weight, input size |
| 15 | Augmentation | Đã có | Đã có | `tables/05`, `25`; `tables/augmentation.md` | Có mức giảm so với clean |
| 16 | OOF | Đã có | **Đã bổ sung bảng** | `tables/11_oof_by_fold_seed.csv`, `06_eval_sets_summary.csv` | 3 fold × 3 seed × 3 config |
| 17 | Lockbox | Đã có | **Đã bổ sung bất định** | `tables/21_lockbox_by_seed.csv`, `06b_lockbox_uncertainty.csv` | mean ± sd + range; binomial CI accuracy |
| 18 | Official Test | Đã có | **Đã bổ sung per-class** | `tables/22_test_per_class_raw_clean.csv`, `06` | raw 12.630 / clean 11.879 |
| 19 | Precision/Recall từng lớp | Cần bổ sung | **Đã bổ sung** | `tables/07`, `08`, `22` | 43 dòng × 3 tập |
| 20 | F1 từng lớp | Cần hoàn thiện | **Đã bổ sung** | `tables/07`, `08`; hình `07` | Xếp lớp yếu lên đầu |
| 21 | Confusion Matrix | Cần hoàn thiện | **Đã bổ sung** | `tables/09`, `10_confusion_lockbox_*.csv`; hình `06` | 43×43 + top 10 cặp |
| 22 | Phân loại lỗi | Cần hoàn thiện | **Một phần** | `tables/17_manual_error_review.csv`, `17b_manual_error_review_diverse.csv`; 2 contact sheet | Máy đã phân nhóm heuristic; **verdict thủ công còn trống** |
| 23 | Robustness | Đã có | Đã có | `tables/25_corruption_summary.csv`; hình `10` | 6 corruption × 3 mức; occlusion 17,6% |
| 24 | Grad-CAM | Cần hoàn thiện | **Đã bổ sung** | `tables/16_gradcam_roi_mass.csv`, `16_gradcam_summary.json`; hình `11` | Mass trong ROI 0,853 đúng / 0,733 sai |
| 25 | Calibration | Cần hoàn thiện | **Đã bổ sung** | `tables/12_calibration_confidence.csv`; hình `12` | ECE, Brier, confidence đúng/sai |
| 26 | Latency | Cần hoàn thiện | **Đã bổ sung** | `tables/13_inference_benchmark.csv`; hình `14` | Batch 1/16/32/64, CPU/GPU, FP32 |
| 27 | Throughput | Cần bổ sung | **Đã bổ sung** | `tables/13_inference_benchmark.csv` | 79k ảnh/s GPU batch 64 |
| 28 | Kích thước mô hình | Cần bổ sung | **Đã bổ sung** | `tables/13_inference_benchmark.json` | 5,55 MB, 1.448.779 params |
| 29 | Bộ nhớ khi suy luận | Cần bổ sung | **Đã bổ sung** | `tables/13_inference_benchmark.csv` | VRAM 81,7 MB; RAM process ~1.060 MB |
| 30 | Triển khai thực tế | Cần bổ sung | **Một phần** | `tables/15_deployment_status.csv`; `app.py`; `e2e_ui.json` | Demo đã dùng D2/v2, e2e 9/9; còn API/batch/lịch sử/camera |
| 31 | Tái lập | Đã có | Đã có | `outputs/results/v2/reproducibility.json`; `tests/` | Cùng seed lệch 0,0; 22 test pass |
| 32 | Nghiệm thu | Đã có | Đã có | `outputs/results/v2/ACCEPTANCE.json`, `acceptance.md` | 12/12 |
| 33 | Hạn chế | Đã có | Đã có | `tables/24_results_evidence.csv`, `15_deployment_status.csv` | Phân loại tốt/không đủ/kém |
| 34 | Kế hoạch tương lai | Đã có | Đã có | `BAO_CAO_v2.md` ch.6.4 | Chia 3 giai đoạn |

---

## 3. Danh mục 15 hình theo đề xuất

> Các file trong bảng này nằm trong `outputs/figures/v3/` trừ khi ghi đường dẫn đầy đủ.

| # | Hình đề xuất | File | Nguồn | Trạng thái |
|---:|---|---|---|---|
| 1 | Phân bố dữ liệu 43 lớp | `outputs/figures/v3/01_class_distribution.png` | `tables/01` | Mới |
| 2 | Sơ đồ chia tập | `02_split_diagram.png` | `tables/02` | Mới |
| 3 | Sơ đồ kiến trúc D2 | `03_architecture_d2.png` | `tables/23` | Mới |
| 4 | Learning curve | `04_learning_curves.png` | `outputs/runs/v2_d2_s42/metrics.csv`, `v2_f07_s42` | Mới (train+val) |
| 5 | So sánh mô hình | `05_model_comparison.png` | `tables/05` | Mới |
| 6 | Confusion matrix | `06_confusion_matrix.png` | `tables/10`, `09` | Mới |
| 7 | F1 từng lớp | `07_per_class_f1.png` | `tables/07` | Mới |
| 8 | Ảnh hưởng các yếu tố | `08_factor_effects.png` | `factorial_effects.json` | Mới |
| 9 | OOF theo fold × seed | `09_oof_fold_seed.png` | `tables/11`, `variance.json` | Mới |
| 10 | Robustness | `10_robustness.png` | `corruption.csv`, `tables/25` | Mới |
| 11 | Grad-CAM | `11_gradcam.png` | `tables/16`, `data/gradcam_cams.npz` | Mới (D2) |
| 12 | Calibration | `12_calibration.png` | `data/calibration_screening.npz`, `selective_risk_oof.npz` | Mới |
| 13 | Precision–Recall | `13_precision_recall.png` | `data/screening_val_probs_d2_s42.npz` | Mới |
| 14 | Hiệu năng suy luận | `14_inference_perf.png` | `tables/13` | Mới |
| 15 | Ví dụ dự đoán thực tế | `15_prediction_examples.png` | `screening_val_probs_d2_s42.npz`, ảnh gốc crop ROI | Mới |
| + | Chất lượng dữ liệu | `16_data_quality.png` | `tables/04` | Hỗ trợ |
| + | Ảnh cực trị chất lượng | `16b_data_quality_examples.png` | `tables/04` | Hỗ trợ |
| + | Ảnh mẫu 43 lớp | `17_class_samples.png` | ảnh gốc crop ROI + `train_meta.csv` | Hỗ trợ |
| + | Phiếu rà soát lỗi | `outputs/figures/v3/contact_sheets/error_sheet_01..02.png` | `tables/17b` | Hỗ trợ |

---

## 4. Danh mục phụ lục A–V

| Phụ lục | Nội dung | File | Trạng thái |
|---|---|---|---|
| A | Danh sách 43 lớp | `tables/01_class_catalog.csv` | Đã có |
| B | Mô tả dữ liệu | `data_report.json`, `tables/04` | Đã có |
| C | Phân phối từng lớp | `tables/01`, `03` | Đã có |
| D | Ảnh bị loại và lý do | `outputs/results/excluded_train_duplicates.csv` | Đã có |
| E | Chi tiết Train/Val/Dev/OOF/Lockbox | `tables/02`, `11` | Đã có |
| F | Audit rò rỉ | `audit_report.json`, `leak_delta.json` | Đã có |
| G | 16 cấu hình factorial | `tables/05`, `tables/factorial.md` | Đã có |
| H | Toàn bộ 111 run | `tables/20_experiment_index.csv` | Đã có |
| I | Cấu hình CNN | `tables/23_architecture_spec.csv` | Đã có |
| J | P/R/F1 43 lớp | `tables/07`, `08`, `22` | Đã có |
| K | Confusion matrix đầy đủ | `tables/10_confusion_lockbox_*.csv` | Đã có |
| L | Danh sách ảnh dự đoán sai | `tables/17_manual_error_review.csv` | Đã có (verdict trống) |
| M | Corruption robustness | `tables/25_corruption_summary.csv` | Đã có |
| N | OOF theo fold/seed | `tables/11_oof_by_fold_seed.csv` | Đã có |
| O | Lockbox theo seed | `tables/21_lockbox_by_seed.csv` | Đã có |
| P | Official Test raw/clean | `tables/22_test_per_class_raw_clean.csv` | Đã có |
| Q | Learning curves | `outputs/figures/v3/04_learning_curves.png` | Đã có |
| R | Grad-CAM | `outputs/figures/v3/11_gradcam.png`, `tables/16` | Đã có |
| S | Calibration | `outputs/figures/v3/12_calibration.png`, `tables/12` | Đã có |
| T | Phần cứng/phần mềm | `tables/14_hardware_software.json` | Đã có |
| U | Nhật ký thực nghiệm | `docs/kehoach/1/KET_QUA_TRIEN_KHAI_V2.md` | Đã có |
| V | Mã nguồn và tái lập | `BAO_CAO_v2.md` Phụ lục E; `tests/` | Đã có |

---

## 5. Nguồn sự thật cho các số headline

| Số headline | Giá trị | File |
|---|---|---|
| Pool chia split | 39.201 | `tables/02_split_summary.csv` |
| Lockbox | 8.302 ảnh / 277 track | `tables/02` |
| OOF macro-F1 (X2) | 97,0400 ± 0,1869 | `tables/06_eval_sets_summary.csv` |
| OOF accuracy (X2) | 97,9244 | `tables/06c_oof_accuracy_check.json` |
| Lockbox accuracy | 98,8758 ± 0,1010 | `tables/06b_lockbox_uncertainty.csv` |
| Lockbox macro-F1 | 97,5981 ± 0,6246 | `tables/06b` |
| Official raw | 98,6989 / 98,0398 | `tables/06` |
| Official clean | 98,8692 / 98,2499 | `tables/06` |
| D2 params | 1.448.779 | `tables/13_inference_benchmark.json` |
| Checkpoint | 5,55 MB | `tables/13` |
| Latency GPU batch 1 | 0,86 ms model-only; 1,24 ms end-to-end | `tables/13` |
| Tổng thời gian | 3,99 h + 8,2 phút | `tables/training_cost_totals.json` |
| Số run | 111 CNN + 1 baseline | `tables/20_experiment_index.csv` |
| Số lỗi D2 | 144 ảnh / 23 track | `tables/17_manual_error_review_summary.json` |

---

## 6. Lệnh sinh lại toàn bộ

### 6.1. Lệnh theo nhóm dữ liệu

| Nhóm dữ liệu | Lệnh | Thời gian |
|---|---|---|
| Bảng 43 lớp, split, model comparison, per-class, confusion, OOF, test | `python scripts/report_v3_data.py --only tables` | ~1 phút |
| Xác suất screening val, calibration, threshold v2 | `python scripts/report_v3_data.py --only probs` | ~15 giây |
| Chất lượng dữ liệu (blur/brightness/contrast) | `python scripts/report_v3_data.py --only quality` | ~20 giây |
| Benchmark CPU/GPU, VRAM, RAM | `python scripts/report_v3_data.py --only benchmark` | ~1 phút |
| Grad-CAM + mass trong ROI | `python scripts/report_v3_data.py --only gradcam` | ~1 phút |
| Phiếu rà soát lỗi | `python scripts/report_v3_data.py --only review` | ~10 giây |
| 20 hình | `python scripts/report_v3_figures.py --only all` | ~1 phút |
| Hardware/software, deployment, đính chính, glossary, inventory | `python scripts/report_v3_meta.py` | ~15 giây |
| Manifest sha256 | `python scripts/report_v3_manifest.py` | ~10 giây |

### 6.2. Lệnh đầy đủ

```powershell
# 1. Bảng, xác suất, chất lượng dữ liệu, benchmark, Grad-CAM, phiếu lỗi
.\.venv\Scripts\python.exe scripts\report_v3_data.py --only all

# 2. 20 hình
.\.venv\Scripts\python.exe scripts\report_v3_figures.py --only all

# 3. Hardware/software, deployment, đính chính, glossary
.\.venv\Scripts\python.exe scripts\report_v3_meta.py

# 4. Manifest sha256
.\.venv\Scripts\python.exe scripts\report_v3_manifest.py
```

Thời gian: ~3–5 phút trên máy train. Không train lại, không mở lockbox.

---

## 7. Những mục không thể chuẩn bị từ repo

| Mục | Vì sao | Ảnh hưởng tới báo cáo | Đề xuất |
|---|---|---|---|
| Kiểm tra nhãn/lỗi thủ công | Cần mắt người | Không được kết luận nguyên nhân lỗi | Dành 30–45 phút điền 41 ảnh |
| Camera thật / video | Không có dữ liệu | Không được tuyên bố triển khai | Ghi hạn chế |
| Máy cấu hình thấp | Chỉ có i9 + RTX 5070 | Không được suy rộng hiệu năng | Đo lại trên laptop nếu cần |
| GPU/CPU utilization 111 run | Không instrument lúc train | Không báo được % utilization | Ghi "không đo" |
| CI track-bootstrap lockbox | One-shot, không có per-image | Chỉ báo mean ± sd + range | Chấp nhận; không mở lại lockbox |

---

## 8. Quy tắc sử dụng pack

1. Mỗi bảng trong báo cáo chỉ dùng **một tập**; ghi rõ tập ở tiêu đề.
2. Không chép số từ tài liệu đánh giá; chỉ dùng file trong mục 5.
3. Không dùng CI OOF cho lockbox; không dùng 98,004% cho OOF.
4. Không nói aug "all" giúp D2; dùng `24_results_evidence.csv`.
5. Không đếm 144 ảnh lỗi như 144 ca độc lập; ghi 23 track.
6. Không gọi latency model-only là thời gian app.
7. Không mô tả demo là "đã triển khai"; dùng `15_deployment_status.csv`.
8. Mọi hình/bảng phải khớp `manifest.json` (sha256) trước khi nộp.
