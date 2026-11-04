# PHÂN TÍCH ĐÁNH GIÁ BÁO CÁO V2 & KẾ HOẠCH TÁI TẠO

| Thông tin | Giá trị |
|---|---|
| Tài liệu được phân tích | `docs/bao_cao/Đánh giá báo cáo v2.docx` |
| Đối tượng bị đánh giá | `docs/bao_cao/BAO_CAO_v2.md` / `.docx` (24 trang) |
| Ngày phân tích | 04/11/2026 |
| Người thực hiện | Agent (chuẩn bị tài nguyên), chờ chủ dự án duyệt |
| Trạng thái | **Đã chuẩn bị tài nguyên và đã tạo báo cáo v3: BAO_CAO_v3.md + BAO_CAO_v3.docx** |
| Nguyên tắc số liệu | Mọi số trong tài liệu này lấy từ artifact thật; số trong tài liệu đánh giá chỉ là **giả định** và đã được kiểm tra lại |

---

## 0. Tóm tắt điều hành

Báo cáo v2 **không thiếu thực nghiệm**. Vấn đề là báo cáo được viết như một
nhật ký kỹ thuật: đúng số nhưng không dẫn người đọc đi từ câu hỏi → thiết kế →
kết quả → độ tin cậy → hạn chế. Người phản biện đã chỉ ra đúng bản chất: có
rất nhiều số liệu nhưng chưa thành **hệ thống thông tin**.

Ba nhóm vấn đề, theo mức ưu tiên:

1. **Trình bày và tường thuật (ưu tiên 1).** Không có tóm tắt điều hành; thuật
   ngữ xuất hiện trước khi được giải thích; bảng dày đặc; ba tập đánh giá bị
   trộn lẫn; không có bảng 43 lớp, không có bảng so sánh mô hình duy nhất.
2. **Dữ liệu trình bày còn thiếu (ưu tiên 2).** Per-class P/R/F1, confusion
   matrix có phân tích, calibration diễn giải, Grad-CAM định lượng, phân loại
   lỗi có kiểm tra thủ công, robustness có mức giảm so với ảnh sạch.
3. **Vận hành và triển khai (ưu tiên 3).** Phần cứng/phần mềm, tổng thời gian,
   latency theo batch/thiết bị, throughput, dung lượng mô hình, RAM/VRAM, trạng
   thái demo, tái lập, nghiệm thu dạng bảng kiểm.

Đã chuẩn bị **43 bảng/số liệu + 20 hình + 4 script + manifest**. Không chạy lại
111 run, không mở lại lockbox. Các số liệu đã được kiểm tra chéo và phát hiện
**8 điểm lệch** so với báo cáo cũ (mục 4), trong đó quan trọng nhất là
**OOF accuracy 98,004% thực chất là accuracy screening của D2; OOF đúng là
97,924%**.

> **Về các số trong tài liệu đánh giá:** người dùng đã nêu rõ các số đó là giả
> định, không phải số thật. Tôi không lấy chúng làm nguồn. Mọi số trong tài liệu
> này được tính lại từ `outputs/results/v2/`, `outputs/splits/v2/` và các lần đo
> mới; số nào trùng artifact thì giữ, số nào lệch thì ghi ở mục 4.

---

## 1. Cách đọc đánh giá của người phản biện

Tài liệu đánh giá dài 16 bảng, chia thành 11 mục. Nếu bỏ lớp trình bày, có thể
quy về **sáu luận điểm**:

| # | Luận điểm | Bản chất | Mức độ |
|---|---|---|---|
| 1 | "Nhiều số liệu nhưng chưa thành hệ thống thông tin" | Thiếu kiến trúc thông tin, không phải thiếu kết quả | Cao |
| 2 | "Người không chuyên không đọc được" | Thiếu glossary, thiếu tóm tắt, thuật ngữ không giải thích | Cao |
| 3 | "Thiếu bảng 43 lớp, per-class, confusion analysis" | Dữ liệu có trong JSON/CSV nhưng chưa xuất bảng người đọc | Cao |
| 4 | "Thiếu phần vận hành: hardware, latency, throughput, model size, RAM/VRAM" | Chưa đo hoặc đo thiếu ngữ cảnh | Trung bình |
| 5 | "Trộn ba tập đánh giá" | Nguy cơ hiểu sai OOF/lockbox/test | Cao |
| 6 | "Không được nói augmentation all giúp D2 thắng" | Diễn giải sai kết quả âm | Cao |

Điểm quan trọng: người phản biện **không yêu cầu chạy lại thực nghiệm**. Họ yêu
cầu **tổ chức lại thông tin** và **bổ sung các phép đo còn thiếu**. Vì vậy chi
phí tái tạo chủ yếu là thời gian viết, không phải GPU.

---

## 2. Sáu vấn đề cốt lõi — phân tích chi tiết

### 2.1. Vấn đề 1: báo cáo là nhật ký, không phải báo cáo kết quả

**Biểu hiện trong báo cáo v2:**

- Mở đầu bằng "Báo cáo này thay thế các kết luận cũ dựa trên `split_random`…"
  — người đọc chưa biết dự án làm gì đã gặp tranh luận kỹ thuật.
- Chương 1 dành cho "Bốn phản biện và cách trả lời", tức là báo cáo tự vệ trước
  khi trình bày kết quả.
- 24 trang nhưng không có một trang nào trả lời trọn vẹn: "Dự án làm gì, kết quả
  bao nhiêu, mô hình nào, vì sao, còn hạn chế gì?".

**Hệ quả:** người đọc phải tự lắp ghép. Giảng viên đọc 5 phút đầu không thấy
bảng kết quả tổng hợp; sinh viên muốn hiểu "macro-F1 là gì" phải đọc đến chương 3.

**Cách xử lý:** đưa **Tóm tắt điều hành 1–2 trang** lên đầu, gồm một bảng 10 dòng
(mục tiêu, dữ liệu, phương pháp, số run, mô hình, kết quả OOF/lockbox/test, điểm
nổi bật, hạn chế, kết luận). Sau đó mới đến phương pháp.

### 2.2. Vấn đề 2: người không chuyên không đọc được

**Biểu hiện:**

- Thuật ngữ `OOF`, `lockbox`, `sign-group`, `track`, `macro-F1/43`, `ECE`,
  `Brier`, `Holm–Bonferroni`, `cluster bootstrap` xuất hiện dày đặc.
- Không có glossary; không có ví dụ số cụ thể cho Accuracy vs Macro-F1.
- Bảng 16 dòng factorial với 11 cột — đúng về kỹ thuật nhưng khó đọc.

**Hệ quả:** người đọc không đánh giá được độ tin cậy của kết quả, chỉ thấy
"0,975981" mà không biết nó tốt hay xấu, khác accuracy thế nào.

**Cách xử lý:**

- Glossary 20 thuật ngữ (`tables/28_glossary.csv`).
- Một ví dụ tính tay trên chính dữ liệu lockbox (`tables/18_metric_worked_example.csv`,
  `metric_worked_example.json`): lớp 19 có 30 ảnh, F1 0,760; lớp 4 có 750 ảnh,
  F1 ~0,999; macro-F1 cho mỗi lớp trọng số 1/43.
- Mỗi hình có một câu "đọc gì" ngay dưới hình.

### 2.3. Vấn đề 3: thiếu bảng 43 lớp và phân tích per-class

**Biểu hiện:** báo cáo v2 chỉ nêu "lớp 19 F1 0,760, lớp 29 F1 0,776" trong một
caption; không có bảng 43 dòng; không có precision/recall; không có support.

**Dữ liệu thật đã có:** `lockbox_result.json` chứa per-class P/R/F1 cho 3 seed;
`oof_predictions.csv` chứa 278.091 dòng dự đoán; `official_test_v2.json` chứa
per-class raw/clean.

**Đã chuẩn bị:**

- `tables/01_class_catalog.csv`: 43 lớp, tên Việt/Anh, hình dạng, màu, số ảnh
  pool/test/lockbox, %.
- `tables/07_per_class_lockbox.csv`: P/R/F1 mean/sd, support, xếp hạng.
- `tables/08_per_class_oof.csv`: P/R/F1 trên OOF.
- `tables/22_test_per_class_raw_clean.csv`: P/R/F1 trên official test raw/clean.
- `outputs/figures/v3/07_per_class_f1.png`: F1 43 lớp, lớp yếu ở trên.

**Phát hiện định lượng:** tương quan giữa support và F1 trên lockbox rất yếu
(Spearman ρ = 0,126; p = 0,419; n = 43 lớp). Lớp 8 (120 km/h, 660 ảnh) có số lỗi
ước tính cao thứ hai, trong khi lớp 37 (30 ảnh) đạt F1 = 1,000. Vì vậy **không
thể quy mọi lỗi cho "thiếu dữ liệu"**; phải xem thêm hình dạng, độ mờ, góc nhìn.
Nguồn: `tables/07b_class_extremes.json`.

**Phân tích nhầm lẫn:** trong 10 cặp nhầm nhiều nhất trên lockbox, **9/10 cặp
cùng hình dạng** và **10/10 cặp cùng màu**; tính cả 19 cặp trong danh sách thì
17/19 cùng hình dạng. Đây là bằng chứng định lượng cho giả thuyết "nhầm do hai
biển có thiết kế giống nhau", không phải do mô hình nhìn nền. Nguồn:
`tables/09_top_confusions_lockbox.csv`; hình `06_confusion_matrix.png`.

### 2.4. Vấn đề 4: thiếu phần vận hành

**Biểu hiện:** báo cáo v2 có thời gian epoch-loop và một con số latency 0,93 ms
nhưng không nêu thiết bị, batch size, phạm vi đo; không có throughput, dung lượng
checkpoint, RAM/VRAM, phần cứng đầy đủ, trạng thái demo.

**Đã đo mới (04/11/2026):**

| Hạng mục | Giá trị | Ghi chú |
|---|---|---|
| Checkpoint D2 | 5.815.209 byte ≈ 5,55 MB | FP32, 1.448.779 params |
| GPU model-only batch 1 | 0,86 ms (median) | CUDA event, FP32 |
| GPU end-to-end 1 ảnh | 1,24 ms (median) | normalize + forward + softmax + top-3 |
| CPU model-only batch 1 | 0,78 ms (median) | 8 thread |
| CPU end-to-end 1 ảnh | 1,33 ms (median) | 8 thread |
| GPU throughput batch 64 | ~79.000 ảnh/s | model-only |
| CPU throughput batch 16 | ~2.250 ảnh/s | model-only |
| Peak VRAM (batch 64) | 81,7 MB allocated / 126 MB reserved | không gồm CUDA context |
| Peak RAM process | ~1.060 MB | toàn process, gồm torch/CUDA |
| Thời gian eval 6.180 ảnh | 1,42 s GPU / 4,93 s CPU | batch 256, gồm normalize |

Nguồn: `tables/13_inference_benchmark.csv`, `.json`, `tables/26_evaluation_time.json`,
`tables/14_hardware_software.json`.

**Lưu ý:** GPU batch 1 **không nhanh hơn CPU** (0,86 vs 0,78 ms) vì overhead
launch/sync. Đây là phát hiện cần nói thẳng, tránh quảng cáo "GPU nhanh hơn".

### 2.5. Vấn đề 5: trộn ba tập đánh giá

Đây là vấn đề nguy hiểm nhất vì nó tạo ra **số sai nhưng trông hợp lý**.

**Bằng chứng cụ thể:** báo cáo v2 viết "OOF … accuracy 0,98004". Khi kiểm tra
lại `oof_predictions.csv`:

| Seed | Accuracy pooled 30.899 ảnh |
|---:|---:|
| 42 | 97,8090% |
| 123 | 98,0291% |
| 2024 | 97,9352% |
| **Trung bình** | **97,9244%** |

98,004% chính là **accuracy screening val của D2** (98,0043% ± 0,297). Đây đúng
là điều người phản biện cảnh báo: "Không nên lấy kết quả Screening của mô hình
D2 rồi trình bày thay cho OOF." Xem `tables/06c_oof_accuracy_check.json`.

**Số đúng phải dùng:**

| Tập | n | Accuracy | Macro-F1/43 | Vai trò |
|---|---:|---:|---:|---|
| Screening val (D2) | 6.180 | 98,0043 | 97,1584 | Chỉ để chọn cấu hình |
| OOF (X2, 3 seed) | 30.899 | **97,9244** | **97,0400** | Chọn winner |
| Lockbox | 8.302 | **98,8758** | **97,5981** | Test nội bộ độc lập |
| Official test raw | 12.630 | 98,6989 | 98,0398 | Benchmark phụ |
| Official test clean | 11.879 | 98,8692 | 98,2499 | Benchmark phụ |

**Quy tắc cho báo cáo mới:** mỗi bảng chỉ dùng một tập; tiêu đề bảng ghi rõ tập;
không đặt OOF cạnh official test mà không có cột "vai trò".

### 2.6. Vấn đề 6: diễn giải sai augmentation

**Sự thật từ artifact:**

- Main effect augmentation "all": **+0,529 điểm**, CI [−1,204; +2,303] → không
  đủ bằng chứng.
- F15 (BN+DO+WD+aug) so với F07 (BN+DO+WD): **−2,555 điểm** trên clean.
- Affine là thành phần hại: **−4,250 điểm**, CI [−6,972; −0,084].
- Corruption trung bình: F07 81,51; F15 80,22; D2 79,65 → aug "all" không tăng
  robustness.

**Vì sao D2 vẫn dùng aug "all"?** Vì D2 là cấu hình đã đăng ký trước (4 khối +
flatten + BN + DO + WD + aug all) và thắng OOF nhờ cấu hình tổng thể; đóng góp rõ
nhất là BN/dropout, không nhờ aug. Báo cáo mới phải viết:

> "D2 thắng OOF nhờ cấu hình tổng thể 4 khối + BN/dropout; bằng chứng mạnh nhất
> là BN/dropout, depth có CI ủng hộ nhưng p Holm 0,373. Augmentation 'all' không
> được chứng minh là có lợi, thậm chí kém F07 2,555 điểm trên clean. Đây là lựa
> chọn theo OOF tổng thể và tie-break, không phải bằng chứng cho aug."

Nguồn: `tables/24_results_evidence.csv`, `tables/05_model_comparison.csv`,
`tables/25_corruption_summary.csv`.

---

## 3. Đối chiếu 34 mục dữ liệu của người phản biện

Tài liệu đánh giá liệt kê 34 mục. Trước khi làm, trạng thái là:

| Trạng thái | Số mục | Ghi chú |
|---|---:|---|
| Đã có trong báo cáo/artifact | 19 | Chủ yếu là kết quả thực nghiệm |
| Cần hoàn thiện cách trình bày | 8 | Per-class, confusion, lỗi, Grad-CAM, calibration, latency |
| Cần bổ sung phép đo mới | 7 | Hardware, tổng thời gian, throughput, model size, RAM/VRAM, triển khai |
| **Tổng** | **34** | |

Sau khi chuẩn bị pack v3:

| Trạng thái | Số mục | Ghi chú |
|---|---:|---|
| Có tài nguyên đầy đủ để viết | 32 | Bảng/hình/số liệu đã kiểm tra |
| Có tài nguyên nhưng cần con người điền | 1 | Kiểm tra thủ công lỗi (17b) |
| Một phần do demo chưa cập nhật | 1 | Trạng thái triển khai (15) |

Ngoài 34 mục trên, còn **2 hạng mục cần thiết bị/dữ liệu ngoài**: đo trên máy
cấu hình thấp và kiểm thử camera thật. Hai hạng mục này không thể tạo từ repo.

Bảng đối chiếu từng mục nằm trong `docs/bao_cao/REPORT_DATA_INVENTORY.md`
(mục 2), kèm file và lệnh sinh lại.

---

## 4. Audit số liệu: 8 điểm lệch đã xác nhận

| # | Mục | Báo cáo cũ | Số đúng | Bằng chứng |
|---:|---|---|---|---|
| 1 | OOF accuracy | 98,004% | **97,9244%** | `06c_oof_accuracy_check.json` |
| 2 | CI lockbox macro-F1 | [95,0511; 97,9422] | mean 97,5981; sd 0,6246; range 3 seed [96,8770; 97,9706] | `06b_lockbox_uncertainty.csv` |
| 3 | Số lỗi screening | 190 lỗi | F15: 190; **D2: 144 lỗi / 23 track** | `17_manual_error_review_summary.json` |
| 4 | Latency | 0,93 ms/ảnh (không ngữ cảnh) | GPU batch 1: 0,86 ms model-only; 1,24 ms end-to-end | `13_inference_benchmark.csv` |
| 5 | Ngưỡng unknown demo | τ = 0,9462 trên val split_random | τ v2 = 0,9590 trên OOF X2 | `threshold_v2_oof.json` |
| 6 | Checkpoint demo | `final_s42` (legacy) | `v2_final_s42` / X2 (D2) | `15_deployment_status.csv` |
| 7 | Tổng thời gian | 4,13 giờ | 3,99 giờ epoch-loop + 8,2 phút baseline | `training_cost_totals.json` |
| 8 | Depth: CI vs Holm | +3,212 [+0,036; +4,440], p Holm 0,373 | Giữ nguyên; phải viết "có bằng chứng vừa, chưa đủ sau Holm" | `main_comparisons.csv` |

Bảng máy đọc: `tables/00_corrections.csv`.

**Điểm 2 cần giải thích thêm:** CI [95,0511; 97,9422] là CI **của OOF**, không
phải lockbox. Với lockbox one-shot, artifact chỉ có 3 seed và aggregate per-class;
không có per-image predictions để bootstrap theo track. Vì vậy báo cáo mới chỉ
được báo **mean ± sd + range 3 seed** và **binomial CI cho accuracy**
[98,649; 99,103] (n = 8.302). Không được tái sử dụng CI của OOF.

**Điểm 3 cần giải thích thêm:** 190 lỗi là của F15 (accuracy 96,56%), không phải
D2 (97,67%). Nếu báo cáo mới lấy D2 làm mô hình cuối thì phải dùng 144 lỗi.
Quan trọng hơn: 144 ảnh lỗi chỉ thuộc **23 track**, nên không được đếm như 144
ca độc lập.

---

## 5. Nguyên nhân gốc

| Nguyên nhân | Biểu hiện | Cách khắc phục |
|---|---|---|
| Báo cáo sinh từ nhật ký kỹ thuật | Kể theo thứ tự đã làm, không theo câu hỏi người đọc | Viết lại theo 15 chương, mỗi chương một câu hỏi |
| Artifact máy đọc nhiều, bảng người đọc ít | JSON/CSV đầy đủ nhưng báo cáo chỉ trích vài số | Xuất bảng Markdown/CSV từ artifact, có caption |
| Không có template báo cáo | Mỗi chương một kiểu | Dùng cấu trúc 15 chương thống nhất |
| Đo vận hành không có protocol | Latency không nêu batch/thiết bị | Protocol benchmark cố định (batch 1/16/32/64, CPU/GPU, FP32) |
| Dùng lẫn tầng đánh giá | 98,004% screening bị gọi là OOF | Bảng 4 tập + quy tắc "một bảng một tập" |
| Diễn giải kết quả âm | Aug "all" được mô tả như yếu tố giúp D2 | Bảng "tốt / không đủ bằng chứng / kém" |

---

## 6. Nguyên tắc và kiến trúc báo cáo mới

### 6.1. Nguyên tắc

1. **Một câu chuyện:** câu hỏi → dữ liệu → thiết kế → kết quả → độ tin cậy →
   hạn chế → hướng phát triển.
2. **Ba tầng tài liệu:** tóm tắt 2–3 trang; báo cáo chính 20–30 trang; phụ lục
   không giới hạn.
3. **Mỗi bảng một thông điệp**, mỗi hình một câu "đọc gì".
4. **Mọi số có nguồn file**; không có số minh họa.
5. **Bốn tập tách bạch:** screening, OOF, lockbox, official test.
6. **"Không đủ bằng chứng" là kết quả hợp lệ**, không làm tròn thành "tốt hơn".
7. **Không dùng test/lockbox để chọn** mô hình, epoch, ngưỡng, augmentation.

### 6.2. Kiến trúc 15 chương và tài nguyên tương ứng

| # | Chương | Nội dung chính | Bảng | Hình |
|---:|---|---|---|---|
| 1 | Tóm tắt dự án | 1–2 trang, bảng 10 dòng | `05`, `06` | — |
| 2 | Giới thiệu bài toán & mục tiêu | bài toán, phạm vi, tiêu chí | `28_glossary` | — |
| 3 | Tổng quan dữ liệu | nguồn, quy mô, 43 lớp | `01`, `class_catalog_summary` | `01`, `17` |
| 4 | Tiền xử lý & chia tập | 3 tầng, audit rò rỉ | `02` | `02` |
| 5 | Kiến trúc mô hình | CNN, D2, so sánh | `23` | `03` |
| 6 | Môi trường & cấu hình huấn luyện | hardware/software, hyperparam | `14`, `19` | — |
| 7 | Thiết kế thực nghiệm | factorial, kiến trúc, hyperparam, aug | `05`, `20` | `08` |
| 8 | Kết quả tổng hợp | bảng một chỗ, 4 tập | `05`, `06`, `06b` | `05` |
| 9 | So sánh & chọn mô hình | vì sao D2, CI, tie-break | `24`, `09` | `05`, `09` |
| 10 | Đánh giá từng lớp | 43 lớp P/R/F1 | `07`, `08`, `22` | `07`, `13` |
| 11 | Phân tích lỗi & robustness | confusion, lỗi, corruption | `09`, `10`, `17`, `25` | `06`, `10`, `15` |
| 12 | Hiệu quả vận hành | latency, throughput, size, RAM/VRAM | `13`, `26` | `14` |
| 13 | Độ tin cậy & tái lập | calibration, repro, nghiệm thu | `12`, `16`, `21` | `12`, `11` |
| 14 | Hạn chế | dữ liệu, mô hình, triển khai | `15`, `24` | — |
| 15 | Kết luận & hướng phát triển | 5 câu hỏi chốt | — | — |

### 6.3. Ba lớp tài liệu

| Lớp | Độ dài | Nội dung | Đối tượng |
|---|---|---|---|
| Tóm tắt | 2–3 trang | bảng kết quả, kết luận, hạn chế | giảng viên, người nghe |
| Báo cáo chính | 20–30 trang | 15 chương trên | người đọc nghiêm túc |
| Phụ lục | không giới hạn | 43 lớp, 111 run, confusion, lỗi, hình | khi bị hỏi |

---

## 7. Tài nguyên đã chuẩn bị

### 7.1. Bảng/số liệu (43 file)

Nhóm theo mục đích:

- **Đính chính & giải thích:** `00_corrections.csv`, `18_metric_worked_example.csv`,
  `28_glossary.csv`.
- **Dữ liệu:** `01_class_catalog.csv`, `02_split_summary.csv`,
  `02b_dataset_summary.csv`, `03_data_quality_per_class.csv`,
  `04_data_quality_summary.json`.
- **Kết quả:** `05_model_comparison.csv`, `06_eval_sets_summary.csv`,
  `06b_lockbox_uncertainty.csv`, `06c_oof_accuracy_check.json`,
  `07_per_class_lockbox.csv`, `08_per_class_oof.csv`,
  `09_top_confusions_lockbox.csv`, `10_confusion_lockbox_*.csv`,
  `11_oof_by_fold_seed.csv`, `21_lockbox_by_seed.csv`,
  `22_test_per_class_raw_clean.csv`, `24_results_evidence.csv`,
  `25_corruption_summary.csv`.
- **Vận hành:** `13_inference_benchmark.csv, 13_inference_benchmark.json`, `14_hardware_software.json`,
  `15_deployment_status.csv`, `19_training_cost_by_group.csv`,
  `20_experiment_index.csv`, `26_evaluation_time.json`.
- **Cơ chế:** `12_calibration_confidence.csv`, `16_gradcam_roi_mass.csv`,
  `16_gradcam_summary.json`, `17_manual_error_review*.csv`,
  `23_architecture_spec.csv`, `threshold_v2_oof.json`,
  `training_cost_totals.json`.

Danh sách đầy đủ + mục đích: `outputs/results/v3/manifest.json`.

### 7.2. Hình (20 file)

15 hình theo đúng danh mục người phản biện đề xuất, cộng 5 hình hỗ trợ:

1. Phân bố 43 lớp — `01_class_distribution.png`
2. Sơ đồ chia tập — `02_split_diagram.png`
3. Kiến trúc D2 — `03_architecture_d2.png`
4. Learning curve train vs val — `04_learning_curves.png`
5. So sánh mô hình — `05_model_comparison.png`
6. Confusion matrix — `06_confusion_matrix.png`
7. F1 từng lớp — `07_per_class_f1.png`
8. Hiệu ứng factorial — `08_factor_effects.png`
9. OOF theo fold × seed — `09_oof_fold_seed.png`
10. Robustness — `10_robustness.png`
11. Grad-CAM — `11_gradcam.png`
12. Calibration — `12_calibration.png`
13. Precision–Recall — `13_precision_recall.png`
14. Hiệu năng suy luận — `14_inference_perf.png`
15. Ví dụ dự đoán — `15_prediction_examples.png`
16. Chất lượng dữ liệu — `16_data_quality.png`
17. Ảnh cực trị chất lượng — `16b_data_quality_examples.png`
18. Ảnh mẫu 43 lớp — `17_class_samples.png`
19. Phiếu rà soát lỗi 1 — `outputs/figures/v3/contact_sheets/error_sheet_01.png`
20. Phiếu rà soát lỗi 2 — `outputs/figures/v3/contact_sheets/error_sheet_02.png`

### 7.3. Script sinh lại

| Script | Vai trò |
|---|---|
| `scripts/report_v3_data.py` | Bảng, xác suất screening, chất lượng dữ liệu, benchmark, Grad-CAM, phiếu lỗi |
| `scripts/report_v3_figures.py` | 20 hình |
| `scripts/report_v3_meta.py` | Hardware/software, deployment, lockbox uncertainty, đính chính, glossary |
| `scripts/report_v3_manifest.py` | Manifest sha256 |

---

## 8. Những gì chưa thể chuẩn bị (và vì sao)

| Hạng mục | Trạng thái | Lý do | Cách xử lý trong báo cáo |
|---|---|---|---|
| Kiểm tra nhãn/lỗi thủ công | **Chưa** | Cần mắt người xem 41 ảnh | Ghi "chưa kiểm tra thủ công"; để cột trống; không kết luận nguyên nhân |
| Camera thật / video | **Chưa** | Không có dữ liệu, cần thu thập và ẩn danh | Ghi là hạn chế; không tuyên bố triển khai |
| Máy cấu hình thấp | **Chưa** | Chỉ có i9-14900KF | Chỉ báo CPU trên máy train; không suy rộng |
| GPU/CPU utilization 111 run | **Chưa** | Không instrument lúc train; không tái tạo được | Ghi "không đo"; đề xuất thêm logging nếu chạy lại |
| CI track-bootstrap cho lockbox | **Chưa** | One-shot, không có per-image predictions | Báo mean ± sd + range; binomial CI cho accuracy |
| Demo dùng checkpoint v2 | **Chưa** | Ngoài phạm vi "chuẩn bị tài nguyên" | Ghi rõ demo đang dùng legacy; tài nguyên đã sẵn |

---

## 9. Rủi ro khi viết báo cáo mới

| Rủi ro | Dấu hiệu | Cách chặn |
|---|---|---|
| Dùng 98,004% làm OOF | Bảng OOF ghi 98,004 | Dùng `06_eval_sets_summary.csv` |
| Dùng CI OOF làm CI lockbox | Lockbox có CI [95,05; 97,94] | Dùng `06b_lockbox_uncertainty.csv` |
| Nói aug "all" giúp D2 | Câu "nhờ aug all" | Dùng `24_results_evidence.csv` |
| So sánh lockbox với official test | Hai số cạnh nhau không ghi vai trò | Cột "vai trò" + cảnh báo |
| Biến "chưa kết luận" thành "tốt hơn" | Depth, width, class weight | Bảng 3 trạng thái |
| Đếm 144 lỗi như 144 ca | "144 lỗi" không kèm 23 track | Ghi "144 ảnh / 23 track" |
| Gọi latency model-only là app | "0,86 ms phản hồi" | Ghi rõ model-only/end-to-end/UI |
| Trình bày demo như sản phẩm | "đã triển khai" | Dùng `15_deployment_status.csv` |

---

## 10. Checklist sẵn sàng viết

- [x] Bảng 43 lớp đầy đủ tên, số ảnh, %.
- [x] Per-class P/R/F1 trên lockbox, OOF, official test.
- [x] Confusion matrix 43×43 + top cặp nhầm.
- [x] Calibration: ECE, Brier, reliability, confidence đúng/sai.
- [x] Robustness 6 corruption × 3 mức + mức giảm.
- [x] Grad-CAM định lượng (mass trong ROI).
- [x] Hardware/software đầy đủ.
- [x] Latency/throughput CPU-GPU theo batch.
- [x] Dung lượng checkpoint, params, RAM/VRAM.
- [x] Trạng thái triển khai demo.
- [x] Bảng kết quả một chỗ + 4 tập tách bạch.
- [x] Bảng "tốt / không đủ bằng chứng / kém".
- [x] Đính chính 8 điểm lệch.
- [x] Glossary 20 thuật ngữ.
- [ ] Điền verdict thủ công 41 ảnh lỗi (cần người).
- [ ] Đo trên máy cấu hình thấp (cần thiết bị).
- [ ] Quay/chụp camera thật (cần dữ liệu).

---

## 11. Đề xuất bước tiếp theo

1. **Duyệt kiến trúc 15 chương** (mục 6.2) và mức độ chi tiết của phụ lục.
2. **Dành 30–45 phút rà soát thủ công** 41 ảnh trong 2 contact sheet, điền
   `17b_manual_error_review_diverse.csv`. Đây là dữ liệu duy nhất còn thiếu để
   phần "phân tích lỗi" có bằng chứng thủ công.
3. **Quyết định có cập nhật demo** sang checkpoint v2 + τ = 0,9590 hay không.
   Nếu có, đây là thay đổi code riêng, cần chạy lại e2e.
4. **Viết báo cáo mới** theo 15 chương, dùng đúng file trong pack v3; không
   viết lại số từ đầu.
5. **Kiểm tra chéo lần cuối**: mọi bảng trong báo cáo phải khớp file nguồn;
   chạy `scripts/report_v3_manifest.py` để chốt hash.

---

## 12. Kết luận

Báo cáo v2 **đủ chất liệu kỹ thuật** nhưng **chưa đủ tư cách một báo cáo dự án**.
Người phản biện đúng ở cả ba tầng: cách kể, cách trình bày số, và cách diễn giải
kết quả âm. Việc tái tạo không cần chạy lại 111 run; cần **tổ chức lại thông tin**
và **bổ sung các phép đo vận hành**. Pack v3 đã chuẩn bị 43 bảng, 20 hình, 4 script,
manifest và 8 đính chính. Sau khi chủ dự án duyệt và hoàn tất 41 ảnh rà soát thủ
công, có thể viết báo cáo mới mà không phải mở lại lockbox hay train lại.
