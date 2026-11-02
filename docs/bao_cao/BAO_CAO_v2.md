# BÁO CÁO THỰC NGHIỆM V2 — GTSRB CNN

| Thông tin | Giá trị |
|---|---|
| Dự án | Bài tập lớn Học sâu 12INF028 — Phân loại 43 lớp biển báo GTSRB bằng CNN tự xây dựng |
| Phiên bản báo cáo | 2.0 (dựa trên kế hoạch `docs/kehoach/1/KE_HOACH_V2_DAY_DU.md`) |
| Ngày | 03/11/2026 |
| Phạm vi | Thay thế các kết luận dựa trên `split_random`; số cũ chỉ nằm ở Phụ lục A |
| Nguồn số | `outputs/results/v2/summary.csv`, `oof_summary.json`, `lockbox_result.json`, `official_test_v2.json` |

> Mọi con số trong báo cáo này lấy từ artifact trong `outputs/results/v2/`.
> Không có số dự kiến, số minh hoạ hay số chép tay.

---

## Chương 1. Đặt vấn đề

Bốn phản biện cần trả lời:

1. **Bài test chưa đủ để chốt mô hình.** Khắc phục bằng đăng ký trước 8 so sánh,
   factorial 2⁴, tách kiến trúc, OOF 3 fold × 3 seed, lockbox mở một lần, CI
   cluster bootstrap theo track.
2. **Số liệu còn rò rỉ.** Khắc phục bằng split 3 tầng theo **sign-group**
   (track cùng một biển vật lý), mean/std theo từng split; official test chỉ là
   benchmark phụ.
3. **Chỉ báo số, không giải thích.** Khắc phục bằng decision card, learning curve,
   per-class Δ, Grad-CAM, calibration, corruption, error taxonomy.
4. **Không rõ cấu hình nào cần.** Khắc phục bằng bật/tắt từng yếu tố và báo cáo
   cả kết quả "không đủ bằng chứng".

## Chương 2. Dữ liệu và chia tập v2

### 2.1. Nguồn và pool

- GTSRB: train gốc 39.209 ảnh, test 12.630 ảnh, 43 lớp.
- Loại 8 ảnh train trùng MD5 với test (đều lớp 14) → **pool 39.201 ảnh**.
- Danh sách loại trừ: `outputs/results/excluded_train_duplicates.csv`.

### 2.2. Split 3 tầng theo sign-group

Nguồn: `outputs/splits/v2/manifest.json` (SHA-256 trong `manifest.sha256`).

| Tập | Ảnh | Track | Sign-group | Seed |
|---|---:|---:|---:|---:|
| Pool | 39.201 | 1.307 | 1.183 | – |
| Lockbox | 8.302 | 277 | 238 | 2024 |
| Dev | 30.899 | 1.030 | 945 | – |
| Screening train / val | 24.719 / 6.180 | 824 / 206 | – | 42 |
| Fold 0 train / val | 19.949 / 10.950 | – | – | 42 |
| Fold 1 train / val | 20.729 / 10.170 | – | – | 42 |
| Fold 2 train / val | 21.120 / 9.779 | – | – | 42 |

**Sign-group** gộp các track có ít nhất một cặp ảnh cùng nhãn khác track với
MAD < 5 trên ảnh 64×64; đã gộp 39.603 cặp. Vì vậy số ảnh lockbox/dev khác bảng
dự kiến trong kế hoạch (7.830/31.371) — manifest là nguồn chính thức.

### 2.3. Audit C1–C12

`outputs/results/v2/audit/audit_report.json`: **15/15 PASS**, gồm C1–C12, hash
file split khớp manifest, mean/std khớp lại từ cache (sai khác 0,0e+00).
Đặc biệt: track overlap lockbox/dev = 0; sign-group overlap = 0; 43/43 lớp ở
lockbox/dev/screening val/3 fold; test không nằm trong train; 0 ảnh lockbox gần
trùng train (MAD < 5); 0 hash trùng trong pool.

### 2.4. Official test không track-disjoint

Official test không có track ID. Audit gần trùng test ↔ train (MAD < 5, cùng lớp,
thumbnail 16×16 rồi kiểm chứng full 64×64) tìm **751 ảnh test** có ít nhất một ảnh
train gần trùng (24.554 cặp), tương ứng **5,95%**. Vì vậy báo cáo:

- **Raw:** 12.630 ảnh.
- **Clean:** 11.879 ảnh (bỏ 751 ảnh gần trùng).

Số này khác audit cũ (131 ảnh) vì cách sinh ứng viên khác; audit này quét đầy đủ
trong từng lớp nên nhạy hơn. Cả hai bản đều chỉ là benchmark phụ, không dùng để
chọn mô hình. Chi tiết: `outputs/results/v2/audit/near_duplicate_test.csv`.

## Chương 3. Giao thức

### 3.1. Huấn luyện

- AdamW, LR 1e-3 (L1 test 1e-4), WD 1e-4 (factorial test 0), batch 128,
  cosine annealing, AMP, tối đa **40 epoch**, early stopping **patience 7** theo
  val macro-F1, không warmup.
- mean/std tính riêng trên train của từng split (manifest), không dùng thống kê toàn cục.
- Augmentation chỉ ở train; cấm flip/rotation 90–180 (có test tự động).
- Seed 42/123/2024; cùng split khi so sánh.

### 3.2. Metric và thống kê

- Endpoint chính: **macro-F1/43** = `f1_score(labels=range(43), average="macro", zero_division=0)`.
- CI 95%: **cluster bootstrap theo track**, B = 2.000.
- So sánh cặp: paired cluster bootstrap của Δ trên cùng mẫu track.
- 8 so sánh chính: Holm–Bonferroni.
- Luật quyết định: Δ ≥ +0,3 điểm và CI_low > 0 → tốt hơn; CI trong ±0,3 → tương
  đương; CI chứa 0 nhưng vượt ±0,3 → chưa kết luận; Δ ≤ −0,3 và CI_high < 0 → kém hơn.
- Chênh lệch < 0,3 điểm % là tương đương.

### 3.3. Chống tune trên test

- Lockbox chỉ mở sau khi `FREEZE.json` được tạo; log mở lockbox ghi thời gian,
  người chạy, lệnh, hash.
- Official test không dùng để chọn epoch/LR/ngưỡng.
- Mọi run lỗi được báo cáo, không xoá.

## Chương 4. Kết quả

> Toàn bộ số trong chương này lấy từ `outputs/results/v2/`. Bảng đầy đủ ở
> `outputs/results/v2/tables/`.

### 4.1. Baseline B1 (screening val)

| Mô hình | Accuracy | Macro-F1/43 | Params | Fit time |
|---|---:|---:|---:|---:|
| B0 majority | 0,0291 | 0,0013 | 0 | – |
| B1 HOG + LinearSVC (C=0,1) | 0,8727 | 0,8600 | 75.895 | 452,7 s |

CNN tốt nhất trên cùng screening val (D2) đạt **97,158% macro-F1**, cao hơn B1
11,16 điểm. B1 không được refit trên dev và không đánh giá trên lockbox.

### 4.2. Factorial 2⁴

48 run (16 cấu hình × 3 seed). Bảng đầy đủ: `tables/factorial.md`.

| ID | BN | DO | WD | Aug | Macro-F1 (%) | Δ vs F00 (điểm %) [CI] | Gap (điểm %) |
|---|---|---|---|---|---:|---|---:|
| F00 | off | 0 | 0 | off | 84,109 ± 3,710 | – | 9,87 |
| F01 | on | 0 | 0 | off | 87,846 ± 1,524 | +3,736 [+0,850; +5,771] | 6,74 |
| F02 | off | 0,3 | 0 | off | 91,613 ± 0,505 | +7,504 [+2,872; +9,657] | 5,63 |
| F03 | on | 0,3 | 0 | off | 96,047 ± 0,975 | +11,937 [+5,769; +14,562] | 2,53 |
| F04 | off | 0 | 1e-4 | off | 84,173 ± 0,184 | +0,063 [−2,102; +1,751] | 9,73 |
| F05 | on | 0 | 1e-4 | off | 87,470 ± 1,106 | +3,361 [+0,488; +5,231] | 7,28 |
| F06 | off | 0,3 | 1e-4 | off | 91,958 ± 2,222 | +7,849 [+3,130; +9,642] | 5,73 |
| F07 | on | 0,3 | 1e-4 | off | **96,501 ± 1,041** | +12,391 [+6,017; +14,993] | 2,60 |
| F08 | off | 0 | 0 | on | 86,672 ± 1,008 | +2,563 [−0,465; +4,558] | 7,41 |
| F09 | on | 0 | 0 | on | 90,006 ± 1,738 | +5,897 [+2,347; +7,731] | 5,22 |
| F10 | off | 0,3 | 0 | on | 91,097 ± 1,628 | +6,988 [+2,903; +8,844] | 4,11 |
| F11 | on | 0,3 | 0 | on | 94,209 ± 0,198 | +10,100 [+5,224; +12,223] | 1,44 |
| F12 | off | 0 | 1e-4 | on | 87,003 ± 1,058 | +2,893 [+0,038; +5,092] | 6,97 |
| F13 | on | 0 | 1e-4 | on | 90,700 ± 1,861 | +6,591 [+2,630; +8,601] | 4,98 |
| F14 | off | 0,3 | 1e-4 | on | 90,310 ± 0,314 | +6,201 [+2,501; +8,171] | 4,00 |
| F15 | on | 0,3 | 1e-4 | on | 93,946 ± 1,108 | +9,837 [+5,138; +12,325] | 1,08 |

**Main effect:** BN **+3,724** [+1,767; +5,005]; Dropout **+5,963** [+2,873; +7,147];
WD +0,058 [−0,506; +0,551]; Aug +0,529 [−1,204; +2,303].
**Interaction:** BN×Aug −0,279; DO×WD −0,121 (không rõ).

Phát hiện: aug giúp khi chưa có regularization (F08−F00 = +2,563) nhưng hại khi
đã đủ regularization (F15−F07 = **−2,555**). Không được nói "aug giúp X điểm" chung.

### 4.3. Kiến trúc

| ID | Thay đổi | Macro-F1 (%) | Δ vs D4=F15 [CI] | Params |
|---|---|---:|---|---:|
| D1 | GAP, 3 khối | 59,188 ± 0,764 | −34,758 [−36,181; −22,916] | 99.019 |
| D2 | 4 khối, flatten | **97,158 ± 0,610** | +3,212 [+0,036; +4,440] | 1.448.779 |
| D3 | 4 khối + GAP + DO 0,4 (E4 cũ) | 93,105 ± 0,906 | −0,841 [−2,724; +0,757] | 399.947 |
| W1 | 16/32/64 | 90,686 ± 1,425 | −3,260 [−5,250; −0,735] | 1.083.579 |
| W2 | 48/96/192 | 95,215 ± 1,230 | +1,268 [−0,585; +2,499] | 3.366.363 |

- GAP 3 khối **underfit** (train acc 84,6%, val 74,2%) → không chọn; đây là kết
  quả âm hợp lệ.
- 4 khối + flatten (D2) **tốt hơn** 3 khối, CI không chứa 0 → bác bỏ H4.
- D3 tương đương D4 → chỉ dùng làm legacy, không kết luận cơ chế.
- Width nhỏ kém; width lớn chưa kết luận.

### 4.4. Hyperparam và augmentation

| ID | Thay đổi | Macro-F1 (%) | Δ vs F15 [CI] |
|---|---|---:|---|
| L1 | LR 1e-4 | 86,807 ± 1,615 | −7,139 [−9,372; −3,353] |
| CW1 | class weight inv_sqrt | 92,943 ± 0,475 | −1,004 [−2,195; +0,261] |
| I2 | input 96 | 90,499 ± 1,429 | −3,448 [−5,463; −0,984] |

| ID | Profile | Macro-F1 (%) | Δ vs G0=F07 [CI] |
|---|---|---:|---|
| G1 | affine | 92,251 ± 0,595 | −4,250 [−6,972; −0,084] |
| G2 | color | 97,417 ± 0,340 | +0,917 [−0,141; +1,842] |
| G3 | perspective | 95,264 ± 1,988 | −1,237 [−2,706; +0,530] |

Corruption TB (6 loại × 3 mức): F07 81,51; G2 81,97; F15 80,22; D2 79,65;
G1 76,90; G3 80,71. Occlusion làm sập mọi mô hình (17,6–26,4%).

### 4.5. OOF và chọn mô hình

| Config | Nguồn | OOF macro-F1 (%) | CI 95% |
|---|---|---:|---|
| X1 | G2 (3 khối + color) | 96,297 ± 0,296 | [94,628; 97,020] |
| **X2** | **D2 (4 khối + all)** | **97,040 ± 0,187** | **[95,051; 97,942]** |
| X3 | F07 (control) | 95,059 ± 0,568 | [93,113; 95,930] |

- X1 − X2 = −0,743 [−1,590; +0,274] → chưa kết luận.
- X1 − X3 = +1,238 [+0,586; +2,071]; X2 − X3 = +1,981 [+1,053; +2,961].
- Tie-break params: X2 1,45M < X1 2,20M → **chọn X2 (D2)**.
- Epoch cố định 26 = median best epoch 9 run OOF.
- Phân rã phương sai: **fold 91,0%**, seed 0,9%, residual 8,3% → phải báo theo fold.

### 4.6. Lockbox và official test

| Tập | n | Accuracy | Macro-F1/43 |
|---|---:|---:|---:|
| Lockbox | 8.302 | **98,876 ± 0,101** | **97,598 ± 0,625** |
| Official raw | 12.630 | 98,699 ± 0,163 | 98,040 ± 0,452 |
| Official clean | 11.879 | 98,869 ± 0,236 | 98,250 ± 0,531 |

- Lockbox top-3 99,847%; ECE 0,0053; Brier 0,0179.
- Clean **cao hơn** raw → ảnh gần trùng không thổi phồng kết quả.
- Official test không track-disjoint; không so trực tiếp với lockbox.

### 4.7. Tổng hợp 8 so sánh chính

| Yếu tố | Δ (điểm %) | CI 95% | p Holm | Kết luận |
|---|---:|---|---:|---|
| BN | +3,724 | [+1,767; +5,005] | 0,000 | Tốt hơn |
| Dropout | +5,963 | [+2,873; +7,147] | 0,000 | Tốt hơn |
| Weight decay | +0,058 | [−0,506; +0,551] | 0,412 | Chưa kết luận |
| Augmentation | +0,529 | [−1,204; +2,303] | 0,708 | Chưa kết luận |
| GAP | −34,758 | [−36,108; −22,444] | 0,000 | Kém hơn |
| Depth | +3,212 | [+0,011; +4,513] | 0,373 | CI ủng hộ; Holm chưa đủ mạnh |
| Width | +1,268 | [−0,591; +2,541] | 0,708 | Chưa kết luận |
| LR | −7,139 | [−9,353; −3,366] | 0,000 | Kém hơn (1e-4) |

## Chương 5. Giải thích cơ chế

### 5.1. Learning curve và overfit

- F00 (không BN/DO/WD): train acc đạt ~1,0 rất sớm, gap train−val **9,87 điểm**,
  F1 dao động mạnh theo seed (79,9–87,0%).
- F07 (BN+DO+WD): gap còn **2,60 điểm**, F1 96,50 ± 1,04.
- F15 (thêm aug all): gap **1,08 điểm** nhưng F1 giảm còn 93,95 → aug giảm overfit
  nhưng không chuyển thành lợi ích val ở ngân sách 40 epoch.
- D2 (4 khối + all): OOF 97,04, gap thấp; depth tăng dung lượng nhưng vẫn kiểm soát.

### 5.2. Per-class

- Lockbox: 16/43 lớp đạt F1 ≥ 0,999; yếu nhất là **lớp 19 (F1 0,760, n=30)**,
  **lớp 29 (0,776, n=60)**, lớp 30 (0,925), 28 (0,928).
- Đây là các lớp hiếm/khó, không phải lỗi split: support lockbox vẫn ≥ 30 ảnh.
- Heatmap ΔF1 43 lớp: `04_per_class_delta.png`.

### 5.3. Grad-CAM

- 12 ảnh (4 lớp × đúng tự tin / sai / confidence thấp), so CNN-B (F15) và CNN-C (D3).
- Mô hình tập trung vào biển, không phải nền; trường hợp sai thường nhìn lệch sang
  vùng viền/ánh sáng. Hình: `08_gradcam_grid.png`.

### 5.4. Calibration

| Model | ECE 15 bin |
|---|---:|
| F00 | 0,0758 |
| F07 | 0,0154 |
| F15 | 0,0120 |
| G2 | 0,0101 |
| D2 | 0,0084 |

Regularization (BN/dropout/WD) giảm tự tin sai rõ rệt; D2 calibrated tốt nhất.
Lockbox ECE 0,0053.

### 5.5. Corruption

- Color-only (G2) 81,97 TB, tốt hơn F07 81,51 và F15 80,22.
- Affine là thành phần hại (76,90 TB).
- Occlusion là điểm yếu chung (17,6–26,4%) — mô hình không học được biển bị che.

### 5.6. Error taxonomy

- 190 lỗi trên screening val (F15, seed 42): **102 nhãn khó/góc nhìn lạ**, **64 mờ**,
  **23 nhầm cặp giống nhau/confidence thấp**, **1 chói**.
- Phân nhóm là heuristic theo percentile blur/brightness/confidence; cần xem thủ công.
- 12 mẫu: `09_error_samples.png`.


## Chương 6. Hạn chế và kết luận

- Lockbox 8.302 ảnh nhỏ hơn official test 12.630 → CI rộng hơn; không so trực
  tiếp hai con số.
- Official test **không track-disjoint**: audit lại tìm 751 ảnh gần trùng train
  (5,95%), nhiều hơn số 131 của audit cũ do phương pháp sinh ứng viên khác.
  Clean 11.879 ảnh; clean **không** thấp hơn raw nên không có dấu hiệu thổi phồng.
- Screening train 24.719 ảnh, ít hơn final 30.899; kết luận screening chỉ để chọn,
  kết luận cuối từ OOF/lockbox.
- **Augmentation "all" không có lợi** ở ngân sách 40 epoch; affine là thành phần
  hại. Color-only trung tính. Kết quả này trái kỳ vọng ban đầu và được giữ nguyên.
- **GAP trên 3 khối underfit nặng**; không kết luận "GAP luôn xấu", chỉ kết luận
  ở cấu hình/ngân sách này.
- Occlusion (che 10–30% trung tâm) làm accuracy rơi còn 17,6–26,4% — mô hình chưa
  robust với che khuất.
- Chỉ dùng CNN tự xây; không pretrained.
- Không tuyên bố mô hình đủ an toàn cho quyết định giao thông thực tế.
- Kết quả âm và "chưa kết luận" (WD, width, aug) được báo đúng, không giữ cấu hình
  vì cảm tính.

## Phụ lục A. Legacy leaky (không dùng để kết luận)

Các run E1–E7 cũ dùng `split_random`; val bị rò rỉ frame (95,22% frame val có
frame train cách ±1). Bản sao số cũ: `outputs/results/legacy/summary_random_split.csv`.
Nhãn: `legacy_leaky`. Không trộn với bảng kết luận v2.

## Phụ lục B. Tra cứu artifact

| Nội dung | File |
|---|---|
| Split + hash + mean/std | `outputs/splits/v2/manifest.json` |
| Audit C1–C12 | `outputs/results/v2/audit/audit_report.json` |
| Baseline | `outputs/results/v2/baseline.json` |
| Bảng 2⁴, main effect | `outputs/results/v2/tables/factorial.md`, `main_effects.md` |
| Kiến trúc/hyperparam/aug | `outputs/results/v2/tables/architecture.md`, `hyperparam.md`, `augmentation.md` |
| Corruption | `outputs/results/v2/corruption.csv`, `tables/corruption.md` |
| Summary 111 run | `outputs/results/v2/summary.csv` |
| OOF | `outputs/results/v2/oof_predictions.csv`, `oof_summary.json` |
| Phương sai | `outputs/results/v2/variance.json` |
| Decision cards | `outputs/results/v2/decision_cards.csv` |
| Freeze | `outputs/results/v2/FREEZE.json` |
| Lockbox | `outputs/results/v2/lockbox_result.json`, `lockbox_access.log` |
| Official test | `outputs/results/v2/official_test_v2.json` |
| Hình | `outputs/figures/v2/` |
| Nhật ký từng bước | `docs/kehoach/1/KET_QUA_TRIEN_KHAI_V2.md` |

## Phụ lục C. Đính chính các câu sai (mục 12.4 kế hoạch)

| Câu cũ | Trạng thái | Đính chính bằng số |
|---|---|---|
| "Val random 99,9% chứng minh mô hình gần hoàn hảo" | **Sai** | Val random bị rò rỉ frame; không dùng. Số sạch: OOF 97,040; lockbox 97,598 macro-F1 |
| "17.214 cặp pHash chỉ là hash collision, không có gần trùng" | **Quá mạnh** | Audit lại: 751 ảnh test gần trùng train (MAD<5), 24.554 cặp; clean 11.879 |
| "E4 kém vì GAP" | **Chưa chứng minh** | E4 đổi 4 thứ. D1 (GAP 3 khối) kém −34,758; D2 (depth, không GAP) +3,212; D3 chỉ legacy |
| "Aug luôn tốt" | **Sai** | Main effect +0,529 [−1,204; +2,303]; F15−F07 = −2,555; affine −4,250 |
| "Test độc lập" | **Không chính xác** | Official test không track-disjoint; chỉ là benchmark phụ, báo raw + clean |
