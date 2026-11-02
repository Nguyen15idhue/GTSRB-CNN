# MẪU KẾT QUẢ GIẢ ĐỊNH — ĐỂ TRẢ LỜI 4 VẤN ĐỀ

> ## ⚠️ CẢNH BÁO
> **Toàn bộ số trong file này là GIẢ ĐỊNH**, chỉ để minh họa **cấu trúc thông tin** sẽ có sau khi chạy thật.
> **KHÔNG phải kết quả chạy thật. KHÔNG chép bất kỳ số nào vào báo cáo/slide.**
> Khi chạy thật, thay toàn bộ bằng `outputs/results/v2/`: `summary.csv`, `oof_predictions.csv`, `decision_cards.csv`, `lockbox_result.json`, `official_test_v2.json`.
>
> Mục đích của file: trả lời câu hỏi *"sau khi làm xong, tôi sẽ có những bảng/hình/thông tin gì, và mỗi thông tin trả lời vấn đề nào?"*

---

## 0. Kịch bản giả định dùng trong file

Đây là **một kịch bản minh họa** (không phải dự đoán):

- CNN-B (3 khối 32/64/128 + BN + dropout 0,3 + WD 1e-4 + aug) thắng rõ trên OOF.
- BN và dropout có tác dụng; WD chưa đủ bằng chứng; aug **không tăng clean accuracy** nhưng **tăng robustness**.
- 4 khối và GAP kém hơn CNN-B; width lớn hơn 32/64/128 không tăng.
- LR 1e-3 tốt nhất; 3e-3 phân kỳ; class weight tương đương về macro-F1 nhưng tăng recall lớp hiếm.
- Input 64 là lựa chọn cân bằng; 96 không hơn đáng kể nhưng chậm hơn.

> Kết quả thật có thể khác. Luật quyết định trong `KE_HOACH_V2_DAY_DU.md` mục 4.5 sẽ quyết định, không phải kịch bản này.

---

## 1. Bảng split và audit — trả lời vấn đề "còn leak không"

### 1.1. Split manifest

| Tập | Track | Ảnh | Lớp | Overlap track | Overlap sign-group | Vai trò |
|---|---:|---:|---:|---:|---:|---|
| Pool | 1.307 | 39.201 | 43 | – | – | Sau loại 8 MD5 |
| Lockbox | 261 | 7.830 | 43 | **0** | **0** | Test one-shot |
| Dev | 1.046 | 31.371 | 43 | 0 | 0 | Train/val |
| Screening train | 838 | 25.131 | 43 | 0 | 0 | Train |
| Screening val | 208 | 6.240 | 43 | 0 | 0 | Val |
| Fold 0 train / val | ~697 / ~349 | 20.541 / 10.830 | 43 / 43 | 0 | 0 | OOF |
| Fold 1 train / val | ~697 / ~349 | 21.360 / 10.011 | 43 / 43 | 0 | 0 | OOF |
| Fold 2 train / val | ~697 / ~349 | 20.907 / 10.464 | 43 / 43 | 0 | 0 | OOF |
| Official test | – | 12.630 | 43 | không đo được | không đo được | Benchmark phụ |
| Official test clean | – | 12.499 | 43 | không đo được | không đo được | Sau bỏ 131 ảnh gần trùng |

### 1.2. Audit official test

| Hạng mục | Số | Tỷ lệ | Ghi chú |
|---|---:|---:|---|
| Ảnh test gần trùng train (MAD < 5) | 131 / 12.630 | 1,04% | Ứng với 24 track train |
| Ảnh test gần trùng train (MAD < 10) | 409 / 12.630 | 3,24% | Mức rộng hơn |
| Trong 131 ảnh, dự đoán đúng | 130 / 131 | 99,24% | Không phải nguồn thổi phồng lớn |
| Bỏ 131 ảnh: accuracy | 98,559% → 98,552% | −0,007 điểm | Ảnh hưởng nhỏ |
| Class 10 trong 131 ảnh | 77 / 131 | 58,8% | Tập trung ở một lớp lớn |
| Lockbox có ảnh gần trùng train | 0 | 0% | Sạch theo thiết kế |

### 1.3. So sánh "leak cũ" và "sạch mới"

| Bộ đánh giá | Accuracy | Macro-F1/43 | Track overlap | Ảnh gần trùng | Kết luận |
|---|---:|---:|---:|---:|---|
| Val random cũ (E3) | 99,94% | 99,95% | 1.306 track | 95,22% frame cách ±1 | **Legacy leaky — không dùng** |
| Val stratified_group cũ (final s42) | 98,35% | 97,06% | 0 | 0 | Sạch track, nhưng đã dùng chọn mô hình |
| **Lockbox v2 (giả định)** | **98,0%** | **96,8%** | **0** | **0** | **Số độc lập chính** |
| Official test raw (giả định) | 98,2% | 97,9% | không đo được | 131 ảnh | Benchmark phụ |
| Official test clean (giả định) | 98,1% | 97,8% | không đo được | 0 | Kiểm tra độ nhạy |

**Thông tin trả lời vấn đề 2:**

- Chênh lệch **99,95% → 96,8%** (khoảng 3,1 điểm F1) là bằng chứng định lượng của leak cũ.
- Lockbox 0 track/sign-group overlap, 0 ảnh gần trùng → không còn đường leak.
- Official test vẫn được báo cáo nhưng **không gọi là độc lập**.
- Nếu giảng viên hỏi "số mới thấp hơn 98,56% cũ?", trả lời: do test sạch/khó hơn, không phải do train ít hơn (final vẫn 31.371 ảnh).

---

## 2. Baseline — trả lời "có mốc so sánh chưa"

| ID | Mô hình | Accuracy | Macro-F1/43 | Thời gian | Ghi chú |
|---|---|---:|---:|---:|---|
| B0 | Majority class | 5,7% | 0,25% | – | Sàn |
| B1 | HOG 64×64 + LinearSVC | 96,6% | 97,4% | 51 s trích + 540 s fit | C chọn trên screening val |
| B1b | HOG refit trên dev | 96,4% | 97,1% | ~11 phút | Test lockbox |
| F00 | CNN-A | 95,8% | 94,8% | 5,1 phút | Mốc CNN tối thiểu |

**Thông tin trả lời vấn đề 1:** có baseline cổ điển và baseline CNN; mọi mô hình sâu phải vượt mốc này.

---

## 3. Factorial 2⁴ — trả lời "BN/dropout/WD/aug có cần không"

### 3.1. Bảng đầy đủ 16 cấu hình

| ID | BN | Dropout | WD | Aug | Val acc | Macro-F1/43 | Δ vs F00 | CI 95% Δ | Gap train−val | Best epoch | Params | Time |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| F00 | off | 0 | 0 | off | 95,8 | 94,8 ± 0,2 | – | – | 4,2 | 24 | 2,20M | 5,0 ph |
| F01 | **on** | 0 | 0 | off | 97,3 | 96,3 ± 0,2 | +1,5 | [1,2; 1,8] | 3,1 | 27 | 2,20M | 5,1 ph |
| F02 | off | **0,3** | 0 | off | 96,3 | 95,3 ± 0,2 | +0,5 | [0,2; 0,8] | 3,8 | 26 | 2,20M | 5,1 ph |
| F03 | on | 0,3 | 0 | off | 98,0 | 97,0 ± 0,2 | +2,2 | [1,9; 2,5] | 2,4 | 30 | 2,20M | 5,2 ph |
| F04 | off | 0 | **1e-4** | off | 96,1 | 95,1 ± 0,2 | +0,3 | [0,0; 0,6] | 3,9 | 25 | 2,20M | 5,1 ph |
| F05 | on | 0 | 1e-4 | off | 97,6 | 96,6 ± 0,2 | +1,8 | [1,5; 2,1] | 2,8 | 29 | 2,20M | 5,2 ph |
| F06 | off | 0,3 | 1e-4 | off | 96,6 | 95,6 ± 0,2 | +0,8 | [0,5; 1,1] | 3,5 | 28 | 2,20M | 5,2 ph |
| F07 | on | 0,3 | 1e-4 | off | 98,1 | 97,1 ± 0,2 | +2,3 | [2,0; 2,6] | 2,1 | 31 | 2,20M | 5,2 ph |
| F08 | off | 0 | 0 | **on** | 95,9 | 94,9 ± 0,2 | +0,1 | [−0,1; 0,3] | 3,9 | 25 | 2,20M | 5,3 ph |
| F09 | on | 0 | 0 | on | 97,5 | 96,5 ± 0,2 | +1,7 | [1,4; 2,0] | 2,9 | 28 | 2,20M | 5,3 ph |
| F10 | off | 0,3 | 0 | on | 96,4 | 95,4 ± 0,2 | +0,6 | [0,3; 0,9] | 3,6 | 27 | 2,20M | 5,3 ph |
| F11 | on | 0,3 | 0 | on | 98,2 | 97,2 ± 0,2 | +2,4 | [2,1; 2,7] | 2,2 | 32 | 2,20M | 5,3 ph |
| F12 | off | 0 | 1e-4 | on | 96,2 | 95,2 ± 0,2 | +0,4 | [0,1; 0,7] | 3,7 | 26 | 2,20M | 5,3 ph |
| F13 | on | 0 | 1e-4 | on | 97,7 | 96,7 ± 0,2 | +1,9 | [1,6; 2,2] | 2,7 | 30 | 2,20M | 5,3 ph |
| F14 | off | 0,3 | 1e-4 | on | 96,7 | 95,7 ± 0,2 | +0,9 | [0,6; 1,2] | 3,4 | 29 | 2,20M | 5,3 ph |
| F15 | on | 0,3 | 1e-4 | on | **98,3** | **97,3 ± 0,2** | **+2,5** | **[2,1; 2,9]** | **1,7** | **34** | **2,20M** | **5,4 ph** |

> Ghi chú: cột **Gap** = train macro-F1/43 − val macro-F1/43 (trung bình 3 seed), không phải accuracy gap.

### 3.2. Main effect

| Yếu tố | Bật | Tắt | Δ macro-F1 | CI 95% | p (Holm) | Kết luận |
|---|---|---|---:|---|---:|---|
| BN | on | off | **+1,6** | [1,3; 1,9] | <0,001 | **Giữ** |
| Dropout | 0,3 | 0 | **+0,6** | [0,3; 0,9] | 0,002 | **Giữ** |
| WD | 1e-4 | 0 | +0,2 | [0,0; 0,5] | 0,060 | **Chưa kết luận**; giữ 1e-4 vì không hại |
| Aug | on | off | +0,1 | [−0,1; 0,3] | 0,310 | **Tương đương trên clean**; xem corruption |

### 3.3. Interaction

| Cặp | Δ interaction | CI 95% | Cách đọc |
|---|---|---:|---|
| BN × Aug | +0,1 | [−0,1; 0,3] | Aug không làm BN mất tác dụng |
| DO × WD | 0,0 | [−0,2; 0,2] | Không có interaction đáng kể |
| BN × DO | +0,2 | [0,0; 0,4] | BN và dropout bổ trợ nhẹ |

**Thông tin trả lời vấn đề 4:**

- BN: **cần**, +1,6 điểm, CI không chứa 0.
- Dropout: **cần**, +0,6 điểm.
- WD: **chưa đủ bằng chứng**, chỉ +0,2; không được nói "WD tốt" nếu CI chứa 0.
- Aug: **không cần cho clean accuracy**, nhưng cần cho robustness (mục 6); đây là câu trả lời có sắc thái, không phải "có/không" đơn giản.
- Bảng cũng cho thấy F15 (E3 cũ) = 97,3, F00 (CNN-A) = 94,8: khoảng cách 2,5 điểm, không phải 99,9% như val random.

---

## 4. Kiến trúc — trả lời "vì sao CNN-B, không phải A/C/GAP/4 khối"

| ID | Cấu hình | Params | Val acc | Macro-F1/43 | Δ vs D4 | CI 95% Δ | Gap | Latency (ms/ảnh) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| D4 | 3 khối + flatten (= F15) | 2,20M | 98,3 | **97,3 ± 0,2** | – | – | 1,7 | 4,8 |
| D1 | 3 khối + GAP | **0,42M** | 97,3 | 96,3 ± 0,2 | **−1,0** | [−1,3; −0,7] | 2,0 | 3,9 |
| D2 | 4 khối + flatten | 2,75M | 97,6 | 96,6 ± 0,2 | **−0,7** | [−1,0; −0,4] | 1,7 | 5,2 |
| D3 | 4 khối + GAP + DO 0,4 (= E4) | **0,40M** | 97,1 | 96,1 ± 0,3 | **−1,2** | [−1,5; −0,9] | 2,1 | 2,1 |
| C1 | 4 khối + GAP + DO 0,3 | 0,42M | 97,2 | 96,2 ± 0,2 | **−1,1** | [−1,4; −0,8] | 2,0 | 2,1 |
| W1 | 3 khối 16/32/64 | 0,14M | 97,8 | 96,8 ± 0,3 | −0,5 | [−0,8; −0,2] | 2,4 | 2,6 |
| W2 | 3 khối 48/96/192 | 4,95M | 98,2 | 97,2 ± 0,2 | −0,1 | [−0,3; 0,1] | 2,0 | 7,4 |
| W3 | 3 khối 64/128/256 | 8,80M | 98,3 | 97,3 ± 0,2 | 0,0 | [−0,2; 0,2] | 1,7 | 9,5 |

> Ghi chú: cột **Gap** = train macro-F1/43 − val macro-F1/43; **Latency** đo model-only, batch 1, median 100 lần.

**Cách đọc:**

- D1 (chỉ thêm GAP) kém D4 **1,0 điểm** → GAP không miễn phí; nếu CI chứa 0 thì mới được chọn GAP.
- D2 (chỉ thêm depth) kém D4 **0,7 điểm** → 4 khối overfit hơn, không đáng.
- D3/C1 tách được confound dropout 0,4: C1 (0,3) vẫn kém, nên không phải do dropout.
- W3 (8,8M) không hơn D4 (2,2M) → width lớn không đáng; W2 cũng tương đương nhưng 2,25× params.
- **Kết luận giả định:** chọn D4/CNN-B; nếu D1 tương đương trong CI, đổi sang GAP để giảm 5× params.

**Thông tin trả lời vấn đề 1 và 4:** đây là bảng chứng minh "vì sao CNN-B" bằng số, không phải bằng cảm tính; đồng thời trả lời "4 khối/GAP/width có cần không".

---

## 5. Hyperparameter — trả lời "vì sao lr 1e-3, dropout 0,3, input 64…"

| ID | Yếu tố | Giá trị | Macro-F1/43 | Δ vs F15 | CI 95% Δ | Ghi chú |
|---|---|---|---:|---:|---|---|
| L1 | LR | 1e-4 | 94,9 ± 0,3 | −2,4 | [−2,8; −2,0] | Underfit |
| L2 | LR | 3e-4 | 96,2 ± 0,2 | −1,1 | [−1,4; −0,8] | Chậm |
| L3 | LR | 1e-3 | **97,3** | – | – | Mốc |
| L4 | LR | 3e-3 | NaN (epoch 4) | – | – | **Phân kỳ**; báo cáo là kết quả |
| L5 | LR + warmup | 3e-3, warmup 3 | 96,0 ± 0,3 | −1,3 | [−1,7; −0,9] | Warmup cứu được một phần |
| P1 | Dropout | 0,2 | 96,9 ± 0,2 | −0,4 | [−0,7; −0,1] | Hơi thấp |
| P2 | Dropout | 0,5 | 95,8 ± 0,4 | −1,5 | [−2,0; −1,0] | Underfit |
| W4 | WD | 1e-3 | 96,4 ± 0,3 | −0,9 | [−1,3; −0,5] | Quá mạnh |
| CW1 | Class weight | inv_sqrt | 97,4 ± 0,2 | +0,1 | [−0,1; 0,3] | Tương đương macro-F1; recall lớp hiếm +2,1 |
| CW2 | Class weight | inv_freq | 97,1 ± 0,2 | −0,2 | [−0,5; 0,1] | Không hơn |
| LS1 | Label smoothing | 0,1 | 97,2 ± 0,2 | −0,1 | [−0,3; 0,1] | Tương đương |
| I1 | Input | 48 | 96,4 ± 0,3 | −0,9 | [−1,2; −0,6] | Mất chi tiết |
| I2 | Input | 96 | 97,4 ± 0,2 | +0,1 | [−0,1; 0,3] | Tương đương; chậm 1,8× |
| BS1 | Batch | 64 | 97,2 ± 0,2 | −0,1 | [−0,3; 0,1] | Tương đương |
| BS2 | Batch | 256 | 97,1 ± 0,2 | −0,2 | [−0,4; 0,0] | Tương đương |

**Thông tin trả lời vấn đề 1 và 4:**

- 3e-3 **phân kỳ thật** (NaN ở epoch 4); 3e-3 + warmup vẫn kém → kết luận "1e-3 là lựa chọn ổn định", không phải "3e-3 không chạy được vì 10 epoch".
- Dropout 0,3 là điểm tốt; 0,5 underfit.
- Input 96 không hơn đáng kể nhưng chậm 1,8× → chọn 64.
- Class weight không cải thiện macro-F1, nhưng tăng recall lớp hiếm; báo trade-off, không mặc định bật.

---

## 6. Augmentation — trả lời "aug có thực sự cần không"

### 6.1. Clean accuracy

| ID | Profile | Macro-F1/43 | Δ vs G0 | CI 95% Δ |
|---|---|---:|---:|---|
| G0 | none | 97,1 ± 0,2 | – | – |
| G1 | affine | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |
| G2 | color | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |
| G3 | perspective | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |
| G4 | affine + color | 97,3 ± 0,2 | +0,2 | [−0,1; 0,3] |
| G5 | all (= F15) | **97,3 ± 0,2** | +0,2 | [−0,1; 0,3] |
| G6 | affine + perspective | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |
| G7 | color + perspective | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |
| G8 | all + color mạnh | 97,2 ± 0,2 | +0,1 | [−0,1; 0,3] |

**Kết luận clean:** mọi profile aug đều **tương đương** (CI chứa 0 hoặc chỉ +0,2). Không thể nói "aug tăng accuracy sạch".

### 6.2. Corruption robustness

| Profile | Blur | Noise | Bright | Contrast | JPEG | Occlusion | Scale | Rotate | TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 none | 72,1 | 70,5 | 82,3 | 80,2 | 79,4 | 75,6 | 78,1 | 79,3 | **78,2** |
| G5 all | 84,3 | 82,1 | 90,1 | 88,0 | 87,2 | 83,4 | 86,2 | 87,0 | **86,4** |
| G8 all + color mạnh | 83,8 | 80,9 | 89,4 | 87,2 | 86,5 | 82,1 | 85,4 | 86,7 | **85,1** |

**Δ G5 − G0:** clean **+0,2** (tương đương), corruption trung bình **+8,2** [7,1; 9,3] (rõ ràng).

**Thông tin trả lời vấn đề 4:**

- Nếu chỉ tối ưu GTSRB sạch: **aug không cần**; bỏ aug giảm chi phí train/demo.
- Nếu mục tiêu là ảnh ngoài đời/demo: **giữ aug** vì tăng robustness ~8 điểm, dù clean không tăng.
- Đây là câu trả lời "có điều kiện", trung thực; không tuyên bố "aug luôn tốt".

---

## 7. OOF, lockbox, official test — trả lời "số cuối là bao nhiêu"

### 7.1. OOF 3-fold × 3 seed

| Cấu hình | OOF accuracy | OOF macro-F1/43 | CI 95% | Δ vs CNN-B | CI 95% Δ |
|---|---:|---:|---|---:|---|
| CNN-A (F00) | 96,3 | 95,2 | [94,9; 95,5] | −1,9 | [−2,3; −1,5] |
| **CNN-B (F15)** | **98,2** | **97,1** | **[96,8; 97,4]** | – | – |
| CNN-C (D3) | 97,1 | 96,0 | [95,7; 96,3] | −1,1 | [−1,5; −0,7] |
| GAP (D1) | 97,3 | 96,2 | [95,9; 96,5] | −0,9 | [−1,3; −0,5] |

### 7.2. Lockbox một lần

| Cấu hình | Accuracy | Macro-F1/43 | CI 95% | Δ vs CNN-B | CI 95% Δ |
|---|---:|---:|---|---:|---|
| CNN-A | 96,1 | 95,0 | [94,3; 95,7] | −1,8 | [−2,6; −1,0] |
| **CNN-B** | **98,0** | **96,8** | **[96,1; 97,4]** | – | – |
| CNN-C | 96,8 | 95,7 | [95,0; 96,4] | −1,1 | [−1,9; −0,3] |
| GAP | 97,0 | 95,9 | [95,2; 96,6] | −0,9 | [−1,7; −0,1] |

### 7.3. Official test

| Cấu hình | Raw acc | Raw macro-F1 | Clean acc | Clean macro-F1 |
|---|---:|---:|---:|---:|
| CNN-A | 97,4 | 96,9 | 97,3 | 96,8 |
| **CNN-B** | **98,2** | **97,9** | **98,1** | **97,8** |
| CNN-C | 97,7 | 97,3 | 97,6 | 97,2 |
| GAP | 97,9 | 97,5 | 97,8 | 97,4 |

**Cách đọc:**

- OOF và lockbox **cùng kết luận**: CNN-B > GAP > CNN-C > CNN-A.
- Lockbox thấp hơn OOF một chút (96,8 vs 97,1) và thấp hơn official test (96,8 vs 97,9): official test dễ hơn/không track-disjoint; không so trực tiếp.
- CI lockbox rộng hơn OOF (261 track vs 1.046 track), đúng dự kiến.
- **Số chính để báo cáo là lockbox 96,8 [96,1; 97,4]**, không phải official test.

---

## 8. Phân rã phương sai — trả lời "seed hay fold quan trọng hơn"

| Nguồn | Phương sai | % tổng | Ý nghĩa |
|---|---:|---:|---|
| Seed | 0,12 | 25% | Dao động do khởi tạo |
| Fold | 0,28 | 58% | **Cách chia track quan trọng hơn seed** |
| Residual | 0,08 | 17% | Nhiễu còn lại |

**Kết luận:** multi-seed một mình không đủ; phải báo cáo theo fold/OOF. Nếu chỉ chạy 3 seed trên một split, CI sẽ hẹp giả.

---

## 9. Per-class — trả lời "mô hình tốt ở đâu, kém ở đâu"

| Lớp | Support lockbox | CNN-A F1 | CNN-B F1 | CNN-C F1 | Nhận xét |
|---:|---:|---:|---:|---:|---|
| 0 | 30 | 0,82 | **0,91** | 0,86 | Lớp hiếm, BN giúp rõ |
| 10 | 390 | 0,98 | **0,99** | 0,99 | Lớp lớn, dễ |
| 19 | 30 | 0,78 | **0,88** | 0,82 | Lớp hiếm, khó |
| 21 | 60 | 0,90 | **0,94** | 0,92 | Trung bình |
| 23 | 90 | 0,93 | **0,96** | 0,94 | Trung bình |
| 27 | 60 | 0,85 | **0,92** | 0,88 | Hiếm |
| 37 | 30 | 0,80 | **0,90** | 0,84 | Hiếm |
| 42 | 60 | 0,88 | **0,93** | 0,90 | Hiếm |

**Cặp nhầm chính (giả định):**

| Cặp | Số lỗi CNN-B | Nhận xét |
|---|---:|---|
| 0 → 19 | 4 | Biển số giống nhau, ảnh nhỏ |
| 19 → 0 | 3 | Nhầm hai chiều |
| 21 → 23 | 5 | Hình dạng/khung giống |
| 30 → 31 | 3 | Góc nhìn |
| 37 → 41 | 2 | Hiếm, ít mẫu |

**Thông tin trả lời vấn đề 3:** cho thấy mô hình tốt ở lớp lớn, yếu ở lớp hiếm; BN/regularization giúp lớp hiếm nhiều nhất; có bằng chứng để chọn macro-F1 thay vì accuracy.

---

## 10. Calibration và robustness — trả lời "mô hình có đáng tin không"

| Cấu hình | ECE | Brier | Corruption TB | Params | Latency |
|---|---:|---:|---:|---:|---:|
| CNN-A | 0,041 | 0,080 | 72,1 | 2,20M | 4,7 ms |
| **CNN-B** | **0,022** | **0,050** | **86,4** | 2,20M | 4,8 ms |
| CNN-C | 0,028 | 0,060 | 81,2 | 0,40M | 2,1 ms |
| GAP | 0,024 | 0,052 | 84,1 | 0,42M | 3,9 ms |

**Cách đọc:** CNN-B không chỉ accuracy cao hơn mà còn calibration tốt hơn (ECE 0,022 vs 0,041) và robustness cao hơn (86,4 vs 72,1). Đây là bằng chứng cơ chế, không chỉ điểm.

---

## 11. Decision card — bảng tổng hợp để trả lời "giữ hay bỏ"

| Yếu tố | Kết quả giả định | Δ [CI] | Cơ chế | Kết luận |
|---|---|---|---|---|
| BN | on > off | +1,6 [1,3; 1,9] | Ổn định activation, giảm gap 4,2 → 1,7 | **Giữ** |
| Dropout | 0,3 > 0 | +0,6 [0,3; 0,9] | Giảm overfit; 0,5 underfit | **Giữ** |
| WD | 1e-4 vs 0 | +0,2 [0,0; 0,5] | Co weight nhẹ | **Chưa kết luận**; giữ 1e-4 vì không hại |
| Aug | clean tương đương | +0,1 [−0,1; 0,3] | Tăng bất biến | **Giữ nếu cần robustness**; bỏ nếu chỉ clean |
| Depth | 3 > 4 | +0,7 [0,4; 1,0] | 4 khối overfit | **Giữ 3 khối** |
| GAP | off > on | +1,0 [0,7; 1,3] | GAP mất chi tiết không gian | **Không dùng GAP** (nếu CI chứa 0 thì đổi) |
| Width | 32/64/128 đủ | W3 0,0 [−0,2; 0,2] | Diminishing return | **Giữ 32/64/128** |
| LR | 1e-3 | 3e-3 phân kỳ | Ổn định tối ưu | **Giữ 1e-3** |
| Class weight | macro-F1 tương đương | +0,1 [−0,1; 0,3] | Tăng recall lớp hiếm +2,1 | **Không mặc định**; dùng nếu ưu tiên recall |
| Input | 64 | 96: +0,1 [−0,1; 0,3], chậm 1,8× | Chi tiết vs chi phí | **Giữ 64** |
| Batch | 128 | 64/256 tương đương | Throughput | **Giữ 128** |

---

## 12. Khung "vì sao CNN-B" đã điền (giả định)

> Trên OOF 31.371 ảnh dev (3 fold × 3 seed), **CNN-B** đạt macro-F1/43 = **97,1** [96,8; 97,4], accuracy = **98,2%**. CNN-B được chọn vì:
>
> 1. **So với CNN-A:** hơn **1,9 điểm** [1,5; 2,3]. Cơ chế: BN ổn định tối ưu, giảm gap train−val từ **4,2 → 1,7**; dropout + WD giảm overfit.
> 2. **So với CNN-C / 4 khối / GAP:** hơn **1,1 điểm** [0,7; 1,5] so với CNN-C, **0,9 điểm** [0,5; 1,3] so với GAP, **0,7 điểm** [0,4; 1,0] so với 4 khối. Cơ chế: 4 khối overfit (train 99,3 / val 97,6); GAP mất chi tiết không gian (96,3).
> 3. **Về regularization:** BN **+1,6** [1,3; 1,9]; dropout **+0,6** [0,3; 0,9]; WD **+0,2** [0,0; 0,5] → **chưa kết luận**, giữ 1e-4 vì không hại; aug clean **+0,1** [−0,1; 0,3] → **tương đương**, nhưng corruption **+8,2** [7,1; 9,3] → giữ cho demo.
> 4. **Về đánh đổi:** 2,20M params, 4,8 ms/ảnh; W3 8,80M params không hơn; GAP 0,42M nhưng kém 1,0 điểm.
> 5. **Xác nhận lockbox:** **96,8** [96,1; 97,4], cùng thứ hạng OOF; official test 97,9 raw / 97,8 clean chỉ là phụ.
>
> Theo luật quyết định (Δ ≥ 0,3 và CI > 0), kết luận: **chọn CNN-B**. Aug được giữ vì robustness, không phải vì clean accuracy; WD giữ vì không hại; class weight không mặc định.

> **Nếu kết quả thật khác:** nếu GAP tương đương (CI chứa 0 và trong ±0,3), đổi sang GAP; nếu CNN-A thắng, chọn A; nếu aug không tăng cả clean lẫn corruption, bỏ aug. Kịch bản này chỉ là một khả năng.

---

## 13. Bảng đối chiếu: mỗi vấn đề cần thông tin gì

| Vấn đề | Thông tin phải có | Nằm ở mục | Câu trả lời mẫu |
|---|---|---|---|
| 1. Bài test đủ để chốt mô hình | Factorial 2⁴; kiến trúc; hyperparam; OOF; lockbox; CI; decision card | 3–7, 11–12 | "CNN-B hơn A 1,9 điểm [1,5; 2,3], hơn C 1,1, hơn GAP 0,9; lockbox xác nhận 96,8 [96,1; 97,4]." |
| 2. Còn leak không | Split manifest; audit 131 ảnh; so sánh random/group/lockbox; official raw/clean | 1 | "Random cũ 99,95%; lockbox sạch 96,8%; 0 track/sign-group overlap; official test chỉ là phụ." |
| 3. Giải thích cơ chế | Learning curve; per-class; Grad-CAM; calibration; corruption; error taxonomy | 6, 9, 10 | "BN giảm gap 4,2 → 1,7; lớp hiếm 0/19/37 tăng 8–10 điểm F1; ECE 0,041 → 0,022; corruption 72,1 → 86,4." |
| 4. Cấu hình nào cần | Main effect BN/DO/WD/aug; aug profile; depth/GAP/width; LR/input/class weight | 3–6, 11 | "BN cần; dropout cần; WD chưa kết luận; aug clean-neutral nhưng +8,2 robustness; 4 khối/GAP/width lớn không cần." |

---

## 14. Danh sách bảng/hình sẽ có trong báo cáo

### Bảng

1. Split manifest + audit (mục 1).
2. Baseline (mục 2).
3. Factorial 2⁴ đầy đủ (mục 3.1).
4. Main effect + interaction (mục 3.2–3.3).
5. Kiến trúc/Pareto (mục 4).
6. Hyperparameter (mục 5).
7. Augmentation clean + corruption (mục 6).
8. OOF + lockbox + official test (mục 7).
9. Variance decomposition (mục 8).
10. Per-class + confusion pairs (mục 9).
11. Calibration/robustness/latency (mục 10).
12. Decision cards (mục 11).

### Hình

1. Sơ đồ split 3 tầng.
2. Learning curves A/B/C.
3. Interaction plot BN×aug, DO×WD.
4. Pareto params vs macro-F1.
5. Per-class Δ heatmap.
6. Grad-CAM grid.
7. Reliability diagram.
8. Corruption curves 8 panel.
9. Confusion matrix chuẩn hóa hàng.
10. Error taxonomy.

---

## 15. Nhắc lại lần cuối

- File này **không phải kết quả**. Mọi số là giả định.
- Khi chạy thật, thay bằng `outputs/results/v2/`.
- Không chép số giả định vào `BAO_CAO.md`, slide, hay README.
- Nếu kết quả thật khác kịch bản này, **không sửa số**; dùng luật quyết định và báo cáo đúng.
