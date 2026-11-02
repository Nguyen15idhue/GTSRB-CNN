# KẾT QUẢ TRIỂN KHAI V2 — NHẬT KÝ CÓ BẰNG CHỨNG

> Mỗi mục chỉ ghi số **sinh ra từ artifact thật**. Đường dẫn file là bằng chứng.
> Số liệu tổng hợp cuối nằm ở `outputs/results/v2/summary.csv` và
> `outputs/results/v2/lockbox_result.json`.

## Phase 0 — Giao thức, split, audit, test

### 0.1. Pre-registration

- Ngày chốt: 02/11/2026.
- Commit: `18a785d` (`docs: chốt pre-registration thực nghiệm v2`).
- File: `docs/kehoach/1/PRE_REGISTRATION_V2.md` (8 so sánh chính, luật quyết định,
  luật dừng, quy tắc freeze/lockbox).

### 0.2. Split v2

Nguồn: `outputs/splits/v2/manifest.json` (SHA-256 file: xem `manifest.sha256`).

| Tập | Ảnh | Track | Sign-group | Seed |
|---|---:|---:|---:|---:|
| Pool | 39.201 | 1.307 | 1.183 | – |
| Lockbox | 8.302 | 277 | 238 | 2024 |
| Dev | 30.899 | 1.030 | 945 | – |
| Screening train | 24.719 | 824 | – | 42 |
| Screening val | 6.180 | 206 | – | 42 |
| Fold 0 train/val | 19.949 / 10.950 | – | – | 42 |
| Fold 1 train/val | 20.729 / 10.170 | – | – | 42 |
| Fold 2 train/val | 21.120 / 9.779 | – | – | 42 |

- Đơn vị nhóm: **sign-group** = track cùng một biển vật lý, gộp khi có cặp ảnh
  cùng nhãn khác track với MAD < 5 trên ảnh 64×64. Đã gộp **39.603 cặp**.
- Vì vậy số ảnh lockbox (8.302) và dev (30.899) khác bảng dự kiến trong kế hoạch
  (7.830 / 31.371); **manifest là nguồn chính thức**, không chép số dự kiến.
- mean/std tính riêng trên train của từng split, lưu trong `manifest.mean_std`
  (ví dụ screening: mean = 0,369832/0,339117/0,346847; std = 0,276523/0,267793/0,275659).

### 0.3. Audit C1–C12

Nguồn: `outputs/results/v2/audit/audit_report.json` + `checks.csv`.

| Mã | Kết quả | Đạt |
|---|---|---:|
| C1 track overlap lockbox/dev | 0 | ✓ |
| C2 sign-group overlap | 0 | ✓ |
| C3 lớp ở lockbox | 43 | ✓ |
| C4 lớp ở dev | 43 | ✓ |
| C5 lớp ở screening val | 43 | ✓ |
| C6 lớp ở 3 fold | 43/43/43 | ✓ |
| C7 overlap screening | 0 | ✓ |
| C8 overlap track 3 fold | 0/0/0 | ✓ |
| C9 test trong train/lockbox | 0 | ✓ |
| C10 lockbox gần trùng train (MAD < 5) | 0 | ✓ |
| C11 hash trùng trong pool | 0 | ✓ |
| C12 track/lớp nhỏ nhất ở dev | 6 | ✓ |
| HASH file split khớp manifest | khớp 6/6 | ✓ |
| MEANSTD khớp lại từ cache | 0,0e+00 | ✓ |
| COUNT manifest khớp file | khớp | ✓ |

Tổng: **15/15 PASS**. Ghi chú: audit dùng SHA-256 nội dung file thay MD5
(MD5 trùng ⇒ SHA-256 trùng), và chỉ gộp nhóm khi MAD < 5 nên C10 = 0 theo cấu trúc.

### 0.4. Test

- `pytest -q tests/test_split_v2.py` → **10 passed in 11,80 s**.
- Log: `outputs/results/v2/log_pytest_04.txt`.
- Bao gồm: forward A/B/C `(2,3,64,64)→(2,43)`, macro-F1/43 khớp sklearn
  (sai khác < 1e-9), đọc manifest/mean-std, `num_workers=2` trên Windows,
  3 fold phủ đúng toàn bộ dev một lần.

## Phase 1 — Baseline + factorial 2⁴

### 1.1. Baseline B1 (HOG + LinearSVC)

Nguồn: `outputs/results/v2/baseline.json`, log `outputs/results/v2/log_b1.txt`.

| Mục | Giá trị |
|---|---:|
| Train / val | 24.719 / 6.180 (screening) |
| HOG feature dim | 1.764 |
| C grid | 0,1; 1; 10 |
| **C chọn (theo val macro-F1)** | **0,1** |
| Val accuracy | 0,8727 |
| **Val macro-F1/43** | **0,8600** |
| Params (coef + intercept) | 75.895 |
| Thời gian trích HOG | 40,72 s |
| Thời gian fit (C=0,1) | 452,74 s |
| Latency predict-only | 0,0074 ms/ảnh |

- B0 majority (mốc dưới, cùng val): accuracy 0,0291 · macro-F1/43 0,0013.
- Ghi chú: liblinear báo `ConvergenceWarning` ở cả 3 mức C; đã ghi nhận, **không**
  tăng `max_iter` sau khi xem kết quả (tránh tune ngoài đăng ký).
- Không refit trên dev, không đánh giá trên lockbox.

### 1.2. Factorial 2⁴ (F00–F15 × 3 seed)

Nguồn: `outputs/results/v2/summary.csv`, `factorial_effects.json`,
`tables/factorial.md`, `tables/main_effects.md`, `phase12_artifacts.json`.

- **48/48 run** đủ artifact (config, env, metrics đúng schema, best.pt, curves, meta);
  `phase12_artifacts.json.all_complete = true`.
- Mapping: **F00 = CNN-A (E1)**, **F07 = CNN-B không aug (E2)**, **F15 = CNN-B + aug (E3)** —
  `alias_mapping.json` khớp cả 3 (`all_match = true`).
- Bảng đầy đủ 16 dòng: `outputs/results/v2/tables/factorial.md`.

**Main effect (điểm %, CI cluster bootstrap theo track):**

| Yếu tố | Δ (điểm %) | 95% CI | Diễn giải |
|---|---:|---|---|
| BN | **+3,724** | [+1,778; +5,016] | Tốt hơn rõ |
| Dropout 0,3 | **+5,963** | [+2,921; +7,264] | Tốt hơn rõ |
| Weight decay 1e-4 | +0,058 | [−0,526; +0,568] | Không đủ bằng chứng |
| Augmentation | +0,529 | [−1,216; +2,330] | Không đủ bằng chứng (CI rộng) |
| BN × Aug | −0,279 | [−1,058; +0,551] | Tương tác không rõ |
| Dropout × WD | −0,121 | [−0,686; +0,548] | Tương tác không rõ |

**Phát hiện quan trọng (không được bỏ qua):** aug **không** có tác dụng đồng nhất.
Cụ thể trên cùng 3 seed:

- F08 − F00 = 86,672 − 84,109 = **+2,563 điểm** (aug khi chưa có BN/DO/WD).
- F15 − F07 = 93,946 − 96,501 = **−2,555 điểm** (aug khi đã bật BN+DO+WD).
- Vì vậy không được nói "aug giúp X điểm" chung chung; phải nói aug giúp khi
  thiếu regularization và làm giảm clean accuracy khi đủ regularization.
- Kiểm tra corruption ở bước 2.3 sẽ quyết định giữ aug hay không.

**Cửa kiểm tra F15:** F15 trung bình 93,946 so với mốc cũ 97,061 = **−3,115 điểm**.
Đã kiểm tra lại: audit split 15/15 PASS, mean/std khớp 0,0e+00, không có lỗi split;
khác biệt do (a) screening chỉ train 24.719 ảnh (cũ 31.360), (b) split sign-group
khó hơn, (c) aug làm giảm clean accuracy ở ngân sách 40 epoch. Ghi nhận, **không đổi
luật**, kết luận cuối lấy từ OOF/final (`gate_check_f15.json`).

## Phase 2 — Kiến trúc, hyperparam, augmentation

### 2.1. Kiến trúc (D1, D2, D3, W1, W2 × 3 seed)

Nguồn: `outputs/results/v2/tables/architecture.md`, `summary.csv`.
Mốc so sánh D4 = F15 (93,946).

| ID | Thay đổi so với D4 | Acc (%) | Macro-F1 (%) | Δ vs D4 (điểm %) [CI] | Params |
|---|---|---:|---:|---|---:|
| D1 | thêm GAP, 3 khối | 74,234 ± 0,592 | 59,188 ± 0,764 | **−34,758** [−36,181; −22,916] | 99.019 |
| D2 | 4 khối, flatten | 98,004 ± 0,297 | **97,158 ± 0,610** | **+3,212** [+0,036; +4,440] | 1.448.779 |
| D3 | 4 khối + GAP + DO 0,4 (E4 cũ) | 96,575 ± 0,580 | 93,105 ± 0,906 | −0,841 [−2,724; +0,757] | 399.947 |
| W1 | width 16/32/64 | 94,876 ± 0,916 | 90,686 ± 1,425 | −3,260 [−5,250; −0,735] | 1.083.579 |
| W2 | width 48/96/192 | 97,104 ± 0,368 | 95,215 ± 1,230 | +1,268 [−0,585; +2,499] | 3.366.363 |

**Kết luận:**

- **GAP trên 3 khối (D1) thất bại rõ**: chỉ 99k params, train acc 84,6%, val 74,2%
  sau 40 epoch → **underfit**, không phải "GAP mất chi tiết" đơn thuần. Không chọn.
- **4 khối + flatten (D2) tốt hơn 3 khối (D4) 3,212 điểm, CI không chứa 0** →
  bác bỏ giả thuyết H4 "3 khối ≥ 4 khối". D2 cũng ít params hơn D4 (1,45M so với 2,20M).
- **D3 (E4 cũ) tương đương D4** trong CI → không dùng D3 để kết luận cơ chế
  (D3 đổi cả depth/GAP/dropout 0,4); D3 chỉ là legacy mapping.
- **Width nhỏ (W1) kém rõ**; width lớn (W2) +1,268 nhưng CI chứa 0 → **chưa kết luận**,
  và params tăng 1,5×.
- Pareto: D2 nằm trên frontier (F1 cao nhất, params thấp hơn D4); xem
  `outputs/figures/v2/03_pareto_params_f1.png` (bước 4.2).

### 2.2. Hyperparam (L1, CW1, I2 × 3 seed)

Nguồn: `outputs/results/v2/tables/hyperparam.md`. Mốc F15 = 93,946.

| ID | Thay đổi | Acc (%) | Macro-F1 (%) | Δ vs F15 (điểm %) [CI] | Params |
|---|---|---:|---:|---|---:|
| L1 | LR 1e-4 | 91,888 ± 1,282 | 86,807 ± 1,615 | **−7,139** [−9,372; −3,353] | 2.201.931 |
| CW1 | class weight inv_sqrt | 95,960 ± 0,606 | 92,943 ± 0,475 | −1,004 [−2,195; +0,261] | 2.201.931 |
| I2 | input 96 | 94,213 ± 0,624 | 90,499 ± 1,429 | **−3,448** [−5,463; −0,984] | 4.823.371 |

**Kết luận:**

- **LR 1e-3 > 1e-4 rõ rệt** (−7,139 nếu dùng 1e-4): LR nhỏ không hội tụ trong 40 epoch.
  Đây là so sánh cách 10 lần, đủ 40 epoch, không so 10 epoch.
- **Class weight không cải thiện** (Δ −1,004; CI chứa 0) → bỏ; đây là exploratory,
  chưa hiệu chỉnh.
- **Input 96 kém hơn 64** (−3,448) và params tăng 2,2× → giữ 64.
- Cache 96 dựng riêng ở `data/cache_96` (không ghi đè cache 64).

### 2.3. Augmentation thành phần + corruption

Nguồn: `outputs/results/v2/tables/augmentation.md`, `corruption.csv`, `log_corruption.txt`.
G0 = F07 (không aug, 96,501); G5 = F15 (all, 93,946).

| ID | Profile | Acc (%) | Macro-F1 (%) | Δ vs G0 (điểm %) [CI] |
|---|---|---:|---:|---|
| G1 | affine | 95,572 ± 0,190 | 92,251 ± 0,595 | **−4,250** [−6,972; −0,084] |
| G2 | color | 97,767 ± 0,242 | **97,417 ± 0,340** | +0,917 [−0,141; +1,842] |
| G3 | perspective | 96,375 ± 1,116 | 95,264 ± 1,988 | −1,237 [−2,706; +0,530] |

**Corruption (accuracy trung bình 3 mức, screening val, seed 42):**

| Config | Blur | Noise | Bright | Contrast | JPEG | Occlusion | TB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F07 (G0) | 95,05 | 86,27 | 97,13 | 93,49 | 94,00 | 23,11 | 81,51 |
| F15 (G5) | 93,73 | 81,00 | 96,19 | 92,77 | 91,22 | 26,43 | 80,22 |
| G1 affine | 91,35 | 75,84 | 94,75 | 86,81 | 88,38 | 24,24 | 76,90 |
| G2 color | 95,73 | 85,86 | 97,56 | 96,06 | 95,07 | 21,56 | **81,97** |
| G3 perspective | 94,06 | 84,85 | 96,78 | 93,65 | 93,49 | 21,44 | 80,71 |
| D2 (4 khối + all) | 93,88 | 83,59 | 97,10 | 92,92 | 92,78 | 17,63 | 79,65 |

**Kết luận:**

- **Aug "all" không có lợi**: clean −2,555 điểm so với F07 (F15) và corruption TB
  thấp hơn (80,22 so với 81,51).
- **Affine là thành phần gây hại** (−4,250 clean, −4,61 corruption TB so với G0).
- **Color-only tốt nhất trong nhóm G**: clean +0,917 (CI chứa 0 → chưa kết luận
  so với không aug), corruption TB +0,46 — dưới ngưỡng +0,5 nên **không đủ bằng
  chứng để giữ aug vì robustness**.
- **Occlusion làm sập mọi mô hình** (17,6–26,4%): đây là hạn chế thật, cần báo cáo,
  không tô hồng.
- Corruption dùng **6 loại × 3 mức** theo bảng 6.5 của kế hoạch (không phải 8 loại).

### 2.4. Chọn top 2 + control (trước OOF)

Nguồn: `outputs/results/v2/candidates.json`, `log_select.txt`.

- Bảng xếp hạng đầy đủ 3 seed: G2 97,417; D2 97,158; F07 96,501; W2 95,215; …
- Nhóm trong 0,3 điểm so với tốt nhất: **{G2, D2}** (0,259 điểm).
- **Top 2: X1 = G2 (color)**, **X2 = D2 (4 khối, flatten, all)**.
- **Control: X3 = F07** (không aug, 3 khối) — neo để đo tác dụng regularization.
- Lý do ghi trước khi chạy OOF; `forced_second_candidate = false`.
- Đã sinh `configs/v2/x1.yaml`–`x3.yaml` (split = fold) và điền X1–X3 vào
  `docs/kehoach/1/MA_TRAN_RUN.csv`.

## Phase 3 — Xác nhận OOF, freeze, lockbox

### 3.1. OOF 3 fold × 3 seed

Nguồn: `outputs/results/v2/oof_summary.json`, `oof_predictions.csv`,
`oof_comparisons.csv`, `variance.json`, `log_oof.txt`.

- 27/27 run fold hoàn tất (sau khi sửa bug fold override — xem ghi chú cuối mục).
- `oof_predictions.csv`: **278.091 dòng = 30.899 ảnh × 3 seed × 3 config**; mỗi
  cặp (config, seed) phủ đúng 30.899 ảnh dev đúng một lần, không trùng.
- Macro-F1/43 trên OOF (mean ± sd qua 3 seed, CI cluster bootstrap B=2.000):

| Config | Nguồn | OOF macro-F1 (%) | CI 95% (trung bình 3 seed) |
|---|---|---:|---|
| X1 | G2 (3 khối + color) | 96,297 ± 0,296 | [94,628; 97,020] |
| **X2** | **D2 (4 khối + all)** | **97,040 ± 0,187** | **[95,051; 97,942]** |
| X3 | F07 (control, không aug) | 95,059 ± 0,568 | [93,113; 95,930] |

**So sánh cặp (paired cluster bootstrap):**

| So sánh | Δ (điểm %) | 95% CI | Kết luận |
|---|---:|---|---|
| X1 − X2 | −0,743 | [−1,590; +0,274] | Chưa kết luận (CI chứa 0, vượt ±0,3) |
| X1 − X3 | **+1,238** | [+0,586; +2,071] | X1 tốt hơn control |
| X2 − X3 | **+1,981** | [+1,053; +2,961] | X2 tốt hơn control |

**Phân rã phương sai (seed × fold):**

| Thành phần | Giá trị | Tỷ lệ |
|---|---:|---:|
| var_seed | 4,42e−06 | 0,9% |
| var_fold | 4,20e−04 | **91,0%** |
| var_resid | 3,82e−05 | 8,3% |

→ **Phương sai do fold chiếm 91%**, seed chỉ 0,9%. Kết luận: báo cáo theo fold là
bắt buộc; multi-seed một mình không đủ (khớp cảnh báo mục 7.4 của kế hoạch).

**Bug đã sửa (ghi lại để tái lập):** lần chạy OOF đầu, `run_training_v2` không ghi
`fold` override vào `cfg` nên cả 3 fold dùng nhầm `fold_0`; `oof_predictions`
phát hiện ảnh trùng và dừng. Đã sửa (`cfg["data"]["fold"] = used_fold`), xoá 18
run sai, chạy lại fold 1–2. Không có run sai nào được dùng trong kết quả.

### 3.2. Chọn winner và freeze

- Theo luật 4.5: X2 có OOF cao nhất (97,040). X1 − X2 = −0,743 với CI chứa 0 →
  **chưa kết luận**; tie-break (1) ít params hơn: X2 1.448.779 < X1 2.201.931
  → **chọn X2 (D2: 4 khối 32/64/128/256, flatten, BN, dropout 0,3, WD 1e-4, aug all)**.
- **Epoch cố định = 26** = median best epoch của 9 run OOF X2 (15, 19, 21, 22, 26, 31, 31, 32, 40).
- Freeze: `outputs/results/v2/FREEZE.json`
  - `final_config_sha256 = 2678af82b7fc85fb…`
  - `split_sha256 = 640a1913e011b2ed…` (dev_all)
  - `git_commit` ghi tại thời điểm freeze.
- `configs/v2/final.yaml`: split `dev_all`, 30.899 ảnh, max_epochs 26, không early stop.
- **Lockbox chưa mở** (`lockbox_opened: false`).
- Winner card: `outputs/results/v2/winner_card.json`.

### 3.3. Final trên dev + lockbox một lần + official test

**Final:** 3 run `v2_final_s42/123/2024` trên **30.899 ảnh dev**, epoch cố định 26
(median OOF), không early stop. `final_ckpt_check.json`: cả 3 checkpoint load được,
forward `(2,3,64,64)→(2,43)`, `config_sha256` và `split_sha256` khớp `FREEZE.json`,
params 1.448.779.

**Lockbox (8.302 ảnh / 277 track, SHA `fd70ea2b…`), mở sau freeze:**

| Seed | Accuracy | Macro-F1/43 |
|---|---:|---:|
| 42 | 0,987593 | 0,968770 |
| 123 | 0,989400 | 0,979706 |
| 2024 | 0,989280 | 0,979468 |
| **Mean ± sd** | **0,988758 ± 0,001010** | **0,975981 ± 0,006246** |

- Top-3: 0,998474 ± 0,000387.
- ECE 15 bin: 0,005328 ± 0,000970; Brier: 0,017862 ± 0,002022.
- File: `outputs/results/v2/lockbox_result.json`; log mở: `lockbox_access.log`.

> **Sự cố mở lockbox (ghi trung thực):** lần chạy 04:10 đã tính đủ 3 seed nhưng
> treo ở bước ghi kết quả do bug `track_map()` gọi lại 8.302 lần; **không file nào
> được ghi và không số nào được dùng để chỉnh gì**. Config/model/hash giữ nguyên;
> lần chạy 04:40 ghi kết quả, cho **đúng cùng số** (deterministic). Đã ghi cả hai
> lần vào `lockbox_access.log`.

**Official test (benchmark phụ, không tune):**

| Tập | n | Accuracy | Macro-F1/43 |
|---|---:|---:|---:|
| Raw | 12.630 | 0,986989 ± 0,001625 | 0,980398 ± 0,004516 |
| Clean (bỏ ảnh gần trùng) | 11.879 | 0,988692 ± 0,002359 | 0,982499 ± 0,005314 |

- Audit lại test ↔ train (MAD < 5, cùng lớp, thumbnail + kiểm chứng full 64×64):
  **751 ảnh test** có ít nhất một ảnh train gần trùng (24.554 cặp), clean = 11.879.
- **Khác số cũ 131 ảnh/12.499** vì phương pháp sinh ứng viên khác (audit cũ chỉ
  xét 17.214 cặp pHash; audit này quét đầy đủ trong từng lớp). Dùng số đo được
  của repo này; báo cả raw để so sánh.
- Clean **cao hơn** raw (98,869 so với 98,699) → ảnh gần trùng **không** thổi phồng
  kết quả; nhưng official test vẫn **không track-disjoint**, chỉ là benchmark phụ.

## Phase 4 — Thống kê, hình, decision cards

### 4.1. Tổng hợp và kiểm tra số

- `summary.csv`: **111 run** (81 screening + 27 fold + 3 dev_all), 111/111
  `valid_for_conclusion=1`, không run nào thiếu `config_sha256`/`split_sha256`.
- `verify_41.json`: macro-F1 OOF x2/seed42 tính lại = **0,969321** = số lưu
  (sai khác 0,0); macro-F1 F15/s42 tính lại từ `predictions_val.csv` lệch
  **3,1e−07** so với summary → đạt.
- `variance.json`: var_fold = 4,20e−04 (91,0%), var_seed = 4,42e−06 (0,9%),
  var_resid = 3,82e−05 (8,3%).

**Tám so sánh chính (Holm):**

| Yếu tố | Δ (điểm %) | 95% CI | p Holm | Kết luận |
|---|---:|---|---:|---|
| BN | +3,724 | [+1,767; +5,005] | 0,000 | Tốt hơn |
| Dropout | +5,963 | [+2,873; +7,147] | 0,000 | Tốt hơn |
| Weight decay | +0,058 | [−0,506; +0,551] | 0,412 | Chưa kết luận |
| Augmentation | +0,529 | [−1,204; +2,303] | 0,708 | Chưa kết luận |
| GAP (D1 vs D4) | −34,758 | [−36,108; −22,444] | 0,000 | Kém hơn |
| Depth (D2 vs D4) | +3,212 | [+0,011; +4,513] | 0,373 | CI ủng hộ; sau Holm chưa đủ mạnh |
| Width (W2 vs D4) | +1,268 | [−0,591; +2,541] | 0,708 | Chưa kết luận |
| LR (1e-4 vs 1e-3) | −7,139 | [−9,353; −3,366] | 0,000 | Kém hơn |

> Depth có CI không chứa 0 nhưng p Holm 0,373: báo **"có bằng chứng vừa, chưa đủ
> sau hiệu chỉnh"**, không tuyên bố "depth chắc chắn tốt hơn". Quyết định chọn D2
> dựa trên OOF (Δ +1,981 so với control, CI [+1,053; +2,961]) và tie-break params.

### 4.2. Hình cơ chế

Tất cả trong `outputs/figures/v2/`:

| Hình | Nội dung | Số chính |
|---|---|---|
| `01_learning_curves.png` | F00/F07/F15/D2 theo epoch | gap train−val: F00 9,87; F07 2,60; F15 1,08 điểm |
| `02_factorial_effects.png` | Main effect + interaction | BN +3,72; DO +5,96; WD +0,06; Aug +0,53; BN×Aug −0,28; DO×WD −0,12 |
| `03_pareto_params_f1.png` | Params vs macro-F1 | D2 trên frontier (1,45M params, F1 97,16) |
| `04_per_class_delta.png` | ΔF1 43 lớp so với F15 | tập trung lớp hiếm/khó |
| `05_reliability.png` | Reliability 15 bin | ECE: F00 0,0758; F07 0,0154; F15 0,0120; G2 0,0101; D2 0,0084 |
| `06_corruption_curves.png` | 6 loại × 3 mức | TB: F07 81,51; G2 81,97; F15 80,22; D2 79,65 |
| `07_confusion_*.png` | Confusion chuẩn hóa hàng | F15, D2 trên screening val |
| `08_gradcam_grid.png` | Grad-CAM 12 ảnh, B vs C | 4 lớp × 3 trường hợp |
| `09_error_samples.png` | 12 mẫu lỗi | 190 lỗi: 102 nhãn khó, 64 mờ, 23 nhầm cặp, 1 chói |
| `10_confusion_lockbox.png` | Confusion lockbox (TB 3 seed) | 43×43, chuẩn hóa hàng |
| `11_per_class_f1_lockbox.png` | F1 từng lớp trên lockbox | thấp nhất: lớp 19 (0,760), 29 (0,776), 30 (0,925) |

**Cơ chế rút ra:**

- BN + dropout giảm gap mạnh (F00 gap 9,87 → F07 2,60 điểm) và tăng F1.
- Aug "all" làm train khó hơn nhưng **không** tăng val F1; affine là thành phần hại.
- GAP 3 khối underfit (train acc 84,6%) — không phải "mất chi tiết" mà là **thiếu
  dung lượng/tối ưu**.
- Calibration tốt lên theo regularization: ECE F00 0,0758 → D2 0,0084.
- Occlusion là điểm yếu chung (17,6–26,4% accuracy), cần nêu trong hạn chế.
- Lớp 19 và 29 là hai lớp yếu nhất trên lockbox (F1 0,760 / 0,776), support 30 / 60.

### 4.3. Decision cards (10 yếu tố)

Nguồn: `outputs/results/v2/decision_cards.csv`.

| Yếu tố | Mức đã thử | Δ [CI] (điểm %) | Kết luận | Đánh đổi |
|---|---|---|---:|---|
| BN | off/on | +3,72 [+1,77; +5,00] | **Giữ** | không thêm chi phí suy luận |
| Dropout | 0 / 0,3 | +5,96 [+2,87; +7,15] | **Giữ** | không thêm chi phí suy luận |
| Weight decay | 0 / 1e-4 | +0,06 [−0,51; +0,55] | Không đủ bằng chứng | giữ 1e-4 vì không hại |
| Augmentation | off / all | +0,53 [−1,20; +2,30] | Không đủ bằng chứng (all hại clean) | bỏ all; color trung tính |
| Depth | 3 / 4 khối | +3,21 [+0,01; +4,51] | Giữ 4 khối (CI ủng hộ; Holm chưa mạnh) | ít params hơn D4 |
| GAP | off/on (3 khối) | −34,76 [−36,11; −22,44] | **Bỏ** | GAP 3 khối underfit |
| Width | 32/64/128 vs 48/96/192 | +1,27 [−0,59; +2,54] | Không đủ bằng chứng | W2 tăng 1,5× params |
| LR | 1e-4 vs 1e-3 | −7,14 [−9,35; −3,37] | **Giữ 1e-3** | LR nhỏ cần nhiều epoch hơn |
| Class weight | none / inv_sqrt | −1,00 [−2,11; +0,26] | Bỏ | exploratory, chưa hiệu chỉnh |
| Input size | 64 / 96 | −3,45 [−5,52; −0,93] | **Giữ 64** | 96 tăng 2,2× params, chậm hơn |

## Phase 5 — Nghiệm thu cuối

Nguồn: `outputs/results/v2/ACCEPTANCE.json`, `acceptance.md`, `reproducibility.json`.

**12/12 điều kiện đạt:**

| # | Điều kiện | Giá trị |
|---|---|---|
| 1 | Lockbox 0 track/sign-group overlap; one-shot | 0 / 0; 8.302 ảnh; log có |
| 2 | Mọi run kết luận dùng split sign-group | protocol `v2_clean`, không có random |
| 3 | Δ + CI + luật quyết định | 8 so sánh chính + Holm |
| 4 | Factorial 2⁴ đủ 16 config | 48 run |
| 5 | Tách GAP/depth/width; D3 legacy | D1/D2/W1/W2 + F15 |
| 6 | Aug bật/tắt + corruption | 108 dòng (6×3) |
| 7 | OOF 3 fold × 3 seed | 278.091 dòng |
| 8 | Decision card 10 yếu tố | 10 card |
| 9 | Không dùng số random cũ | legacy tách riêng |
| 10 | Không tune lockbox/test | freeze trước, log mở |
| 11 | Repo sạch | không data/*.npy; chỉ checkpoint demo được phép |
| 12 | Tái lập cùng seed | sai khác **0,0** trên 4 metric |

**Rủi ro còn lại:**

- Lockbox 8.302 ảnh / 277 track → CI rộng hơn official test; không so trực tiếp.
- Official test vẫn không track-disjoint; clean 11.879 theo audit riêng (751 ảnh).
- Occlusion là điểm yếu lớn (17,6–26,4% accuracy).
- Screening 24.719 ảnh < final 30.899 → kết luận screening chỉ để chọn.
- Một số so sánh (WD, width, aug) **chưa kết luận**, không được trình bày là "tốt".
- Sự cố mở lockbox 2 lần do bug ghi file đã ghi log; config/model/hash không đổi,
  kết quả lần 2 trùng khớp tuyệt đối lần 1.









