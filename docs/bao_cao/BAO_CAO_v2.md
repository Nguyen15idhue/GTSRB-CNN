# BÁO CÁO THỰC NGHIỆM V2 — GTSRB CNN

## Phân loại 43 lớp biển báo giao thông Đức bằng CNN tự xây dựng

| Thông tin | Giá trị |
|---|---|
| Dự án | Bài tập lớn Học sâu 12INF028 |
| Phiên bản | 2.0 — tái thiết kế đánh giá theo kế hoạch `docs/kehoach/1/KE_HOACH_V2_DAY_DU.md` |
| Ngày | 03/11/2026 |
| Người thực hiện | Kỹ sư ML chính kiêm auditor tái lập |
| Phạm vi | 18 bước: split 3 tầng, factorial 2⁴, kiến trúc, hyperparam, augmentation, OOF, lockbox, báo cáo, nghiệm thu |
| Nguồn số | `outputs/results/v2/`, `outputs/splits/v2/`, `outputs/figures/v2/` |
| Kết quả chính | OOF macro-F1 **0,970400**; lockbox **0,975981** [0,950511; 0,979422]; official test raw 0,980398 / clean 0,982499 |
| Nghiệm thu | **12/12** điều kiện (`outputs/results/v2/ACCEPTANCE.json`) |

---

## Tóm tắt

Báo cáo này thay thế các kết luận cũ dựa trên `split_random` — nơi 95,22% ảnh
validation có frame cùng track với train và 1-NN đạt 90,5% mà không học gì.
Phiên bản v2 chia lại 39.201 ảnh thành ba tầng theo **sign-group**: lockbox 8.302
ảnh chỉ mở một lần, dev 30.899 ảnh dùng cho screening và 3-fold OOF, official test
12.630 ảnh chỉ là benchmark phụ.

Tổng cộng **111 run** được chạy thật: 48 run factorial 2⁴, 15 run kiến trúc, 9 run
hyperparam, 9 run augmentation, 27 run OOF, 3 run final. Mô hình thắng là **D2**
(4 khối conv 32/64/128/256 + flatten + BN + dropout 0,3 + WD 1e-4 + aug all,
1.448.779 params), chọn bằng OOF 3 fold × 3 seed và tie-break ít params hơn.

Kết quả độc lập: OOF macro-F1/43 **0,970400 ± 0,001869**; lockbox accuracy
**0,988758 ± 0,001010**, macro-F1 **0,975981 ± 0,006246**, top-3 0,998474, ECE
0,005328. Official test raw **0,986989 / 0,980398**, clean 11.879 ảnh
**0,988692 / 0,982499**.

Ba phát hiện trái kỳ vọng được giữ nguyên: (1) augmentation "all" **không** tăng
clean accuracy, affine là thành phần hại; (2) GAP trên 3 khối **underfit nặng**
(F1 0,59188); (3) 4 khối **tốt hơn** 3 khối (Δ +3,212 điểm, CI [+0,036; +4,440]).
Phương sai do fold chiếm **91,0%**, seed chỉ 0,9% — bằng chứng rằng báo cáo theo
fold là bắt buộc.

---

## Chương 1. Giới thiệu

### 1.1. Bài toán

GTSRB gồm 43 lớp biển báo giao thông Đức, ảnh là các frame video đã crop quanh
biển. Nhiệm vụ: huấn luyện CNN **tự xây dựng** (không pretrained) nhận ảnh
RGB 64×64, trả logits 43 lớp; đánh giá bằng macro-F1/43 vì các lớp hiếm.

### 1.2. Phạm vi

Báo cáo tập trung vào **thiết kế thực nghiệm và bằng chứng**, không phải demo
Streamlit. Các nội dung:

1. Chia lại dữ liệu 3 tầng theo sign-group, audit rò rỉ.
2. Factorial 2⁴ tách BN / dropout / weight decay / augmentation.
3. Tách kiến trúc: depth, GAP, width.
4. Hyperparam tối thiểu: LR, class weight, input size.
5. Augmentation thành phần + corruption robustness.
6. Chọn mô hình bằng OOF 3 fold × 3 seed, CI cluster bootstrap theo track.
7. Freeze, mở lockbox một lần, official test raw/clean.
8. Decision card cho 10 yếu tố, giải thích cơ chế.

### 1.3. Bốn phản biện và cách trả lời

| # | Phản biện | Cách trả lời trong v2 | Bằng chứng |
|---|---|---|---|
| 1 | Bài test chưa đủ để chốt mô hình | Pre-registration, factorial, tách kiến trúc, OOF, lockbox, CI, luật quyết định | mục 4.5–4.8; `main_comparisons.csv` |
| 2 | Số liệu còn leak | Split 3 tầng theo sign-group; lockbox 0 overlap; mean/std theo split; official test chỉ phụ | mục 2.4–2.8; `audit_report.json` |
| 3 | Chỉ báo số, không giải thích | Learning curve, per-class, Grad-CAM, calibration, corruption, error taxonomy | chương 5; `outputs/figures/v2/` |
| 4 | Không rõ cấu hình nào cần | Factorial, aug components, báo cáo cả "không đủ bằng chứng" | mục 4.2–4.5, 4.8; `decision_cards.csv` |

### 1.4. Cảnh báo đọc số cũ

Các run E1–E7 trong `outputs/results/summary.csv` dùng `split_random`; val bị rò
rỉ frame. Số cũ **không** được dùng trong bất kỳ kết luận nào của báo cáo này;
chúng chỉ nằm ở Phụ lục A với nhãn `legacy_leaky`.

---

## Chương 2. Dữ liệu và chia tập v2

### 2.1. Nguồn dữ liệu

- GTSRB: train gốc **39.209** ảnh, test **12.630** ảnh, 43 lớp.
- Train tải từ archive chính thức `GTSRB_Final_Training_Images.zip` (không dùng
  bản rút gọn 26.640 ảnh của torchvision).
- Test tải qua `torchvision.datasets.GTSRB`, nhãn đọc từ `GT-final_test.csv`.
- 8 ảnh train trùng MD5 với test (đều lớp 14, thuộc track `00023`) bị loại khỏi
  tập huấn luyện → **pool 39.201 ảnh**. File trên đĩa vẫn giữ 39.209 để số đếm
  khớp công bố; danh sách loại trừ ở `outputs/results/excluded_train_duplicates.csv`.

![Hình 2.1. Phân bố số ảnh theo lớp (đỏ: lớp dưới 300 ảnh)](outputs/figures/01_class_distribution.png)

*Hình 2.1. Phân bố 43 lớp; lớp ít nhất 210 ảnh (lớp 0), nhiều nhất 2.250 ảnh (lớp 2), tỷ lệ 10,714 lần.*

![Hình 2.2. Lưới 43 ảnh mẫu, mỗi lớp một ảnh](outputs/figures/03_sample_grid.png)

*Hình 2.2. Mỗi lớp một ảnh mẫu; ảnh đã crop theo ROI để không lộ người/xe ở nền.*

### 2.2. Quy mô thật

| Hạng mục | Giá trị |
|---|---:|
| Train trên đĩa | 39.209 |
| Loại do trùng MD5 với test | 8 |
| **Pool dùng chia split** | **39.201** |
| Test chính thức | 12.630 |
| Số lớp | 43 |
| Ảnh/lớp: min – median – max | 210 – 600 – 2.250 |
| Số lớp dưới 300 ảnh | 9 |
| Track (pool) | 1.307 |
| Frame/track: min – median – max | 29 – 30 – 30 |

Nguồn: `outputs/results/data_report.json`, `outputs/splits/v2/manifest.json`.

### 2.3. Track và sign-group

**Track** là một đoạn video ngắn của cùng một biển, thường 30 frame. Tên file
train có dạng `<track>_<frame>.ppm`; track chỉ duy nhất trong một lớp, nên khóa
nhóm là `track_key = "<label>_<track>"`.

Các frame trong cùng track gần như liên tiếp: khoảng cách frame trung vị = 1,
95,22% frame val random có frame train cách ±1 (khi chia random). Vì vậy nếu
chia ngẫu nhiên theo ảnh, val gần như đã "nhìn thấy" train.

**Sign-group** là mức chặt hơn: hai track khác nhau có thể là cùng một biển vật
lý được quay ở hai đoạn khác nhau. Cách dựng:

1. Tính thumbnail xám 16×16 cho từng ảnh.
2. Trong mỗi lớp, so từng cặp ảnh khác track; giữ ứng viên có MAD thumbnail < 8.
3. Kiểm chứng lại trên ảnh 64×64; nếu **MAD < 5** thì nối hai ảnh.
4. Union-find gộp thành sign-group; tên nhóm `<label>_g<k>`.
5. Frame trong cùng track luôn được gộp (track là một biển).

Kết quả: 39.603 cặp được gộp, 1.307 track → **1.183 sign-group**. Không xóa ảnh
nào; chỉ dùng để chia nhóm. Chi tiết: `outputs/results/v2/audit/sign_groups.csv`.

> Rủi ro: nếu hai biển khác nhau nhưng rất giống (cùng thiết kế) bị gộp nhầm,
> split sẽ bảo thủ hơn (test khó hơn), không gây leak. Đây là đánh đổi có chủ đích.

### 2.4. Split 3 tầng

![Hình 2.3. Sơ đồ split 3 tầng theo sign-group](outputs/figures/v2/12_split_v2.png)

*Hình 2.3. Pool 39.201 → lockbox 8.302 / dev 30.899; dev → screening 24.719/6.180 và 3-fold OOF.*

| Tầng | Quy tắc | Ảnh | Track | Sign-group | Seed |
|---|---|---:|---:|---:|---:|
| Lockbox | 20% sign-group mỗi lớp | 8.302 | 277 | 238 | 2024 |
| Dev | phần còn lại | 30.899 | 1.030 | 945 | – |
| Screening train | 80% sign-group của dev | 24.719 | 824 | – | 42 |
| Screening val | 20% sign-group của dev | 6.180 | 206 | – | 42 |
| Fold 0 train / val | 2/3 – 1/3 sign-group của dev | 19.949 / 10.950 | – | – | 42 |
| Fold 1 train / val | – | 20.729 / 10.170 | – | – | 42 |
| Fold 2 train / val | – | 21.120 / 9.779 | – | – | 42 |
| Final train | toàn bộ dev | 30.899 | 1.030 | 945 | – |

Thuật toán (mã giả):

```text
for mỗi lớp c:
    groups_c = sorted(sign-group của lớp c)
    k = min(max(1, round(0.2 * len(groups_c))), len(groups_c) - 1)
    lock_c = rng(seed=2024).choice(groups_c, k, replace=False)
lockbox = union(lock_c); dev = pool - lockbox

for mỗi lớp c trong dev:
    k2 = min(max(1, round(0.2 * len(groups_dev_c))), len(groups_dev_c) - 1)
    val_c = rng(seed=42).choice(groups_dev_c, k2, replace=False)
screening_val = union(val_c); screening_train = dev - screening_val

for mỗi lớp c trong dev:
    sắp groups theo số frame giảm dần
    gán lần lượt vào fold ít frame nhất (round-robin cân bằng)
```

Số ảnh lockbox (8.302) khác bảng dự kiến 7.830 vì đơn vị nhóm là sign-group
(không đều nhau), không phải track. **Manifest là nguồn chính thức**, không chép
số dự kiến vào báo cáo.

### 2.5. Audit C1–C12

Nguồn: `outputs/results/v2/audit/audit_report.json` (**15/15 PASS**).

| Mã | Kiểm tra | Kết quả | Đạt |
|---|---|---|---:|
| C1 | Track overlap lockbox/dev | 0 | ✓ |
| C2 | Sign-group overlap lockbox/dev | 0 | ✓ |
| C3 | Lockbox đủ lớp | 43/43 | ✓ |
| C4 | Dev đủ lớp | 43/43 | ✓ |
| C5 | Screening val đủ lớp | 43/43 | ✓ |
| C6 | Mỗi fold đủ lớp | 43/43/43 | ✓ |
| C7 | Overlap screening train/val | 0 | ✓ |
| C8 | Overlap track từng fold | 0/0/0 | ✓ |
| C9 | Test nằm trong train/lockbox | 0 | ✓ |
| C10 | Lockbox gần trùng train (MAD < 5) | 0 | ✓ |
| C11 | Hash trùng trong pool | 0 | ✓ |
| C12 | Track/lớp nhỏ nhất trong dev | 6 | ✓ |
| HASH | Hash 6 file split khớp manifest | khớp | ✓ |
| MEANSTD | mean/std khớp lại từ cache | 0,0e+00 | ✓ |

Audit dùng SHA-256 nội dung file thay MD5 (MD5 trùng ⇒ SHA-256 trùng), và C10
dùng đúng ngưỡng MAD < 5 như lúc dựng sign-group nên bằng 0 theo cấu trúc.

### 2.6. Official test không track-disjoint

Official test không có track ID. Audit gần trùng test ↔ train (MAD < 5, cùng lớp,
thumbnail 16×16 rồi kiểm chứng full 64×64):

| Mục | Giá trị |
|---|---:|
| Cặp gần trùng | 24.554 |
| Ảnh test có ít nhất một ảnh train gần trùng | **751** (5,95%) |
| Clean | **11.879** |
| Raw | 12.630 |

Số này khác audit cũ (131 ảnh/12.499) vì audit cũ chỉ xét 17.214 cặp pHash; audit
này quét đầy đủ trong từng lớp. Clean **cao hơn** raw (98,869 so với 98,699) nên
không có dấu hiệu thổi phồng. Official test vẫn chỉ là benchmark phụ.

### 2.7. Tiền xử lý và augmentation

- Ảnh gốc kích thước thay đổi → `convert("RGB")` → resize bilinear 64×64 → cache
  `uint8` `(N,64,64,3)`. Cache 96×96 dựng riêng ở `data/cache_96` cho thí nghiệm I2,
  không ghi đè cache 64.
- mean/std tính **trên train của từng split** và lưu trong `manifest.mean_std`;
  không dùng thống kê toàn cục.
- Hàm `preprocess` duy nhất trong `src/data.py` dùng chung cho train/eval/demo.
- Augmentation hợp lệ: affine (xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5),
  color jitter (0,2/0,2/0,2/hue 0,02), perspective (0,1, p=0,3).
- **Cấm** horizontal flip, vertical flip, xoay 90/180; có test tự động chặn.

### 2.8. Vì sao dùng sign-group thay vì random

Đây là câu hỏi trọng tâm. Ba bằng chứng định lượng từ `leak_delta.json`:

| Chỉ số | Val random | Val group/sign-group | Ý nghĩa |
|---|---:|---:|---|
| % frame val có frame train cách ±1 | **95,22%** | 0% | Random gần như trùng frame |
| Track train/val trùng | **1.306 / 1.307** | **0** | Random không tách track |
| 1-NN (16×16 xám, không học) | **90,5%** | 31,0% | Val random bị "giải sẵn" |
| Val accuracy cùng model | 99,936% | 98,473% | Chênh 1,46 điểm |
| Macro-F1/43 cùng model | 99,947% | 88,351%* | *group val cũ thiếu 4 lớp nên bị phạt cấu trúc 9,06 điểm; phần do leak thật ≈ 2,53 điểm |

Kết luận: val random **không đo được khả năng tổng quát**; nó đo mức độ gần nhau
của frame. Vì vậy mọi kết luận v2 dùng split theo sign-group, và lockbox được
dùng như test nội bộ độc lập.

---

## Chương 3. Phương pháp

### 3.1. Baseline B1

- HOG trên ảnh xám 64×64: 9 orientations, cell 8×8, block 2×2, L2-Hys → 1.764 chiều.
- `StandardScaler` fit trên screening train; `LinearSVC` OvR, C ∈ {0,1; 1; 10},
  chọn C theo macro-F1/43 trên **screening val**.
- Không refit trên dev, không đánh giá trên lockbox.
- Kết quả: C = 0,1; accuracy 0,8727; macro-F1 **0,8600**; 75.895 params;
  trích HOG 40,72 s; fit 452,74 s. B0 majority: acc 0,0291, F1 0,0013.

### 3.2. Kiến trúc

![Hình 3.1. Pipeline thực nghiệm v2](outputs/figures/v2/13_pipeline_v2.png)

*Hình 3.1. Năm phase: split/audit → baseline/factorial → kiến trúc/hyperparam/aug → OOF/freeze/lockbox → thống kê/báo cáo.*

![Hình 3.2. Kiến trúc D2 — mô hình thắng](outputs/figures/v2/14_architecture_d2.png)

*Hình 3.2. D2: 4 khối conv 32/64/128/256 + BN + ReLU + maxpool, flatten 4×4×256 = 4096, FC 256, dropout 0,3, FC 43 logits.*

| Mô hình | Khối | Widths | BN | Dropout | GAP | FC hidden | Params |
|---|---|---:|---|---:|---|---:|---:|
| CNN-A (F00) | 3 | 32/64/128 | Không | 0 | Không | 256 | 2.201.707 |
| CNN-B (F07/F15) | 3 | 32/64/128 | Có | 0,3 | Không | 256 | 2.201.931 |
| CNN-C (D3) | 4 | 32/64/128/256 | Có | 0,4 | Có | – | 399.947 |
| **D2 (winner)** | 4 | 32/64/128/256 | Có | 0,3 | Không | 256 | **1.448.779** |
| D1 | 3 | 32/64/128 | Có | 0,3 | Có | – | 99.019 |
| W1 | 3 | 16/32/64 | Có | 0,3 | Không | 256 | 1.083.579 |
| W2 | 3 | 48/96/192 | Có | 0,3 | Không | 256 | 3.366.363 |

Forward: `(B,3,64,64) → logits (B,43)`, **không softmax trong model**. Khi BN bật,
conv dùng `bias=False`; khi tắt, `bias=True`.

### 3.3. Thiết kế factorial 2⁴

Cố định: 3 khối 32/64/128, FC 256, GAP off, input 64, batch 128, LR 1e-3,
40 epoch, patience 7, 3 seed. Bốn yếu tố bật/tắt:

| Yếu tố | Mức 0 | Mức 1 |
|---|---|---|
| BN | off | on |
| Dropout | 0 | 0,3 |
| Weight decay | 0 | 1e-4 |
| Augmentation | off | all |

16 tổ hợp F00–F15. Alias: R1 = F00 (CNN-A/E1), R2 = F07 (CNN-B/E2),
R3 = F15 (CNN-B+aug/E3), R4 = D3 (CNN-C+E4). Không chạy trùng.

Main effect của một yếu tố = trung bình F1 của 8 cấu hình mức 1 trừ trung bình 8
cấu hình mức 0. Interaction BN×Aug và DO×WD dùng contrast tích, kiểm chứng bằng
paired bootstrap theo track.

### 3.4. Giao thức huấn luyện cố định

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 (L1 test 1e-4) |
| Weight decay | 1e-4 (factorial test 0) |
| Batch size | 128 |
| Epochs | tối đa 40 |
| Scheduler | cosine annealing, không warmup |
| AMP | bật khi CUDA |
| Early stopping | patience 7 theo **val macro-F1** |
| Label smoothing | 0 |
| Class weight | none (CW1 test inv_sqrt) |
| Input | 64×64 RGB (I2 test 96) |
| Mean/std | theo train của từng split |
| Seed | 42, 123, 2024 |
| `num_workers` | 4–8 (hạ tầng, ghi trong meta) |
| Deterministic | bật (`torch.use_deterministic_algorithms`, warn_only) |

Early stopping chỉ dùng val của split đang train; **không** dùng lockbox/test.
Thời gian huấn luyện được đo ở phạm vi **vòng epoch** (`train_time_s`), không
gồm nạp cache/khởi động/dự đoán cuối; chi tiết và bảng theo nhóm ở mục 4.9.

### 3.5. Metric và thống kê

- **Macro-F1/43:** `f1_score(labels=range(43), average="macro", zero_division=0)`.
  Lớp vắng mặt vẫn bị tính 0, nên không so trực tiếp hai split khác support.
- **Accuracy:** top-1.
- **Top-3:** từ logits.
- **ECE 15 bin:** trung bình có trọng số |accuracy − confidence| theo bin.
- **Brier đa lớp:** trung bình tổng bình phương xác suất so với one-hot.
- **CI 95%:** cluster bootstrap theo **track**, B = 2.000:

```text
tracks = unique(track_key của eval set)
for b in 1..B:
    sampled = rng.choice(tracks, size=len(tracks), replace=True)
    frames  = tất cả frame thuộc các track được chọn
    f1_b    = macro_f1_43(y_true[frames], y_pred[frames])
CI = percentile(f1_b, [2.5, 97.5])
```

- **So sánh cặp:** cùng một mẫu track cho cả hai model, Δ_b = F1_A − F1_B;
  CI percentile; p_boot = 2·min(P(Δ≤0), P(Δ≥0)).
- **Holm–Bonferroni** cho đúng 8 so sánh chính.
- **Phân rã phương sai:** ANOVA 2 chiều seed × fold, báo `var_seed`, `var_fold`,
  `var_resid`, ICC.

### 3.6. Luật quyết định (đăng ký trước)

Cho Δ = macro-F1(config) − macro-F1(reference), CI 95% theo track:

| Tình huống | Kết luận | Hành động |
|---|---|---|
| Δ ≥ +0,3 điểm và CI_low > 0 | Tốt hơn | Giữ |
| CI nằm trong [−0,3; +0,3] | Tương đương | Chọn ít params/nhanh/đơn giản |
| CI chứa 0 nhưng vượt ±0,3 | Chưa kết luận | Báo inconclusive |
| Δ ≤ −0,3 và CI_high < 0 | Kém hơn | Bỏ |
| Aug clean-neutral, corruption Δ ≥ +0,5, CI > 0 | Giữ có điều kiện | Ghi đánh đổi |
| Aug không tăng clean, không tăng corruption | Bỏ | Giảm chi phí |

Tie-break: (1) ít params; (2) latency thấp; (3) ít aug; (4) ít epoch; (5) đơn giản.

### 3.7. Ma trận thực nghiệm và số run

| Block | Cấu hình | Seed/fold | Run | Vai trò |
|---|---|---:|---:|---|
| Baseline B1 | 1 | 1 | 1 | Mốc cổ điển |
| F factorial | 16 | 3 | 48 | BN/DO/WD/Aug |
| D architecture | 5 | 3 | 15 | GAP/depth/width |
| H hyperparam | 3 | 3 | 9 | LR/class weight/input |
| G augmentation | 3 | 3 | 9 | affine/color/perspective |
| X confirmation | 3 | 3 fold × 3 seed | 27 | OOF + control |
| Final | 1 | 3 | 3 | Train dev, test lockbox |
| **Tổng** | | | **112** (1 baseline + 111 run CNN) | |

Bản gọn theo `MA_TRAN_RUN.csv`: không chạy L2/L4/L5/P1/P2/W4/CW2/LS1/I1/BS1/BS2,
G4/G6/G7/G8, C1/W3, B0/B1b — đã chủ động bỏ trong kế hoạch v2.0 mục 6.4/6.5/6.9.

### 3.8. Cài đặt và tái lập

- Config-driven YAML: `configs/v2/*.yaml`, sinh từ `MA_TRAN_RUN.csv` bằng
  `scripts/gen_configs_v2.py`.
- Mỗi run ghi `outputs/runs/v2_<id>_s<seed>[_f<fold>]/`: `config.yaml`, `env.json`,
  `metrics.csv`, `best.pt`, `curves.png`, `meta.json`, `predictions_val.csv`.
- `metrics.csv` đúng cột: `epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb`,
  flush từng epoch.
- `best.pt` là dict: `state_dict`, `config`, `class_names`, `epoch`, `val_macro_f1`,
  `git_commit`, `torch_version`, `config_sha256`, `split_sha256`.
- Mỗi run ghi `config_sha256` (hash file YAML) và `split_sha256` (hash file split);
  final còn khớp `FREEZE.json`.

---

## Chương 4. Kết quả

> Toàn bộ số trong chương này lấy từ `outputs/results/v2/`. Bảng đầy đủ ở
> `outputs/results/v2/tables/`. Số in đậm là số dùng để kết luận.

### 4.1. Baseline

| Mô hình | Accuracy | Macro-F1/43 | Params | Fit time |
|---|---:|---:|---:|---:|
| B0 majority | 0,0291 | 0,0013 | 0 | – |
| B1 HOG + LinearSVC (C=0,1) | 0,8727 | 0,8600 | 75.895 | 452,74 s |

CNN tốt nhất trên cùng screening val (D2) đạt **0,97158 macro-F1**, cao hơn B1
**11,16 điểm**. B1 không được refit trên dev và không đánh giá trên lockbox.

### 4.2. Factorial 2⁴

48 run (16 cấu hình × 3 seed). Bảng đầy đủ:

| ID | BN | DO | WD | Aug | Acc (%) | Macro-F1 (%) | Δ vs F00 (điểm %) [CI] | Gap | Time (s) |
|---|---|---|---|---|---:|---:|---|---:|---:|
| F00 | off | 0 | 0 | off | 90,092 ± 1,902 | 84,109 ± 3,710 | – | 9,87 | 52,6 |
| F01 | on | 0 | 0 | off | 93,263 ± 0,708 | 87,846 ± 1,524 | +3,736 [+0,850; +5,771] | 6,74 | 60,9 |
| F02 | off | 0,3 | 0 | off | 94,121 ± 0,147 | 91,613 ± 0,505 | +7,504 [+2,872; +9,657] | 5,63 | 68,3 |
| F03 | on | 0,3 | 0 | off | 96,969 ± 0,566 | 96,047 ± 0,975 | +11,937 [+5,769; +14,562] | 2,53 | 65,7 |
| F04 | off | 0 | 1e-4 | off | 90,248 ± 0,828 | 84,173 ± 0,184 | +0,063 [−2,102; +1,751] | 9,73 | 45,1 |
| F05 | on | 0 | 1e-4 | off | 92,643 ± 0,587 | 87,470 ± 1,106 | +3,361 [+0,488; +5,231] | 7,28 | 49,2 |
| F06 | off | 0,3 | 1e-4 | off | 94,094 ± 1,167 | 91,958 ± 2,222 | +7,849 [+3,130; +9,642] | 5,73 | 70,2 |
| F07 | on | 0,3 | 1e-4 | off | 96,980 ± 0,532 | **96,501 ± 1,041** | +12,391 [+6,017; +14,993] | 2,60 | 69,8 |
| F08 | off | 0 | 0 | on | 92,039 ± 0,851 | 86,672 ± 1,008 | +2,563 [−0,465; +4,558] | 7,41 | 126,4 |
| F09 | on | 0 | 0 | on | 94,461 ± 1,591 | 90,006 ± 1,738 | +5,897 [+2,347; +7,731] | 5,22 | 155,1 |
| F10 | off | 0,3 | 0 | on | 95,086 ± 0,618 | 91,097 ± 1,628 | +6,988 [+2,903; +8,844] | 4,11 | 173,9 |
| F11 | on | 0,3 | 0 | on | 96,688 ± 0,272 | 94,209 ± 0,198 | +10,100 [+5,224; +12,223] | 1,44 | 168,0 |
| F12 | off | 0 | 1e-4 | on | 92,864 ± 0,178 | 87,003 ± 1,058 | +2,893 [+0,038; +5,092] | 6,97 | 159,3 |
| F13 | on | 0 | 1e-4 | on | 94,757 ± 1,730 | 90,700 ± 1,861 | +6,591 [+2,630; +8,601] | 4,98 | 158,2 |
| F14 | off | 0,3 | 1e-4 | on | 94,730 ± 0,294 | 90,310 ± 0,314 | +6,201 [+2,501; +8,171] | 4,00 | 154,5 |
| F15 | on | 0,3 | 1e-4 | on | 96,564 ± 0,330 | 93,946 ± 1,108 | +9,837 [+5,138; +12,325] | 1,08 | 151,6 |

**Main effect (điểm %):**

| Yếu tố | Δ | 95% CI | Kết luận |
|---|---:|---|---|
| BN | **+3,724** | [+1,767; +5,005] | Tốt hơn |
| Dropout 0,3 | **+5,963** | [+2,873; +7,147] | Tốt hơn |
| Weight decay 1e-4 | +0,058 | [−0,506; +0,551] | Không đủ bằng chứng |
| Augmentation all | +0,529 | [−1,204; +2,303] | Không đủ bằng chứng |
| BN × Aug | −0,279 | [−1,058; +0,551] | Không rõ |
| Dropout × WD | −0,121 | [−0,686; +0,548] | Không rõ |

**Đọc kỹ tương tác:** aug giúp khi chưa có regularization (F08−F00 = **+2,563**)
nhưng hại khi đã đủ regularization (F15−F07 = **−2,555**). Vì vậy không được
tuyên bố "aug giúp X điểm" chung chung.

![Hình 4.1. Learning curve 4 cấu hình đại diện](outputs/figures/v2/01_learning_curves.png)

*Hình 4.1. F00 overfit sớm (gap 9,87 điểm); F07 gap 2,60; F15 gap 1,08 nhưng F1 thấp hơn F07; D2 hội tụ ổn định.*

![Hình 4.2. Main effect và interaction của factorial](outputs/figures/v2/02_factorial_effects.png)

*Hình 4.2. BN và dropout dương rõ; WD/aug không đủ bằng chứng; interaction nhỏ.*

### 4.3. Kiến trúc

| ID | Thay đổi so với D4 | Acc (%) | Macro-F1 (%) | Δ vs D4 (điểm %) [CI] | Params | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|---:|
| D1 | GAP, 3 khối | 74,234 ± 0,592 | 59,188 ± 0,764 | **−34,758** [−36,181; −22,916] | 99.019 | 213,2 | 40,0 |
| D2 | 4 khối, flatten | 98,004 ± 0,297 | **97,158 ± 0,610** | **+3,212** [+0,036; +4,440] | 1.448.779 | 132,8 | 23,3 |
| D3 | 4 khối + GAP + DO 0,4 (E4 cũ) | 96,575 ± 0,580 | 93,105 ± 0,906 | −0,841 [−2,724; +0,757] | 399.947 | 184,9 | 34,0 |
| W1 | 16/32/64 | 94,876 ± 0,916 | 90,686 ± 1,425 | −3,260 [−5,250; −0,735] | 1.083.579 | 181,9 | 33,7 |
| W2 | 48/96/192 | 97,104 ± 0,368 | 95,215 ± 1,230 | +1,268 [−0,585; +2,499] | 3.366.363 | 186,9 | 34,3 |

- **GAP 3 khối thất bại rõ**: 99k params, train acc 84,6% (seed 42), val trung bình
  74,2% sau 40 epoch → underfit vì thiếu dung lượng/tối ưu, không phải "GAP mất
  chi tiết" đơn thuần.
- **4 khối tốt hơn 3 khối** 3,212 điểm, CI không chứa 0 → bác bỏ giả thuyết H4.
- D3 tương đương D4 → chỉ dùng map E4 cũ, không kết luận cơ chế.
- Width nhỏ kém; width lớn +1,268 nhưng CI chứa 0 → chưa kết luận, params 3,37M.

![Hình 4.3. Pareto params vs macro-F1](outputs/figures/v2/03_pareto_params_f1.png)

*Hình 4.3. D2 nằm trên frontier: F1 cao nhất trong nhóm kiến trúc, 1,45M params.*

### 4.4. Hyperparam

| ID | Thay đổi | Acc (%) | Macro-F1 (%) | Δ vs F15 (điểm %) [CI] | Params | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|---:|
| L1 | LR 1e-4 | 91,888 ± 1,282 | 86,807 ± 1,615 | **−7,139** [−9,372; −3,353] | 2.201.931 | 160,5 | 29,0 |
| CW1 | class weight inv_sqrt | 95,960 ± 0,606 | 92,943 ± 0,475 | −1,004 [−2,195; +0,261] | 2.201.931 | 183,3 | 33,7 |
| I2 | input 96 | 94,213 ± 0,624 | 90,499 ± 1,429 | **−3,448** [−5,463; −0,984] | 4.823.371 | 293,1 | 36,7 |

- LR 1e-3 > 1e-4 rõ rệt (1e-4 không hội tụ trong 40 epoch).
- Class weight không cải thiện (CI chứa 0) → bỏ; đây là exploratory.
- Input 96 kém hơn 64 và params tăng 2,2× → giữ 64.

### 4.5. Augmentation và corruption

| ID | Profile | Acc (%) | Macro-F1 (%) | Δ vs G0 (điểm %) [CI] | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|
| G1 | affine | 95,572 ± 0,190 | 92,251 ± 0,595 | **−4,250** [−6,972; −0,084] | 92,1 | 36,0 |
| G2 | color | 97,767 ± 0,242 | **97,417 ± 0,340** | +0,917 [−0,141; +1,842] | 118,7 | 28,3 |
| G3 | perspective | 96,375 ± 1,116 | 95,264 ± 1,988 | −1,237 [−2,706; +0,530] | 64,2 | 22,3 |

**Corruption (accuracy trung bình 3 mức, screening val, seed 42):**

| Config | Blur | Noise | Bright | Contrast | JPEG | Occlusion | TB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F07 (G0) | 95,05 | 86,27 | 97,13 | 93,49 | 94,00 | 23,11 | 81,51 |
| F15 (G5) | 93,73 | 81,00 | 96,19 | 92,77 | 91,22 | 26,43 | 80,22 |
| G1 affine | 91,35 | 75,84 | 94,75 | 86,81 | 88,38 | 24,24 | 76,90 |
| G2 color | 95,73 | 85,86 | 97,56 | 96,06 | 95,07 | 21,56 | **81,97** |
| G3 perspective | 94,06 | 84,85 | 96,78 | 93,65 | 93,49 | 21,44 | 80,71 |
| D2 (4 khối + all) | 93,88 | 83,59 | 97,10 | 92,92 | 92,78 | 17,63 | 79,65 |

![Hình 4.4. Corruption curves 6 loại × 3 mức](outputs/figures/v2/06_corruption_curves.png)

*Hình 4.4. Color-only (G2) robust nhất; affine (G1) kém nhất; occlusion làm sập mọi mô hình.*

Kết luận: "all" không có lợi (clean −2,555 so với F07; corruption TB thấp hơn);
affine là thành phần hại; color trung tính (Δ +0,917, CI chứa 0) và corruption
+0,46 — dưới ngưỡng +0,5 nên **không đủ bằng chứng để giữ aug vì robustness**.

### 4.6. OOF 3 fold × 3 seed

| Config | Nguồn | OOF macro-F1 (%) | CI 95% |
|---|---|---:|---|
| X1 | G2 (3 khối + color) | 96,297 ± 0,296 | [94,628; 97,020] |
| **X2** | **D2 (4 khối + all)** | **97,040 ± 0,187** | **[95,051; 97,942]** |
| X3 | F07 (control) | 95,059 ± 0,568 | [93,113; 95,930] |

| So sánh | Δ (điểm %) | 95% CI | Kết luận |
|---|---:|---|---|
| X1 − X2 | −0,743 | [−1,590; +0,274] | Chưa kết luận |
| X1 − X3 | **+1,238** | [+0,586; +2,071] | X1 tốt hơn control |
| X2 − X3 | **+1,981** | [+1,053; +2,961] | X2 tốt hơn control |

OOF có **278.091 dòng** = 30.899 ảnh × 3 seed × 3 config; mỗi ảnh dev được dự
đoán đúng một lần mỗi seed.

**Phân rã phương sai:** `var_seed` = 4,42e−06 (0,9%), `var_fold` = 4,20e−04
(**91,0%**), `var_resid` = 3,82e−05 (8,3%). Kết luận: báo cáo theo fold là bắt
buộc; multi-seed một mình không đủ.

### 4.7. Lockbox và official test

| Tập | n | Accuracy | Macro-F1/43 |
|---|---:|---:|---:|
| **Lockbox** | 8.302 | **0,988758 ± 0,001010** | **0,975981 ± 0,006246** |
| Official raw | 12.630 | 0,986989 ± 0,001625 | 0,980398 ± 0,004516 |
| Official clean | 11.879 | 0,988692 ± 0,002359 | 0,982499 ± 0,005314 |

- Lockbox top-3 0,998474 ± 0,000387; ECE 0,005328; Brier 0,017862.
- Ba seed lockbox: acc 0,987593 / 0,989400 / 0,989280; F1 0,968770 / 0,979706 / 0,979468.
- Official clean **cao hơn** raw → ảnh gần trùng không thổi phồng.
- Official test không track-disjoint; không so trực tiếp với lockbox.

### 4.8. Tám so sánh chính (Holm)

| Yếu tố | Δ (điểm %) | 95% CI | p Holm | Kết luận |
|---|---:|---|---:|---|
| BN | +3,724 | [+1,767; +5,005] | 0,000 | Tốt hơn |
| Dropout | +5,963 | [+2,873; +7,147] | 0,000 | Tốt hơn |
| Weight decay | +0,058 | [−0,506; +0,551] | 0,412 | Chưa kết luận |
| Augmentation | +0,529 | [−1,204; +2,303] | 0,708 | Chưa kết luận |
| GAP | −34,758 | [−36,108; −22,444] | 0,000 | Kém hơn |
| Depth | +3,212 | [+0,011; +4,513] | 0,373 | CI ủng hộ; Holm chưa đủ mạnh |
| Width | +1,268 | [−0,591; +2,541] | 0,708 | Chưa kết luận |
| LR | −7,139 | [−9,353; −3,366] | 0,000 | Kém hơn (1e-4) |

> Depth có CI không chứa 0 nhưng p Holm 0,373: báo "có bằng chứng vừa, chưa đủ
> sau hiệu chỉnh". Quyết định chọn D2 dựa trên OOF (Δ +1,981 so với control,
> CI [+1,053; +2,961]) và tie-break ít params hơn.

### 4.9. Chi phí huấn luyện

**Phạm vi đo:** `train_time_s` là tổng thời gian **vòng epoch** (epoch 1 → epoch
cuối), đo bằng `time.perf_counter()` trong `train_v2`. Đo trên RTX 5070 12 GB,
i9-14900KF, AMP bật, `num_workers` 4–8. Không gồm: nạp cache/khởi động process,
dự đoán val cuối, vẽ curve, đo latency, phân tích. Thời gian theo đồng hồ thực
tế sẽ lớn hơn con số dưới đây.

**Theo nhóm thí nghiệm:**

| Nhóm | Run | Tổng epoch | Tổng thời gian (phút) | TB/run (s) | Trung vị (s) | Min–Max (s) | TB/epoch (s) | GPU mem max (MB) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| D_architecture | 15 | 496 | 45,0 | 179,9 | 184,3 | 112–215 | 5,44 | 465 |
| FINAL | 3 | 78 | 7,9 | 158,2 | 158,0 | 158–159 | 6,08 | 248 |
| F_factorial | 48 | 1.387 | 86,4 | 108,0 | 103,0 | 35–177 | 3,74 | 319 |
| G_augmentation | 9 | 260 | 13,7 | 91,7 | 95,4 | 52–142 | 3,17 | 319 |
| H_hyperparam | 9 | 298 | 31,8 | 212,3 | 196,6 | 141–318 | 6,41 | 694 |
| X_confirmation | 27 | 892 | 54,6 | 121,3 | 130,2 | 56–193 | 3,67 | 319 |
| **Tổng CNN** | **111** | **3.411** | **239,5** | 129,5 | 135,5 | 35–318 | **4,21** | **694** |
| B1 HOG+SVC | 1 | – | 8,2 | 493,5 | – | – | – | – |

**Theo cấu hình chính:**

| Config | Vai trò | Epochs (TB) | Best epoch (TB) | Time/run (s) | s/epoch | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| F00 | CNN-A, mốc | 19,7 | 12,7 | 52,6 | 2,67 | 0,73 |
| F07 | CNN-B, không aug | 27,7 | 20,7 | 69,8 | 2,52 | 0,74 |
| F15 | CNN-B + aug | 33,3 | 26,3 | 151,6 | 4,55 | 0,73 |
| D1 | 3 khối + GAP | 40,0 | 34,3 | 213,2 | 5,33 | 0,68 |
| **D2** | **4 khối + flatten** | 23,3 | 16,3 | **132,8** | 5,69 | **0,94** |
| D3 | 4 khối + GAP (E4) | 34,0 | 27,3 | 184,9 | 5,44 | 0,87 |
| W1 | width nhỏ | 33,7 | 28,3 | 181,9 | 5,40 | 0,70 |
| W2 | width lớn | 34,3 | 27,7 | 186,9 | 5,44 | 0,73 |
| L1 | LR 1e-4 | 29,0 | 22,0 | 160,5 | 5,54 | 0,73 |
| CW1 | class weight | 33,7 | 28,0 | 183,3 | 5,44 | 0,73 |
| I2 | input 96 | 36,7 | 31,0 | **293,1** | **7,99** | 0,78 |
| G1 | affine | 36,0 | 29,0 | 92,1 | 2,56 | 0,73 |
| G2 | color | 28,3 | 21,3 | 118,7 | 4,19 | 0,73 |
| G3 | perspective | 22,3 | 15,3 | 64,2 | 2,87 | 0,73 |
| X1 | OOF G2 | 36,2 | 30,9 | 133,8 | 3,70 | 0,73 |
| X2 | OOF D2 | 32,6 | 26,3 | 157,6 | 4,85 | 0,95 |
| X3 | OOF control | 30,3 | 23,8 | 72,4 | 2,43 | 0,74 |
| **final** | **D2 trên dev** | **26,0** | **26,0** | **158,2** | **6,08** | **0,93** |

**Tổng kết chi phí:**

- 111 run CNN: **3.411 epoch**, **239,5 phút ≈ 3,99 giờ** thời gian epoch loop.
- Baseline B1: 40,72 s trích HOG + 452,74 s fit = **493,5 s ≈ 8,2 phút**.
- **Tổng cho toàn bộ thực nghiệm: ≈ 4,13 giờ** (chưa gồm phân tích/hình/báo cáo).
- Final: 3 seed × 26 epoch, **157,83 / 158,00 / 158,70 s**; GPU mem ~247–248 MB.
- D2 screening nhanh hơn D1/D3 nhờ early stop sớm (23,3 epoch so với 40,0/34,0),
  dù mỗi epoch nặng hơn (5,69 s so với 5,33/5,44).
- Augmentation làm tăng thời gian rõ: F00 52,6 s → F08 126,4 s (cùng 3 khối,
  chỉ khác aug); F15 151,6 s so với F07 69,8 s.
- I2 (input 96) là cấu hình đắt nhất: 293,1 s/run, 7,99 s/epoch, GPU mem 694 MB.

Nguồn máy đọc: `outputs/results/v2/training_cost.csv`,
`training_cost_summary.json`, `tables/training_cost.md`; log
`outputs/results/v2/log_training_cost.txt`.

---

## Chương 5. Giải thích cơ chế

### 5.1. Learning curve và overfit

- F00 (không BN/DO/WD): train acc ~1,0 rất sớm, gap train−val **9,87 điểm**,
  F1 dao động mạnh theo seed (0,799–0,870).
- F07 (BN+DO+WD): gap **2,60 điểm**, F1 0,96501 ± 0,01041.
- F15 (thêm aug all): gap **1,08 điểm** nhưng F1 giảm còn 0,93946 → aug giảm
  overfit nhưng không chuyển thành lợi ích val ở ngân sách 40 epoch.
- D2 (4 khối + all): OOF 0,97040, kiểm soát overfit tốt hơn F00.

### 5.2. Per-class

![Hình 5.1. ΔF1 từng lớp so với F15 (screening val, seed 42)](outputs/figures/v2/04_per_class_delta.png)

*Hình 5.1. Heatmap ΔF1 43 lớp của F00/F07/F15/D2/G2 so với F15.*

![Hình 5.2. F1 từng lớp trên lockbox, trung bình 3 seed](outputs/figures/v2/11_per_class_f1_lockbox.png)

*Hình 5.2. 16/43 lớp đạt F1 ≥ 0,999; yếu nhất là lớp 19 (0,760, n=30), lớp 29 (0,776, n=60), lớp 30 (0,925), lớp 28 (0,928).*

Các lớp yếu là lớp hiếm/khó, không phải lỗi split: support lockbox vẫn ≥ 30 ảnh.

### 5.3. Confusion matrix

![Hình 5.3. Confusion chuẩn hóa hàng — D2 trên screening val](outputs/figures/v2/07_confusion_d2_s42.png)

*Hình 5.3. D2 trên screening val, seed 42; chuẩn hóa theo hàng.*

![Hình 5.4. Confusion chuẩn hóa hàng — lockbox, trung bình 3 seed](outputs/figures/v2/10_confusion_lockbox.png)

*Hình 5.4. Lockbox 8.302 ảnh; phần lớn khối chéo nằm ở các lớp giống nhau về hình dạng/màu.*

### 5.4. Grad-CAM

![Hình 5.5. Grad-CAM 12 ảnh: CNN-B (F15) vs CNN-C (D3)](outputs/figures/v2/08_gradcam_grid.png)

*Hình 5.5. 4 lớp × 3 trường hợp (đúng tự tin, sai, confidence thấp); model nhìn vào biển, không phải nền; ca sai thường lệch sang viền/ánh sáng.*

### 5.5. Calibration

![Hình 5.6. Reliability diagram 15 bin](outputs/figures/v2/05_reliability.png)

*Hình 5.6. Regularization giảm tự tin sai: ECE F00 0,0758 → F07 0,0154 → F15 0,0120 → G2 0,0101 → D2 0,0084.*

| Model | ECE 15 bin |
|---|---:|
| F00 | 0,075824 |
| F07 | 0,015400 |
| F15 | 0,011968 |
| G2 | 0,010131 |
| D2 | 0,008353 |

Lockbox ECE 0,005328 — mô hình cuối calibrated tốt.

### 5.6. Corruption

Đã trình bày ở mục 4.5 và Hình 4.4. Điểm đáng chú ý: occlusion (che 10–30%
trung tâm) làm accuracy rơi còn 17,63–26,43% ở mọi mô hình. Đây là điểm yếu thật,
không phải lỗi đánh giá: biển bị che mất phần quyết định.

### 5.7. Error taxonomy

![Hình 5.7. 12 mẫu lỗi trên screening val (F15, seed 42)](outputs/figures/v2/09_error_samples.png)

*Hình 5.7. 12 mẫu lỗi đầu tiên; phân nhóm heuristic theo percentile blur/brightness/confidence.*

- 190 lỗi / 6.180 ảnh: **102 nhãn khó/góc nhìn lạ**, **64 mờ**, **23 nhầm cặp
  giống nhau/confidence thấp**, **1 chói**.
- Phân nhóm là heuristic, cần xem thủ công trước khi kết luận nguyên nhân.

### 5.8. Vì sao chọn D2 (khung trả lời)

> Trên OOF 30.899 ảnh dev (3 fold × 3 seed), D2 đạt macro-F1/43 **0,970400**
> [0,950511; 0,979422], accuracy **0,98004**. Cấu hình này được chọn vì:
>
> 1. **So với CNN-A (F00):** hơn **12,391 điểm** (F07 so với F00; D2 so với control
>    +1,981 điểm [1,053; 2,961]). Cơ chế: BN + dropout ổn định tối ưu và giảm gap
>    từ **9,87** xuống **2,60** điểm.
> 2. **So với 4 khối + GAP (D3):** D2 hơn **4,053 điểm** trên screening
>    (97,158 so với 93,105) nhưng D3 đổi cả GAP/dropout nên chỉ là legacy; thí nghiệm
>    tách riêng cho thấy GAP 3 khối (D1) kém **34,758 điểm**, còn depth 4 khối
>    (D2) hơn 3 khối **3,212 điểm**.
> 3. **Về regularization:** BN **+3,724**, dropout **+5,963** (CI > 0); WD
>    **+0,058** và aug **+0,529** → **không đủ bằng chứng**, không giữ vì cảm tính.
> 4. **Về đánh đổi:** 1.448.779 params, latency model-only **0,93 ms/ảnh** (final);
>    ít params hơn X1 (2,20M) nên thắng tie-break khi X1−X2 chưa kết luận.
> 5. **Về aug:** clean Δ của "all" là **−2,555 điểm** so với F07; corruption TB
>    79,65 so với 81,51 của F07 → aug không có lợi; color-only trung tính. Thời
>    gian train của F15 cũng gần gấp đôi F07 (151,6 s so với 69,8 s).
>
> Theo luật quyết định đăng ký trước, kết luận: **D2 tốt hơn control rõ ràng;
> so với X1 là tương đương/chưa kết luận, chọn D2 vì ít params hơn**. Kết quả
> lockbox 0,975981 xác nhận OOF không bị thổi phồng.

---

## Chương 6. Tái lập, nghiệm thu, hạn chế

### 6.1. Tái lập

- Seed 42/123/2024; cùng split; cùng early stopping.
- `scripts/repro_check_v2.py`: chạy 2 lần 3 epoch cùng seed → sai khác
  `train_loss = val_loss = val_acc = val_macro_f1 = 0,0` (`reproducibility.json`).
- `pytest`: **22 passed** (`tests/test_split_v2.py`, `test_stats_v2.py`,
  `test_config_v2.py`).
- `verify_41.json`: macro-F1 OOF tính lại lệch **0,0**; summary F15 lệch **3,1e−07**.
- Mọi run có `git_commit`, `torch_version`, `config_sha256`, `split_sha256`.

Lệnh chạy lại đầy đủ ở Phụ lục E.

### 6.2. Nghiệm thu 12/12

| # | Điều kiện | Đạt | Bằng chứng |
|---|---|---|---|
| 1 | Lockbox 0 track/sign-group overlap; one-shot | ✓ | `audit_report.json`, `lockbox_access.log` |
| 2 | Mọi run kết luận dùng split sign-group | ✓ | `summary.csv` (111 run) |
| 3 | Mỗi kết luận có Δ + CI + luật | ✓ | `main_comparisons.csv` |
| 4 | Factorial đủ 16 config | ✓ | 48 run |
| 5 | Tách GAP/depth/width; D3 legacy | ✓ | `tables/architecture.md` |
| 6 | Aug bật/tắt + corruption | ✓ | `corruption.csv` |
| 7 | OOF 3 fold × 3 seed | ✓ | 278.091 dòng |
| 8 | Decision card 10 yếu tố | ✓ | `decision_cards.csv` |
| 9 | Không dùng số random cũ | ✓ | Phụ lục A |
| 10 | Không tune lockbox/test | ✓ | `FREEZE.json`, log |
| 11 | Repo sạch | ✓ | `git ls-files` |
| 12 | Tái lập cùng seed | ✓ | `reproducibility.json` |

Chi tiết máy đọc: `outputs/results/v2/ACCEPTANCE.json` (**12/12**).

### 6.3. Hạn chế (thật, không giấu)

1. Lockbox 8.302 ảnh / 277 track nhỏ hơn official test 12.630 → CI rộng hơn;
   không so trực tiếp hai con số.
2. Official test **không track-disjoint**; audit lại tìm 751 ảnh gần trùng
   (5,95%), nhiều hơn 131 của audit cũ do phương pháp khác. Clean 11.879.
3. Occlusion là điểm yếu lớn: accuracy 17,63–26,43% khi che 10–30% trung tâm.
4. Screening chỉ 24.719 ảnh, ít hơn final 30.899; kết luận screening chỉ để chọn.
5. Augmentation "all" không có lợi; không thể nói "aug luôn tốt".
6. GAP 3 khối underfit nặng; không suy rộng thành "GAP luôn xấu".
7. WD/width/aug: **chưa kết luận**; không được trình bày là "tốt hơn".
8. Sự cố mở lockbox 2 lần do bug ghi file (lần 1 không ghi được kết quả); config/
   model/hash không đổi, kết quả lần 2 trùng khớp tuyệt đối lần 1. Đã ghi log.
9. Không dùng pretrained; không tuyên bố mô hình đủ an toàn cho giao thông thực tế.

### 6.4. Hướng phát triển

- Bổ sung test theo 3 lockbox seed khác để kiểm tra độ nhạy của split.
- Thêm corruption nặng hơn (motion blur, tuyết/mưa) và occlusion ngẫu nhiên.
- Thử distillation hoặc pruning cho D2 (1,45M params) để giảm latency.
- Cập nhật demo Streamlit sang checkpoint v2 và ngưỡng "không xác định" chọn trên
  validation của split v2 (hiện demo vẫn dùng checkpoint cũ, không thuộc phạm vi v2).

### 6.5. Kết luận

V2 trả lời được bốn phản biện bằng bằng chứng định lượng: split sạch theo
sign-group, chọn mô hình bằng OOF + lockbox, mỗi yếu tố có Δ + CI + luật quyết
định, và mỗi kết luận có ít nhất một hình/bảng cơ chế. Mô hình cuối D2 đạt
macro-F1/43 **0,975981** trên lockbox độc lập và **0,982499** trên official test
clean. Các kết quả âm (GAP 3 khối, aug "all") được giữ nguyên và giải thích.

---

## Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks.
2. scikit-learn: `f1_score`, `classification_report`, `LinearSVC`.
3. Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
4. Holm, S. (1979). *A simple sequentially rejective multiple test procedure.*
5. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern Neural Networks.*
6. Hendrycks, D., Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*
7. Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
8. Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning Benchmarks.*

---

## Phụ lục A. Legacy leaky (không dùng để kết luận)

Các run cũ dùng `split_random`; val bị rò rỉ frame. Bản sao:
`outputs/results/legacy/summary_random_split.csv`. Nhãn: `legacy_leaky`.

| Run | Model | Val acc | Val macro-F1 | Ghi chú |
|---|---|---:|---:|---|
| e0a_majority | majority | 0,0574 | 0,0025 | mốc dưới |
| e0b_hog_svc | HOG+SVC | 0,9657 | 0,9740 | baseline cũ |
| e1_cnna_s42 | CNN-A | 0,9963 | 0,9957 | leak |
| e2_cnnb_s42 | CNN-B | 0,9983 | 0,9984 | leak |
| e3_cnnb_aug_s42 | CNN-B+aug | 0,9994 | 0,9995 | leak |
| e4_cnnc_gap_aug_s42 | CNN-C | 0,9997 | 0,9998 | leak |
| e5a_lr3e4_s42 | CNN-B | 0,9895 | 0,9934 | 10 epoch |
| e5b_lr3e3_s42 | CNN-B | 0,7720 | 0,5672 | 10 epoch, phân kỳ |
| e7_classweight_s42 | CNN-B | 0,9985 | 0,9989 | leak |

## Phụ lục B. Artifact chính

| Nội dung | File |
|---|---|
| Split + hash + mean/std | `outputs/splits/v2/manifest.json` |
| Audit C1–C12 | `outputs/results/v2/audit/audit_report.json` |
| Baseline | `outputs/results/v2/baseline.json` |
| Factorial 2⁴ | `outputs/results/v2/tables/factorial.md`, `factorial_effects.json` |
| Kiến trúc/hyperparam/aug | `tables/architecture.md`, `hyperparam.md`, `augmentation.md` |
| Corruption | `outputs/results/v2/corruption.csv` |
| Summary 111 run | `outputs/results/v2/summary.csv` |
| OOF | `oof_predictions.csv`, `oof_summary.json`, `variance.json` |
| Decision cards | `outputs/results/v2/decision_cards.csv` |
| Freeze | `outputs/results/v2/FREEZE.json` |
| Lockbox | `outputs/results/v2/lockbox_result.json`, `lockbox_access.log` |
| Official test | `outputs/results/v2/official_test_v2.json` |
| Nghiệm thu | `outputs/results/v2/ACCEPTANCE.json` |
| Hình | `outputs/figures/v2/` |
| Nhật ký 18 bước | `docs/kehoach/1/KET_QUA_TRIEN_KHAI_V2.md` |

## Phụ lục C. Bảng 2⁴ đầy đủ

Xem mục 4.2 (16 dòng) hoặc file máy đọc `outputs/results/v2/tables/factorial.md`.

## Phụ lục D. Per-class lockbox

File đầy đủ: `outputs/results/v2/lockbox_per_class.csv` (F1 mean/sd, support 43 lớp).

## Phụ lục E. Lệnh chạy lại

```powershell
python -m src.split_v2 --build
python -m src.audit_v2
python -m src.baseline_v2
python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024
python scripts/run_matrix.py --configs d1,d2,d3,w1,w2 --seeds 42,123,2024
python scripts/run_matrix.py --configs l1,cw1,i2 --seeds 42,123,2024
python scripts/run_matrix.py --configs g1,g2,g3 --seeds 42,123,2024
python -m src.summarize_v2
python scripts/select_candidates.py
python scripts/run_matrix.py --configs x1,x2,x3 --seeds 42,123,2024 --folds 0,1,2
python scripts/run_oof_analysis.py --configs x1,x2,x3
python scripts/freeze_v2.py --winner x2 --source d2 --reason "..."
python scripts/run_matrix.py --configs final --seeds 42,123,2024
python -m src.final_eval_v2 --mode lockbox --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
python -m src.final_eval_v2 --mode test --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
python scripts/run_phase4.py
python scripts/acceptance_v2.py
```

## Phụ lục F. Đính chính các câu sai (mục 12.4 kế hoạch)

| Câu cũ | Trạng thái | Đính chính bằng số |
|---|---|---|
| "Val random 99,9% chứng minh mô hình gần hoàn hảo" | Sai | Val random bị leak; số sạch: OOF 0,970400; lockbox 0,975981 |
| "17.214 cặp pHash chỉ là hash collision" | Quá mạnh | Audit lại: 751 ảnh test gần trùng (MAD<5), 24.554 cặp |
| "E4 kém vì GAP" | Chưa chứng minh | E4 đổi 4 thứ; D1 GAP 3 khối −34,758; D2 depth +3,212 |
| "Aug luôn tốt" | Sai | Main effect +0,529 [−1,204; +2,303]; F15−F07 = −2,555; affine −4,250 |
| "Test độc lập" | Không chính xác | Official test không track-disjoint; chỉ benchmark phụ |
