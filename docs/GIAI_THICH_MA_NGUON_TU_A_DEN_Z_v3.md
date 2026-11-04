# GIẢI THÍCH MÃ NGUỒN V3 — TỪ A ĐẾN Z

## Dành cho người có nền tảng học sâu rất yếu

| Thông tin | Giá trị |
|---|---|
| Tài liệu | Giải thích mã nguồn dự án GTSRB-CNN, bản v3 |
| Đối tượng | Người mới, biết Python cơ bản, chưa quen học sâu |
| Phạm vi | Các bước **mới** của v2/v3; bước cũ chỉ nhắc để tránh nhầm |
| Cách đọc | Mỗi mục trả lời: **làm gì → tại sao → code ở đâu → kết quả** |
| Báo cáo kèm theo | `docs/bao_cao/BAO_CAO_v3.md` và `.docx` |
| Ngày | 04/11/2026 |

---

## 0. Đọc tài liệu này thế nào?

Bạn **không cần đọc hết code** trước. Hãy đi theo 10 chặng:

1. Hiểu bức tranh lớn (Chương 1).
2. Hiểu dữ liệu đi qua những bước nào (Chương 2).
3. Hiểu vì sao phải chia dữ liệu theo track/sign-group (Chương 3).
4. Hiểu mô hình CNN gồm những gì (Chương 4).
5. Hiểu một lần huấn luyện chạy ra sao (Chương 5).
6. Hiểu 111 run được tổ chức thế nào (Chương 6).
7. Hiểu các con số thống kê có nghĩa gì (Chương 7).
8. Hiểu cách chọn mô hình và mở lockbox (Chương 8).
9. Hiểu các hình phân tích và cách đọc (Chương 9).
10. Hiểu demo, tái lập, runbook và cách đọc kết quả (Chương 10–13).

### Mục lục nhanh

| Chương | Nội dung |
|---|---|
| 0 | Cách đọc tài liệu |
| 1 | Bức tranh lớn, pipeline, khái niệm cốt lõi |
| 2 | Dữ liệu, cache 64×64, tiền xử lý, augmentation |
| 3 | Track, sign-group, chia 3 tầng, audit |
| 4 | Mô hình SmallCNN, D2 |
| 5 | Một run huấn luyện, checkpoint, log |
| 6 | 111 run, config, matrix, summary |
| 7 | Thống kê: CI, bootstrap, Holm, variance |
| 8 | Chọn mô hình, OOF, freeze, lockbox, test |
| 9 | Giải thích cơ chế: hình, Grad-CAM, calibration |
| 10 | Demo và triển khai |
| 11 | Tái lập và nghiệm thu |
| 12 | Runbook từ máy trắng |
| 13 | Đọc kết quả, đính chính, FAQ |
| 14 | Glossary và bản đồ file |

Quy ước:

- **In đậm** là khái niệm quan trọng.
- `code` là tên file, hàm hoặc biến.
- “v3” trong tài liệu này = **báo cáo làm lại + pack tài nguyên**, không phải
  một protocol huấn luyện mới. Protocol huấn luyện vẫn là v2.
- Các bước v1 (split random, E1–E7) chỉ nhắc ở mục 1.4; **không dùng** để kết luận.

---

## 1. Bức tranh lớn

### 1.1. Một câu tóm tắt

Dự án dạy một mạng nơ-ron tự xây dựng nhìn ảnh biển báo 64×64 và chọn một trong
43 lớp; sau đó kiểm tra xem nó học thật hay chỉ học thuộc, bằng cách chia dữ liệu
theo **track/sign-group**, chọn mô hình bằng **OOF**, và xác nhận bằng **lockbox**
mở một lần.

### 1.2. Sơ đồ pipeline

```text
Ảnh GTSRB (39.209 train + 12.630 test)
        │
        ▼
[1] Index + ROI + cache 64×64        src/data.py
        │
        ▼
[2] Chia 3 tầng theo sign-group      src/split_v2.py
    ├── Lockbox 8.302 (mở 1 lần)
    └── Dev 30.899
         ├── Screening 24.719 / 6.180
         └── OOF 3 fold × 3 seed
        │
        ▼
[3] Train CNN                        src/train_v2.py
    ├── Factorial 2⁴ (48 run)
    ├── Kiến trúc (15 run)
    ├── Hyperparam (9 run)
    ├── Augmentation (9 run)
    └── OOF (27 run) + Final (3 run)
        │
        ▼
[4] Chọn mô hình                      scripts/select_candidates.py
    OOF → freeze → lockbox → test     src/final_eval_v2.py
        │
        ▼
[5] Phân tích + báo cáo              src/explain_v2.py
    per-class, confusion, Grad-CAM,   scripts/report_v3_*.py
    calibration, robustness
```

### 1.3. Năm phase và file phụ trách

| Phase | Việc chính | File chính | Kết quả |
|---|---|---|---|
| 0 | Dữ liệu, split, audit | `src/data.py`, `src/split_v2.py`, `src/audit_v2.py` | cache, 6 file split, audit 15/15 |
| 1 | Baseline + factorial | `src/baseline_v2.py`, `scripts/run_matrix.py` | 48 run F00–F15 |
| 2 | Kiến trúc, hyperparam, augmentation | `scripts/run_matrix.py` | 33 run D/H/G |
| 3 | OOF, freeze, final, lockbox, test | `scripts/run_oof_analysis.py`, `scripts/freeze_v2.py`, `src/final_eval_v2.py` | 27 run OOF + 3 final |
| 4 | Thống kê, hình, báo cáo | `src/analysis_v2.py`, `src/explain_v2.py`, `scripts/report_v3_*.py` | bảng, hình, báo cáo v3 |

### 1.4. Bước cũ chỉ nhắc để tránh nhầm

Trước v2, dự án dùng `split_random`: chia ngẫu nhiên từng ảnh. Cách này khiến
95,22% ảnh validation có frame train cách ±1 trong cùng track, nên val gần như
đã “nhìn thấy” train. Các run E1–E7 và con số val ~99,9% **không dùng** trong
báo cáo v3. Chúng chỉ nằm ở phụ lục lịch sử.

### 1.5. Một ảnh đi qua hệ thống

```text
PPM gốc (kích thước bất kỳ)
  → convert RGB
  → crop ROI (nếu có)
  → resize 64×64 bilinear
  → lưu cache uint8
  → chuẩn hóa (x/255 − mean)/std
  → tensor (3,64,64)
  → SmallCNN
  → logits (43)
  → softmax
  → top-3 + confidence
```

### 1.6. Một run đi qua hệ thống

```text
config.yaml
  → set_seed(seed)
  → load split (screening/fold/dev_all)
  → build model
  → AdamW + CosineAnnealing + AMP
  → vòng epoch: train → eval → early stop
  → lưu best.pt theo val macro-F1
  → ghi metrics.csv, meta.json, predictions_val.csv
  → summary.csv tổng hợp 111 run
```

### 1.7. Khái niệm cốt lõi trong một bảng

| Khái niệm | Hiểu đơn giản |
|---|---|
| CNN | Nhiều “kính lúp” nhỏ quét ảnh để học viền, màu, hình dạng |
| Convolution | Phép nhân cục bộ giữa bộ lọc và vùng ảnh |
| Batch Normalization | Chỉnh lại thang đo đầu ra để train ổn định |
| ReLU | Giữ tín hiệu dương, bỏ tín hiệu âm |
| Max Pooling | Thu nhỏ ảnh, giữ đặc trưng mạnh nhất |
| Flatten | Trải feature map thành vector |
| Fully Connected | Kết hợp đặc trưng thành điểm 43 lớp |
| Dropout | Tắt ngẫu nhiên neuron khi train để giảm học thuộc |
| Logits | Điểm thô trước softmax |
| Softmax | Biến điểm thô thành xác suất cộng lại bằng 1 |
| Loss | Điểm phạt giữa dự đoán và nhãn thật |
| Optimizer | Cách chỉnh trọng số sau mỗi batch |
| Learning rate | Độ lớn bước chỉnh |
| Epoch | Một vòng duyệt toàn bộ train |
| Batch | Nhóm ảnh mỗi lần cập nhật |
| Overfit | Học thuộc đề cũ, làm đề mới kém |
| Augmentation | Tạo thêm ảnh biến thể để học khó hơn |
| Macro-F1 | Điểm trung bình mỗi lớp, không ưu tiên lớp nhiều ảnh |
| OOF | Thi thử với đề không có trong bài ôn |
| Lockbox | Đề thi cuối kỳ, chỉ mở một lần |
| Calibration | Độ tin cậy có khớp với độ đúng không |
| Grad-CAM | Bản đồ nhiệt cho biết mô hình nhìn vào đâu |
| CI 95% | Khoảng bất định của một ước lượng |
| Bootstrap | Rút mẫu lặp lại để ước lượng khoảng bất định |

---

## 2. Dữ liệu: từ ảnh thô đến cache 64×64

### 2.1. Dữ liệu gốc có gì?

GTSRB có:

- 39.209 ảnh train, 43 lớp, mỗi ảnh có nhãn và ROI (vùng biển).
- 12.630 ảnh test, nhãn trong `GT-final_test.csv`.
- Ảnh là PPM, kích thước thay đổi (25–243 px), không đồng nhất.

Vì ảnh không đồng nhất, bước đầu tiên là **chuẩn hóa kích thước** và **chuẩn hóa
màu**.

### 2.2. Vì sao 64×64?

- Ảnh gốc trung vị chỉ 43×43, nên 64×64 không làm mất nhiều chi tiết.
- 64×64 giữ chi phí tính toán thấp: 1 ảnh chỉ 12.288 số, train nhanh.
- Thử 96×96 (I2) cho kết quả **kém hơn** và params tăng 2,2 lần, nên chọn 64.

### 2.3. Vì sao RGB?

Biển báo dùng màu để phân biệt: đỏ = cấm/nguy hiểm, xanh = chỉ dẫn, vàng = ưu
tiên. Nếu chuyển xám sẽ mất tín hiệu quan trọng.

### 2.4. Vì sao cache?

Đọc và resize 51.839 ảnh PPM mỗi epoch rất chậm. Dự án đọc một lần, resize, lưu
`uint8` vào `.npy`:

```text
X_train.npy: (39201, 64, 64, 3) uint8
y_train.npy: (39201,) int
X_test.npy:  (12630, 64, 64, 3) uint8
```

`uint8` tốn 1 byte/số thay vì 4 byte `float32`, tiết kiệm 4 lần bộ nhớ.

### 2.5. Code chính: `src/data.py`

| Hàm | Việc làm |
|---|---|
| `index_train_images` | Liệt kê ảnh train + nhãn từ tên thư mục |
| `read_test_annotations` | Đọc nhãn test từ CSV |
| `scan_split` | Đọc ROI, kích thước, kiểm tra ảnh lỗi |
| `build_cache` | Resize toàn bộ ảnh → `.npy` |
| `load_cache` | Nạp cache vào RAM |
| `preprocess` | Chuẩn hóa một ảnh thành tensor |
| `make_loaders` | Tạo DataLoader cho train/val |
| `GTSRBTensorDataset` | Dataset trả `(tensor, label)` |

**Hàm quan trọng nhất — `preprocess`:**

```python
def preprocess(image, size=64, mean=None, std=None):
    array = _to_rgb_array(image, size)          # RGB + resize 64
    return _array_to_tensor(array, mean, std)   # /255, trừ mean, chia std
```

Điểm cần nhớ: **train, eval và demo dùng chung đúng một hàm này**. Nếu mỗi nơi
viết một kiểu, demo sẽ sai dù test cao.

### 2.6. Vì sao mean/std lấy theo từng split?

Nếu lấy mean/std toàn bộ dữ liệu, thông tin của val/test lẫn vào train — một dạng
rò rỉ nhẹ. Dự án tính mean/std **chỉ trên train của split đang dùng** và lưu vào
`manifest.mean_std`.

### 2.7. Code chính: `src/data_v2.py`

| Hàm | Việc làm |
|---|---|
| `load_manifest` | Đọc mô tả split + hash |
| `get_indices` | Trả chỉ số train/val của screening, fold, dev, lockbox |
| `get_mean_std` | Lấy mean/std đúng split |
| `load_images` | Nạp cache 64 (hoặc 96) |
| `V2Dataset` | Dataset uint8 → tensor, có augmentation |
| `make_loaders_v2` | Tạo train/val loader theo config |
| `build_augmentation` | Ghép affine/color/perspective theo profile |

### 2.8. Augmentation: làm gì và vì sao?

Augmentation tạo ảnh biến thể từ ảnh gốc **chỉ khi train**:

- **Affine:** xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5.
- **Color:** đổi brightness/contrast/saturation 0,2, hue 0,02.
- **Perspective:** méo phối cảnh nhẹ, p=0,3.

Mục đích: giúp mô hình bớt học thuộc vị trí/ánh sáng cụ thể. Nhưng augmentation
không phải lúc nào cũng tốt — kết quả v3 cho thấy “all” không có lợi.

**Tuyệt đối không** lật ngang/dọc hoặc xoay 90/180: biển báo đổi nghĩa (ví dụ
“rẽ trái” thành “rẽ phải”). Code có `_assert_no_flip` để chặn.

### 2.9. Bạn cần nhớ gì ở chương này?

1. Ảnh được chuẩn hóa về 64×64 RGB, cache `uint8`.
2. `preprocess` là hàm duy nhất dùng cho train/eval/demo.
3. mean/std tính trên train của từng split.
4. Augmentation chỉ cho train; cấm lật ảnh.

---

## 3. Chia dữ liệu: track, sign-group và 3 tầng

### 3.1. Vấn đề: ảnh trong cùng một track gần như trùng nhau

GTSRB được cắt từ video. Mỗi **track** là một đoạn video ~30 frame của **cùng một
biển**. Frame thứ 1 và frame thứ 2 gần như giống hệt.

Nếu chia ngẫu nhiên từng ảnh:

- Train có frame 1–24, val có frame 25–30 của **cùng một biển**.
- Val không còn là “đề mới”; nó gần như đã nằm trong train.

Số liệu thật:

| Chỉ số | Val random |
|---|---:|
| % frame val có frame train cách ±1 | **95,22%** |
| Track train/val trùng | **1.306 / 1.307** |
| 1-NN (16×16 xám, không học) | **90,5%** |

1-NN đạt 90,5% mà không học gì, chứng tỏ val random bị “giải sẵn”.

### 3.2. Track và sign-group là gì?

- **Track:** một đoạn video của cùng một biển, ~30 frame. Tên file train có dạng
  `<track>_<frame>.ppm`.
- **Sign-group:** nhóm các track được gộp vì ảnh gần trùng (MAD < 5). Hai track
  khác nhau vẫn có thể là cùng một biển quay ở hai đoạn.

Đơn vị chia split là **sign-group**, không phải ảnh.

### 3.3. Cách dựng sign-group: `src/split_v2.py`

```text
1. Tính thumbnail xám 16×16 cho từng ảnh.
2. Trong mỗi lớp, so từng cặp khác track; giữ cặp MAD thumbnail < 8.
3. Kiểm chứng lại trên ảnh 64×64; nếu MAD < 5 thì nối.
4. Union-find gộp thành sign-group.
5. Frame trong cùng track luôn được gộp.
```

Các hàm chính:

| Hàm | Việc làm |
|---|---|
| `_thumb` | Ảnh 64×64 → thumbnail xám 16×16 |
| `_hash_values` | aHash/dHash để lọc ứng viên nhanh |
| `_candidate_pairs` | Sinh cặp ứng viên gần trùng |
| `build_sign_groups` | Union-find, trả nhóm + cặp đã gộp |
| `name_groups` | Đặt tên `<label>_g<k>` |

Kết quả: 1.307 track → **1.183 sign-group**, gộp 39.603 cặp. Không xóa ảnh nào.

### 3.4. Chia 3 tầng

```text
Pool 39.201
  ├── Lockbox 8.302 (20% sign-group mỗi lớp, seed 2024)
  └── Dev 30.899
       ├── Screening train 24.719 / val 6.180 (80/20, seed 42)
       └── OOF: chia sign-group của dev thành 3 fold
```

Vì sao 3 tầng?

- **Screening:** dùng để chọn cấu hình nhanh, không phải kết luận cuối.
- **OOF:** mỗi ảnh dev được dự đoán bởi mô hình không train trên fold chứa nó.
- **Lockbox:** giữ riêng, chỉ mở một lần sau khi chốt mô hình.

Nếu dùng chính tập chọn để báo cáo, con số sẽ bị lạc quan. Lockbox là “đề thi
chưa từng xem”.

### 3.5. Code chia nhóm: `_pick_groups` và `_assign_folds`

```python
# Mỗi lớp: chọn round(ratio * số nhóm) nhóm, tối thiểu 1, chừa ít nhất 1
k = min(max(1, round(ratio * len(groups))), len(groups) - 1)
chosen = rng.choice(groups, size=k, replace=False)
```

```python
# Chia fold: nhóm lớn xếp trước, gán vào fold đang ít frame nhất
for group in sorted(groups, key=lambda g: -image_counts[g]):
    fold = argmin(load)
    assignment[group] = fold
    load[fold] += image_counts[group]
```

Cách này giữ 3 fold cân bằng số frame, không chỉ cân bằng số nhóm.

### 3.6. Audit: `src/audit_v2.py`

Audit kiểm tra 15 điều kiện, ví dụ:

- C1: track overlap lockbox/dev = 0.
- C2: sign-group overlap lockbox/dev = 0.
- C3–C6: mỗi tập/fold đủ 43 lớp.
- C7: screening train/val không overlap.
- C9: test không nằm trong train/lockbox.
- C10: lockbox không có ảnh gần trùng train.
- HASH: hash 6 file split khớp manifest.
- MEANSTD: mean/std khớp lại từ cache.

Kết quả: **15/15 PASS**. Đây là bằng chứng split sạch, không phải lời tuyên bố.

### 3.7. Bạn cần nhớ gì ở chương này?

1. Track = một biển; frame trong track gần như trùng.
2. Chia theo ảnh sẽ rò rỉ; chia theo sign-group mới sạch.
3. Screening để chọn; OOF để xác nhận; lockbox để chốt.
4. Audit 15/15 PASS là bằng chứng.

---

## 4. Mô hình: SmallCNN

### 4.1. Một CNN nhìn ảnh như thế nào?

Hãy tưởng tượng bạn soi ảnh bằng nhiều kính lúp nhỏ:

- Mỗi kính lúp (bộ lọc 3×3) tìm một mẫu: viền đỏ, vòng tròn, mũi tên…
- Kết quả là một **feature map**: bản đồ “chỗ nào giống mẫu đó”.
- Càng vào sâu, mô hình càng học đặc trưng phức tạp: từ viền → hình tròn →
  toàn bộ biển.

### 4.2. Các lớp trong `src/models.py`

```python
class SmallCNN(nn.Module):
    def __init__(widths=(32,64,128), use_bn=True, dropout=0.3,
                 gap=False, fc_hidden=256, num_classes=43, in_size=64):
        ...
```

Mỗi khối conv:

```text
Conv2d(3×3, padding=1)
  → BatchNorm2d (nếu bật)
  → ReLU
  → MaxPool2d(2)
```

Sau các khối:

- Nếu `gap=True`: Global Average Pooling → Dropout → FC 43.
- Nếu `gap=False`: Flatten → Dropout → FC 256 → ReLU → Dropout → FC 43.

### 4.3. Vì sao từng thành phần?

| Thành phần | Vì sao có |
|---|---|
| Conv 3×3 | Học đặc trưng cục bộ, ít tham số hơn FC |
| Padding 1 | Giữ kích thước không gian sau conv |
| BatchNorm | Ổn định phân phối đầu ra, train nhanh/ổn hơn |
| ReLU | Phi tuyến, rẻ tính toán |
| MaxPool 2×2 | Giảm kích thước, tăng vùng nhìn |
| Flatten | Chuyển feature map thành vector |
| FC 256 | Kết hợp đặc trưng toàn cục |
| Dropout 0,3 | Giảm overfit |
| FC 43 | Đầu ra logits cho 43 lớp |
| Không softmax trong model | Để dùng `CrossEntropyLoss` trực tiếp |

### 4.4. Kiến trúc D2 — mô hình thắng

```text
Input 3×64×64
→ Conv 3→32 + BN + ReLU + Pool   → 32×32×32
→ Conv 32→64 + BN + ReLU + Pool  → 64×16×16
→ Conv 64→128 + BN + ReLU + Pool → 128×8×8
→ Conv 128→256 + BN + ReLU + Pool→ 256×4×4
→ Flatten 4.096
→ FC 4.096→256 + ReLU + Dropout 0,3
→ FC 256→43
→ logits
```

Số tham số: **1.448.779**. Checkpoint: **5,55 MB**.

### 4.5. Vì sao không dùng pretrained?

Đề bài yêu cầu CNN tự xây dựng. Pretrained (ResNet, MobileNet) sẽ dễ đạt điểm
cao hơn nhưng không chứng minh được năng lực thiết kế mô hình. Báo cáo ghi rõ
đây là hạn chế, không phải “mô hình tốt nhất có thể”.

### 4.6. Bạn cần nhớ gì ở chương này?

1. CNN = nhiều khối conv + pool, rồi FC.
2. BN và dropout là hai thành phần quan trọng nhất của D2.
3. Model trả **logits**, không softmax.
4. D2 có 1,45M tham số, checkpoint 5,55 MB.

---

## 5. Huấn luyện: một run chạy ra sao?

### 5.1. Đầu vào của một run

Một run được mô tả bằng file YAML, ví dụ `configs/v2/d2.yaml`:

```yaml
experiment:
  id: d2
  seed: 42
  block: D_architecture
data:
  split: screening
  image_size: 64
  num_classes: 43
model:
  widths: [32, 64, 128, 256]
  use_bn: true
  dropout: 0.3
  gap: false
  fc_hidden: 256
augmentation:
  enabled: true
  profile: all
train:
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 128
  max_epochs: 40
  patience: 7
```

Bạn đọc file này như một “công thức”: dữ liệu nào, model nào, train bao lâu.

### 5.2. Hàm trung tâm: `run_training_v2`

Nằm trong `src/train_v2.py`. Trình tự:

```text
1. load YAML
2. set_seed(seed)                  # tái lập
3. resolve split (screening/fold/dev_all)
4. make_loaders_v2(...)            # train/val loader
5. build_model(cfg)                # SmallCNN
6. build_criterion(...)            # CrossEntropy (+ class weight)
7. AdamW + CosineAnnealingLR
8. vòng epoch:
     train_epoch → evaluate_epoch → early stopping → lưu best.pt
9. nạp best.pt → dự đoán val → predictions_val.csv
10. đo latency → vẽ curves → ghi meta.json
```

### 5.3. Một epoch train: `train_epoch`

```python
model.train()
for inputs, targets in loader:
    optimizer.zero_grad()
    with autocast(enabled=use_amp):
        logits = model(inputs)
        loss = criterion(logits, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Giải thích từng dòng:

- `model.train()`: bật dropout/BN ở chế độ train.
- `zero_grad()`: xóa gradient cũ.
- `forward`: tính logits.
- `loss`: đo sai số.
- `backward`: tính gradient.
- `step`: chỉnh trọng số.
- `autocast` + `GradScaler`: AMP (FP16) để nhanh hơn trên GPU.

**Vì sao AMP chỉ bật khi CUDA?** FP16 trên CPU không được hỗ trợ tốt; bật sẽ lỗi
hoặc chậm. Code kiểm tra `device.type == "cuda"`.

### 5.4. Một epoch đánh giá: `evaluate_epoch`

```python
model.eval()
with torch.no_grad():
    for inputs, targets in loader:
        logits = model(inputs)
        ...
accuracy = ...
macro_f1 = f1_score(..., average="macro", labels=range(43), zero_division=0)
```

**Vì sao `model.eval()` và `no_grad()`?** `eval()` tắt dropout, dùng running
stats của BN; `no_grad()` không tính gradient, tiết kiệm bộ nhớ và nhanh hơn.
Quên một trong hai sẽ cho val không ổn định.

### 5.5. Vì sao checkpoint chọn theo macro-F1, không theo accuracy?

Ví dụ: lớp 0 chỉ có 30 ảnh, lớp 2 có 450 ảnh trong lockbox. Nếu mô hình bỏ lớp 0
nhưng đoán tốt lớp 2, accuracy vẫn cao. Macro-F1 cho mỗi lớp trọng số 1/43, nên
lớp hiếm được tính công bằng hơn.

Code:

```python
if val_f1 > best_f1:
    best_f1, best_epoch = val_f1, epoch
    _save_checkpoint(...)
else:
    bad += 1
    if bad >= patience:
        break
```

### 5.6. Checkpoint chứa gì?

`best.pt` là một dict, không phải cả object model:

```python
{
  "state_dict": model.state_dict(),
  "config": {...},
  "class_names": {...},
  "epoch": 26,
  "val_macro_f1": 0.9645,
  "git_commit": "...",
  "torch_version": "...",
  "config_sha256": "...",
  "split_sha256": "...",
}
```

**Vì sao không lưu cả object?** Object Python phụ thuộc phiên bản thư viện; lưu
`state_dict` + config an toàn hơn, dễ load lại.

### 5.7. Ghi log từng epoch

`metrics.csv` có đúng thứ tự cột:

```text
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

Mỗi epoch ghi một dòng và `flush()` ngay. Nếu máy tắt giữa chừng, dữ liệu vẫn còn.

### 5.8. Vì sao `meta.json` quan trọng?

`meta.json` lưu:

- `run_id`, `config_id`, `seed`, `fold`, `split_name`.
- `params`, `epochs_run`, `best_epoch`, `best_val_macro_f1`.
- `train_time_s`, `infer_ms_per_img`, `gpu_mem_mb`.
- `git_commit`, `config_sha256`, `split_sha256`, `manifest_sha256`.
- `mean`, `std`, `device`, `amp`, `num_workers`.

Nhờ vậy, một run có thể được kiểm tra lại mà không cần nhớ bạn đã chạy gì.

### 5.9. Các lỗi thường gặp khi train

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| Quên `model.eval()` | Val sai/không ổn định | Luôn eval + no_grad |
| Quên `set_seed` | Kết quả khác giữa các lần | Gọi ở đầu entrypoint |
| Dùng accuracy để chọn checkpoint | Bỏ lớp hiếm | Dùng val macro-F1 |
| Bật AMP trên CPU | Lỗi/chậm | Chỉ bật khi CUDA |
| `num_workers>0` trên Windows không có `if __name__` | Treo | Dùng main guard |
| Ghi log cuối run | Mất dữ liệu khi gián đoạn | Flush từng epoch |

### 5.10. Bạn cần nhớ gì ở chương này?

1. Một run = YAML + seed + split + model + train loop.
2. Checkpoint chọn theo **val macro-F1**.
3. `eval()` + `no_grad()` là bắt buộc khi đánh giá.
4. Mọi thứ được ghi lại trong `metrics.csv`, `meta.json`, `best.pt`.

---

## 6. Chạy 111 run: config, matrix và tổng hợp

### 6.1. Config được sinh từ đâu?

`scripts/gen_configs_v2.py` đọc `docs/kehoach/1/MA_TRAN_RUN.csv` và sinh
`configs/v2/*.yaml`. Mỗi dòng trong ma trận là một cấu hình:

- `f00`–`f15`: factorial 2⁴.
- `d1`, `d2`, `d3`, `w1`, `w2`: kiến trúc.
- `l1`, `cw1`, `i2`: hyperparameter.
- `g1`, `g2`, `g3`: augmentation.
- `x1`, `x2`, `x3`: OOF.
- `final`: train trên dev.

### 6.2. `scripts/run_matrix.py` làm gì?

```text
discover(block/configs) → danh sách YAML
for mỗi YAML × seed × fold:
    nếu run đã có meta.json → SKIP
    nếu chưa → run_training_v2(...)
    ghi log ok/fail
```

Điểm hay:

- **Idempotent:** chạy lại không train trùng, vì đã có `meta.json` thì bỏ qua.
- **Tiếp tục sau lỗi:** một run fail không làm dừng cả ma trận.
- **Log rõ:** `run_matrix_log.csv` ghi thời gian, trạng thái, lỗi.

Ví dụ lệnh:

```powershell
python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024
python scripts/run_matrix.py --configs d1,d2,d3,w1,w2 --seeds 42,123,2024
python scripts/run_matrix.py --configs x1,x2,x3 --seeds 42,123,2024 --folds 0,1,2
```

### 6.3. Vì sao 3 seed?

Mỗi seed thay đổi thứ tự dữ liệu và khởi tạo trọng số. Nếu chỉ chạy 1 seed, bạn
không biết kết quả tốt do mô hình hay do “may mắn”. 3 seed cho phép báo mean ± sd
và kiểm tra độ ổn định.

### 6.4. `src/summarize_v2.py` làm gì?

Nó quét `outputs/runs/v2_*`, đọc `meta.json`, ghép thành `summary.csv` với 111
dòng. Mỗi dòng có:

```text
protocol, block, config_id, run_id, seed, fold, split, model, params,
aug, aug_profile, epochs_run, best_epoch, val_acc, val_macro_f1,
val_macro_f1_ci_low, val_macro_f1_ci_high, test_acc, test_macro_f1,
train_time_s, infer_ms_per_img, git_commit, config_sha256, split_sha256,
valid_for_conclusion, notes
```

Đây là nguồn số cho mọi bảng kết quả. Không có số nào được gõ tay.

### 6.5. Bảng 111 run được tổ chức thế nào?

| Block | Số config | Seed/fold | Run |
|---|---:|---:|---:|
| Baseline | 1 | 1 | 1 |
| Factorial | 16 | 3 | 48 |
| Kiến trúc | 5 | 3 | 15 |
| Hyperparam | 3 | 3 | 9 |
| Augmentation | 3 | 3 | 9 |
| OOF | 3 | 3 fold × 3 seed | 27 |
| Final | 1 | 3 | 3 |
| **Tổng CNN** | | | **111** |

### 6.6. Bạn cần nhớ gì ở chương này?

1. Config-driven: không hard-code hyperparameter trong code.
2. `run_matrix.py` chạy hàng loạt, bỏ qua run đã có.
3. 3 seed để đo độ ổn định.
4. `summary.csv` là nguồn số duy nhất cho báo cáo.

---

## 7. Thống kê: các con số có nghĩa gì?

### 7.1. Vì sao cần khoảng tin cậy (CI)?

Giả sử mô hình A đạt 97,0% và B đạt 96,8%. Chênh 0,2% là thật hay do may mắn?
CI 95% trả lời: nếu lặp lại thí nghiệm nhiều lần, khoảng này chứa giá trị thật
trong ~95% trường hợp.

Nếu CI của hiệu số chứa 0, ta **không đủ bằng chứng** nói A hơn B.

### 7.2. `src/stats_v2.py`

| Hàm | Việc làm |
|---|---|
| `macro_f1_43` | F1 trung bình 43 lớp, lớp vắng = 0 |
| `cluster_bootstrap_f1` | CI cho macro-F1, resample theo **track** |
| `paired_bootstrap_delta` | CI cho Δ giữa 2 model, cùng mẫu track |
| `bootstrap_contrast` | CI cho main effect/interaction factorial |
| `holm` | Hiệu chỉnh p-value khi so nhiều cặp |
| `decision` | Áp luật: tốt hơn/kém hơn/tương đương/chưa kết luận |
| `variance_decomposition` | Phân rã phương sai seed × fold |

### 7.3. Vì sao bootstrap theo track, không theo ảnh?

Ảnh trong cùng track gần như trùng nhau. Nếu resample từng ảnh, bạn sẽ đếm cùng
một biển nhiều lần, làm CI hẹp giả tạo. Resample **track** giữ tính độc lập.

Mã giả:

```text
tracks = unique(track của eval set)
for b in 1..B:
    chọn ngẫu nhiên len(tracks) track (có hoàn lại)
    gom tất cả frame của các track đó
    tính macro-F1
CI = percentile(2.5, 97.5)
```

### 7.4. Vì sao cần Holm?

Nếu so 8 cặp, xác suất có ít nhất một cặp “may mắn” p < 0,05 cao hơn 5%. Holm
hiệu chỉnh p để kiểm soát sai số family-wise.

Cách đọc:

- p Holm < 0,05: còn ý nghĩa sau hiệu chỉnh.
- p Holm ≥ 0,05: chưa đủ bằng chứng, dù CI có thể không chứa 0.

Ví dụ depth: CI [+0,036; +4,440] nhưng p Holm 0,373 → báo “có bằng chứng vừa,
chưa đủ sau hiệu chỉnh”, không kết luận chắc.

### 7.5. `src/analysis_v2.py`

| Hàm | Việc làm |
|---|---|
| `factorial_effects` | Main effect + interaction của BN/DO/WD/Aug |
| `factorial_table` | Bảng 16 dòng F00–F15 |
| `aggregate` | Mean ± sd + Δ so với reference |
| `oof_predictions` | Ghép 3 fold thành OOF cho từng seed |
| `oof_table` | Bảng OOF + so sánh + variance |
| `corrupt_images` | Sinh 6 corruption × 3 mức |
| `corruption_table` | Đánh giá checkpoint trên ảnh corruption |

### 7.6. Factorial 2⁴ được tính thế nào?

Ví dụ main effect BN:

```text
Δ_BN = trung bình F1 của 8 cấu hình BN=on
      − trung bình F1 của 8 cấu hình BN=off
```

Interaction BN×Aug:

```text
Δ = trung bình(BN=on,Aug=on) − trung bình(BN=on,Aug=off)
  − trung bình(BN=off,Aug=on) + trung bình(BN=off,Aug=off)
```

Code dùng `_factorial_weights` để gán trọng số ±1/k, rồi `bootstrap_contrast`
tính CI.

### 7.7. Phân rã phương sai

`variance_decomposition` trả:

- `var_seed`: bao nhiêu phương sai do seed.
- `var_fold`: do fold.
- `var_resid`: phần còn lại.
- `icc_seed`, `icc_fold`: tỷ lệ phương sai do yếu tố đó.

Kết quả v3: **fold 91,0%, seed 0,9%**. Nghĩa là chạy nhiều seed không giúp bằng
chọn fold đúng; báo cáo phải nói rõ fold.

### 7.8. Bạn cần nhớ gì ở chương này?

1. CI cho biết khoảng bất định; CI chứa 0 = chưa kết luận.
2. Bootstrap theo track, không theo ảnh.
3. Holm hiệu chỉnh khi so nhiều cặp.
4. Fold là nguồn biến thiên chính, không phải seed.

---

## 8. Chọn mô hình: OOF → freeze → lockbox → test

### 8.1. Chọn ứng viên: `scripts/select_candidates.py`

```text
1. Đọc summary.csv (chỉ screening).
2. Tính mean macro-F1 3 seed cho từng config.
3. Sắp xếp: F1 cao → ít params → latency thấp → ít aug → ít epoch.
4. Lấy nhóm trong 0.3 điểm % so với tốt nhất.
5. Chọn top2 + control (F07 hoặc F00).
6. Sinh config x1..x3 cho OOF.
```

Điểm quan trọng: **chọn trước khi chạy OOF**, không đổi sau khi xem OOF.

### 8.2. Chạy OOF: `scripts/run_oof_analysis.py`

- Chạy x1/x2/x3 trên 3 fold × 3 seed.
- Ghép 3 fold của mỗi seed → mỗi ảnh dev được dự đoán đúng một lần.
- Kiểm tra không trùng ảnh, đủ 30.899 ảnh.
- Tính macro-F1, CI, so sánh cặp.

Kết quả:

| Config | OOF macro-F1 | CI 95% |
|---|---:|---|
| X1 | 96,2971 | [94,628; 97,020] |
| **X2** | **97,0400** | **[95,051; 97,942]** |
| X3 | 95,0588 | [93,113; 95,930] |

### 8.3. Freeze: `scripts/freeze_v2.py`

Freeze nghĩa là “chốt”:

```text
winner = x2
source = d2
epoch = median best epoch của 9 run OOF = 26
final.yaml = clone d2, đổi split sang dev_all, max_epochs=26
FREEZE.json = {winner, epoch, config_sha256, split_sha256, lockbox_opened=false}
```

Từ đây, **không đổi** config, epoch, ngưỡng. Nếu đổi, phải freeze lại và ghi lý do.

### 8.4. Vì sao epoch cố định 26?

Trong OOF, mỗi fold dừng ở epoch khác nhau. Lấy **median best epoch** của 9 run
làm epoch cho final, để không phụ thuộc một fold may mắn. Final train không có
val, nên không early stop; train đúng 26 epoch.

### 8.5. Lockbox: `src/final_eval_v2.py`

```text
1. _guard_freeze: kiểm tra FREEZE.json tồn tại, config_sha256 khớp.
2. Nếu lockbox_result.json đã tồn tại → từ chối mở lại.
3. Load 8.302 ảnh lockbox.
4. Đánh giá 3 checkpoint final.
5. Tính accuracy, macro-F1, top-3, ECE, Brier, per-class, confusion.
6. Ghi lockbox_result.json + lockbox_access.log.
```

**Vì sao one-shot?** Nếu mở lockbox nhiều lần, bạn sẽ vô tình dùng nó để chọn
mô hình/ngưỡng, làm mất tính độc lập. Một lần là một lần.

### 8.6. Official test: raw và clean

- **Raw:** 12.630 ảnh.
- **Clean:** loại 751 ảnh có ảnh train gần trùng (MAD < 5), còn 11.879.
- Test không có track ID, không track-disjoint → chỉ benchmark phụ.

Kết quả:

| Tập | Accuracy | Macro-F1 |
|---|---:|---:|
| Lockbox | 98,8758 | 97,5981 |
| Test raw | 98,6989 | 98,0398 |
| Test clean | 98,8692 | 98,2499 |

### 8.7. ECE và Brier

- **ECE:** trung bình |accuracy − confidence| theo 15 bin. ECE nhỏ = confidence
  khớp với độ đúng.
- **Brier:** trung bình tổng bình phương xác suất so với one-hot. Càng nhỏ càng tốt.

Lockbox: ECE 0,005328; Brier 0,017862.

### 8.8. Bạn cần nhớ gì ở chương này?

1. Chọn top2 + control trước OOF.
2. OOF chọn winner; freeze chốt config/epoch.
3. Lockbox mở một lần, có guard.
4. Test chỉ benchmark phụ.

---

## 9. Giải thích cơ chế: hình và cách đọc

### 9.1. `src/explain_v2.py` làm gì?

| Hàm | Hình | Đọc gì |
|---|---|---|
| `learning_curves` | loss/acc/F1 theo epoch | Overfit hay underfit |
| `interaction_plot` | main effect + interaction | Yếu tố nào thật sự giúp |
| `pareto_plot` | params vs F1 | Đánh đổi dung lượng/điểm |
| `per_class_heatmap` | ΔF1 từng lớp | Lớp nào được/cải thiện |
| `reliability_plot` | reliability diagram | Confidence có đáng tin |
| `corruption_curves` | accuracy theo mức corruption | Yếu với loại nào |
| `confusion_from_predictions` | confusion matrix | Cặp nào nhầm |
| `_gradcam` | CAM cho 1 ảnh | Mô hình nhìn đâu |
| `gradcam_grid` | 12 ảnh × 2 model | So sánh cơ chế |
| `error_taxonomy` | lưới ảnh lỗi | Nhóm lỗi heuristic |

### 9.2. Grad-CAM hoạt động thế nào?

```text
1. Forward ảnh, lấy feature map của lớp conv cuối.
2. Backward từ logits của lớp dự đoán.
3. Lấy gradient trung bình theo không gian làm trọng số.
4. Nhân trọng số với feature map, ReLU, resize về 64×64.
5. Chuẩn hóa 0–1 → bản đồ nhiệt.
```

Nếu CAM tập trung trong ROI của biển, mô hình đang nhìn đúng chỗ. Nếu tập trung
ở nền, mô hình đang dựa vào chi tiết không phù hợp.

Kết quả v3: mass trong ROI **0,853 khi đúng**, **0,733 khi sai**, mức ngẫu nhiên
0,543. Ca sai vẫn nhìn vào biển, nên lỗi chủ yếu là phân biệt lớp.

### 9.3. Calibration

- Reliability diagram: trục x là confidence, trục y là accuracy thực tế.
- Đường chéo = hoàn hảo.
- Nếu điểm nằm dưới đường chéo: mô hình tự tin quá mức.
- ECE là khoảng cách trung bình; Brier là sai số xác suất.

### 9.4. Corruption

6 loại: blur, noise, brightness, contrast, JPEG, occlusion. Mỗi loại 3 mức.
Kết quả: occlusion là điểm yếu lớn nhất (17,63% accuracy trung bình).

### 9.5. Report pack v3

| Script | Việc làm |
|---|---|
| `scripts/report_v3_data.py` | Sinh 43 bảng, xác suất screening, chất lượng dữ liệu, benchmark, Grad-CAM, phiếu lỗi |
| `scripts/report_v3_figures.py` | Sinh 20 hình |
| `scripts/report_v3_meta.py` | Hardware, deployment, lockbox uncertainty, đính chính, glossary |
| `scripts/report_v3_manifest.py` | Manifest sha256 |

Các script này **không train**, chỉ đọc artifact v2 và đo mới trong phạm vi suy
luận. Đây là lý do báo cáo v3 không cần chạy lại 111 run.

### 9.6. Bạn cần nhớ gì ở chương này?

1. Hình không phải trang trí; mỗi hình trả lời một câu hỏi.
2. Grad-CAM cho biết mô hình nhìn đâu.
3. Calibration cho biết confidence có đáng tin.
4. Report pack v3 chỉ đọc artifact, không train lại.

---

## 10. Demo: frontend, backend và cách nối

### 10.1. Kiến trúc tổng thể

Demo không tách thành React + REST API. Nó dùng **Streamlit**: một framework vừa
làm giao diện, vừa làm server, vừa giữ session.

```text
Trình duyệt (HTML/JS do Streamlit sinh)
        │  WebSocket (không phải REST)
        ▼
Streamlit server (app.py)
        │  gọi hàm Python trực tiếp
        ▼
Backend dự án
  src.predict.Predictor
    → src.data.preprocess
    → SmallCNN (best.pt)
    → softmax + top-k
        │  trả dict Python
        ▼
app.py render: st.success, st.progress, st.metric, st.dataframe
```

Điểm cần hiểu:

- **Không có API JSON riêng.** Streamlit tự đóng gói lời gọi hàm và kết quả qua
  WebSocket. Vì vậy “frontend gọi backend” ở đây là **app.py gọi trực tiếp**
  `Predictor.predict`.
- **Không có hai process.** `app.py` chạy trong cùng process với model.
- **Session state** do Streamlit giữ; mỗi tab trình duyệt là một session.

### 10.2. Backend: `src/predict.py`

| Hàm | Vai trò | Ghi chú |
|---|---|---|
| `load_model` | Đọc `best.pt`, build model, load state_dict, eval | Không pickle object |
| `load_tau` | Đọc ngưỡng τ từ JSON | τ v2 = 0,9590 |
| `should_reject` | `max_prob < tau` → “không xác định” | Theo confidence, không phải OOD |
| `validate_image_bytes` | Chặn file lỗi, sai đuôi, quá 20 MB | Trả PIL Image |
| `predict_topk` | `preprocess` → forward → softmax → top-k | Dùng đúng preprocess của train |
| `Predictor` | Gói model + config + class_names | `@st.cache_resource` nạp một lần |

Một điểm vừa được sửa: `predict_topk` đọc mean/std từ `cfg["data"]` (v1) **hoặc**
`cfg["_resolved"]` (v2). Nếu không, checkpoint v2 sẽ báo `KeyError: 'mean'`.

### 10.3. Frontend: `app.py`

| Thành phần | Streamlit | Vai trò |
|---|---|---|
| Tiêu đề, cảnh báo | `st.title`, `st.info` | Nói rõ phạm vi 43 lớp Đức |
| Sidebar | `st.sidebar`, `st.metric` | Lockbox accuracy/F1/top-3, τ |
| Upload | `st.file_uploader` | Nhận JPG/PNG ≤ 20 MB |
| Ảnh mẫu | `st.selectbox` | Chọn ảnh GTSRB/Hàn Quốc |
| Kết quả | `st.success`, `st.warning`, `st.progress` | Top-3 + confidence |
| Tab | `st.tabs` | “Dự đoán” và “Về mô hình” |
| Bảng số | `st.dataframe` | Lockbox metrics |

**Cách nối cụ thể:**

```python
# app.py
from src.predict import Predictor, load_tau, validate_image_bytes

predictor = get_predictor(str(RUN_DIR), tau)   # @st.cache_resource
image = validate_image_bytes(uploaded.name, uploaded.getvalue())
prediction = predictor.predict(image, k=3, tau=tau)
render_prediction(prediction, tau)
```

`Predictor.predict` trả dict:

```python
{
  "topk": [{"class_id": 16, "name_vi": "...", "prob": 0.99}, ...],
  "max_prob": 0.99,
  "margin": 0.98,
  "unknown": False,
  "tau": 0.9590,
}
```

`render_prediction` chỉ đọc dict này và vẽ. **Không có logic model trong app.py**;
app.py không biết convolution, BN hay softmax.

### 10.4. Một lần upload đi qua những bước nào?

```text
1. Người dùng chọn file
2. Streamlit gửi bytes qua WebSocket
3. app.py: validate_image_bytes
     - > 20 MB? báo lỗi
     - sai đuôi? báo lỗi
     - ảnh hỏng? báo lỗi
4. app.py: predictor.predict(image, k=3, tau)
5. src.predict: preprocess 64×64 + normalize
6. SmallCNN: forward → logits 43
7. softmax → top-3
8. nếu max_prob < τ: unknown = True
9. app.py: render top-3, progress bar, margin
```

### 10.5. Mô hình demo đang dùng

Đã cập nhật sang **D2/v2**:

| Thành phần | Giá trị |
|---|---|
| Checkpoint | `outputs/runs/v2_final_s42/best.pt` |
| Kiến trúc | D2, 4 khối 32/64/128/256, 1.448.779 params |
| Ngưỡng τ | 0,9590 (OOF X2) |
| Sidebar | `app_summary_v2.json` (lockbox 8.302 ảnh, 3 seed) |
| Accuracy/F1 | 98,88% / 0,9760 (lockbox) |

`app.py` tìm theo thứ tự: `GTSRB_RUN_DIR` → `outputs/runs/v2_final_s42` →
`checkpoints/final_s42` (legacy fallback).

### 10.6. Đã kiểm thử gì?

- **Playwright + Chromium**, chạy thật Streamlit ở cổng 8765.
- **9/9 PASS**: trang chủ, upload ảnh GTSRB, biển Hàn Quốc, ảnh không biển,
  file hỏng, file > 20 MB, tab “Về mô hình”, chọn ảnh mẫu, thiếu checkpoint.
- Ảnh chụp: `outputs/figures/11_ui_*.png`.
- Kết quả máy đọc: `outputs/results/e2e_ui.json`.
- UI latency (từ `set_input_files` đến khi hiện kết quả): ~0,39 s.

### 10.7. Hạn chế của demo

- Chỉ 1 ảnh mỗi lần; không batch.
- Không có API cho hệ thống khác.
- Không lưu lịch sử dự đoán.
- Golden test 20 ảnh vẫn là checkpoint legacy; chưa chạy lại cho v2.
- Chưa đo trên máy cấu hình thấp, chưa camera thật.
- τ là ngưỡng confidence, **không phải bộ phát hiện OOD**.

### 10.8. Bạn cần nhớ gì ở chương này?

1. Streamlit vừa là FE vừa là server; giao tiếp qua WebSocket, không REST.
2. `app.py` gọi trực tiếp `Predictor.predict`; không có logic model trong FE.
3. `preprocess` dùng chung với train.
4. Demo đã dùng D2/v2 + τ 0,9590; e2e 9/9 PASS.
5. Hạn chế: 1 ảnh, không API, không lịch sử, chưa camera/máy yếu.

---

## 11. Tái lập và nghiệm thu

### 11.1. Tái lập là gì?

Cùng seed, cùng config, cùng split → chạy lại phải cho **cùng kết quả**. Nếu
khác, kết quả không đáng tin.

### 11.2. `scripts/repro_check_v2.py`

Chạy 2 lần, mỗi lần 3 epoch, cùng seed. So sánh:

| Chỉ số | Sai khác |
|---|---:|
| train_loss | 0,0 |
| val_loss | 0,0 |
| val_acc | 0,0 |
| val_macro_f1 | 0,0 |

Kết quả: **tái lập tuyệt đối** trong cùng môi trường.

### 11.3. Kiểm thử tự động

`pytest` chạy 22 test:

- `tests/test_split_v2.py`: kiểm tra split, overlap, hash.
- `tests/test_stats_v2.py`: kiểm tra macro-F1, bootstrap, Holm.
- `tests/test_config_v2.py`: kiểm tra config, ràng buộc.

Kết quả: **22 passed**.

### 11.4. Nghiệm thu: `scripts/acceptance_v2.py`

12 điều kiện:

| # | Điều kiện |
|---:|---|
| 1 | Lockbox 0 overlap, one-shot |
| 2 | Mọi run kết luận dùng split sign-group |
| 3 | Mỗi kết luận có Δ + CI + luật |
| 4 | Factorial đủ 16 config |
| 5 | Tách GAP/depth/width |
| 6 | Aug bật/tắt + corruption |
| 7 | OOF 3 fold × 3 seed |
| 8 | Decision card 10 yếu tố |
| 9 | Không dùng số random cũ |
| 10 | Không tune lockbox/test |
| 11 | Repo sạch |
| 12 | Tái lập cùng seed |

Kết quả: **12/12**. Máy đọc: `outputs/results/v2/ACCEPTANCE.json`.

### 11.5. Hash và seed

- `config_sha256`: hash file YAML.
- `split_sha256`: hash file split.
- `manifest_sha256`: hash manifest.
- `git_commit`: commit lúc chạy.
- `torch_version`: phiên bản PyTorch.

Nhờ vậy, mỗi kết quả truy được về đúng mã nguồn và dữ liệu.

### 11.6. Bạn cần nhớ gì ở chương này?

1. Tái lập = cùng seed/config/split → cùng kết quả.
2. 22 test + 12/12 nghiệm thu.
3. Hash giúp truy vết.
4. Không có “tái lập tuyệt đối” trên máy/GPU khác nếu không kiểm tra lại.

---

## 12. Runbook: từ máy trắng đến báo cáo

### 12.1. Cài môi trường

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 12.2. Dữ liệu và split

```powershell
python -m src.data --download --report
python -m src.split_v2 --build
python -m src.audit_v2
```

Kỳ vọng:

- Cache `data/cache/X_train.npy` 39.201 ảnh.
- `outputs/splits/v2/` có 6 file split + manifest.
- Audit 15/15 PASS.

### 12.3. Baseline và factorial

```powershell
python -m src.baseline_v2
python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024
```

### 12.4. Kiến trúc, hyperparam, augmentation

```powershell
python scripts/run_matrix.py --configs d1,d2,d3,w1,w2 --seeds 42,123,2024
python scripts/run_matrix.py --configs l1,cw1,i2 --seeds 42,123,2024
python scripts/run_matrix.py --configs g1,g2,g3 --seeds 42,123,2024
```

### 12.5. Tổng hợp, chọn ứng viên, OOF

```powershell
python -m src.summarize_v2
python scripts/select_candidates.py
python scripts/run_matrix.py --configs x1,x2,x3 --seeds 42,123,2024 --folds 0,1,2
python scripts/run_oof_analysis.py --configs x1,x2,x3
```

### 12.6. Freeze, final, lockbox, test

```powershell
python scripts/freeze_v2.py --winner x2 --source d2 --reason "..."
python scripts/run_matrix.py --configs final --seeds 42,123,2024
python -m src.final_eval_v2 --mode lockbox --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
python -m src.final_eval_v2 --mode test --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
```

**Cảnh báo:** lệnh lockbox chỉ chạy **một lần**. Nếu có bug, phải dùng
`--allow-rerun --reason "..."` và ghi lý do.

### 12.7. Phân tích, hình, nghiệm thu

```powershell
python scripts/run_phase4.py
python scripts/acceptance_v2.py
```

### 12.8. Report pack v3 và báo cáo

```powershell
python scripts/report_v3_data.py --only all
python scripts/report_v3_figures.py --only all
python scripts/report_v3_meta.py
python scripts/report_v3_manifest.py
python docs/bao_cao/md_to_docx_v3.py
```

### 12.9. Thời gian tham khảo

| Việc | Thời gian |
|---|---|
| 111 run CNN | ~4,0 giờ epoch-loop |
| Baseline | ~8 phút |
| Report pack v3 | ~3–5 phút |
| Tổng dự án (ước tính) | ~4,5–5 giờ máy |

---

## 13. Đọc kết quả: bốn tập, tám đính chính, năm câu hỏi

### 13.1. Bốn tập đánh giá

| Tập | n | Dùng để | Kết luận? |
|---|---:|---|---|
| Screening val | 6.180 | Chọn cấu hình | Không |
| OOF | 30.899 | Chọn winner | Có |
| Lockbox | 8.302 | Xác nhận độc lập | Có |
| Official test | 12.630 / 11.879 | Benchmark phụ | Không |

**Không so sánh trực tiếp** các con số giữa bốn tập.

### 13.2. Kết quả chốt

| Chỉ số | Giá trị |
|---|---:|
| OOF accuracy | 97,9244% |
| OOF macro-F1 | 97,0400% ± 0,1869 |
| Lockbox accuracy | 98,8758% ± 0,1010 |
| Lockbox macro-F1 | 97,5981% ± 0,6246 |
| Test raw | 98,6989% / 98,0398% |
| Test clean | 98,8692% / 98,2499% |

### 13.3. Tám đính chính quan trọng

| Mục | Cũ | Đúng |
|---|---|---|
| OOF accuracy | 98,004% | **97,9244%** |
| Lockbox CI | [95,0511; 97,9422] | mean 97,5981; sd 0,6246; range 3 seed |
| Số lỗi | 190 (F15) | **144 (D2) / 23 track** |
| Latency | 0,93 ms không ngữ cảnh | GPU 0,86 ms model-only; 1,24 ms end-to-end |
| Ngưỡng demo | 0,9462 | τ v2 = 0,9590, đã dùng |
| Checkpoint demo | legacy | v2_final_s42 (đã tích hợp) |
| Thời gian | 4,13 h | 3,99 h epoch-loop + 8,2 phút |
| Depth | CI vs Holm | “có bằng chứng vừa, chưa đủ sau Holm” |

### 13.4. Năm câu hỏi khi đọc báo cáo

1. **Đã làm gì?** Pipeline 5 phase, 111 run, split 3 tầng.
2. **Kết quả bao nhiêu?** OOF 97,04; lockbox 97,60; test clean 98,25.
3. **Vì sao tin?** Split sạch, audit 15/15, OOF, lockbox one-shot, CI, Holm,
   tái lập, 22 test, 12/12 nghiệm thu.
4. **Còn hạn chế gì?** Occlusion, lớp 19/29, chưa kiểm tra nhãn thủ công, chưa
   camera thật/máy yếu, golden test chưa chạy lại cho v2.
5. **Tiếp theo làm gì?** Cập nhật demo, kiểm tra thủ công, augmentation chọn lọc,
   xử lý occlusion, đo máy yếu.

### 13.5. Cách đọc sai thường gặp

| Đọc sai | Đọc đúng |
|---|---|
| “98,004% là OOF” | 98,004% là screening D2; OOF là 97,9244% |
| “Lockbox CI [95,05; 97,94]” | Đó là CI OOF; lockbox chỉ có mean ± sd/range |
| “Aug all giúp D2 thắng” | Aug all không đủ bằng chứng; D2 thắng nhờ cấu hình tổng thể |
| “144 lỗi = 144 ca” | 144 ảnh lỗi thuộc 23 track |
| “0,86 ms là phản hồi app” | Đó là model-only; end-to-end 1,24 ms; UI còn upload/render |
| “Demo đã dùng D2” | Đúng: app.py mặc định v2_final_s42 |

---

## 14. Glossary và bản đồ file

### 14.1. Glossary rút gọn

| Thuật ngữ | Nghĩa |
|---|---|
| Accuracy | Tỷ lệ đúng tổng thể |
| Precision | Trong các dự đoán lớp X, bao nhiêu đúng |
| Recall | Trong các ảnh thật lớp X, nhận ra bao nhiêu |
| F1 | Trung bình điều hòa precision/recall |
| Macro-F1 | Trung bình F1 43 lớp, mỗi lớp 1/43 |
| Track | Đoạn video ~30 frame của cùng một biển |
| Sign-group | Nhóm track gần trùng, đơn vị chia split |
| OOF | Dự đoán ngoài fold, mỗi ảnh một lần |
| Lockbox | Tập giữ riêng, mở một lần |
| CI 95% | Khoảng bất định |
| ECE | Sai lệch confidence vs accuracy |
| Brier | Sai số xác suất |
| Grad-CAM | Bản đồ nhiệt vùng ảnh ảnh hưởng dự đoán |
| Corruption | Biến đổi ảnh để test độ bền |
| Throughput | Ảnh/giây |

### 14.2. Bản đồ file

| File | Vai trò |
|---|---|
| `src/config.py` | Đọc YAML, hằng số, device, seed, env |
| `src/data.py` | Index, ROI, cache, preprocess, dataset, loader |
| `src/data_v2.py` | Split v2, cache, loader, augmentation |
| `src/split_v2.py` | Dựng sign-group, chia 3 tầng, audit |
| `src/audit_v2.py` | Kiểm tra C1–C12 |
| `src/models.py` | `SmallCNN`, `build_model`, `count_parameters` |
| `src/train_v2.py` | Train loop, checkpoint, metrics, meta |
| `src/baseline_v2.py` | HOG + LinearSVC |
| `src/stats_v2.py` | Macro-F1, bootstrap, Holm, variance |
| `src/analysis_v2.py` | Factorial, aggregate, OOF, corruption |
| `src/final_eval_v2.py` | Lockbox, test, ECE/Brier, guard freeze |
| `src/summarize_v2.py` | Sinh `summary.csv` |
| `src/decision_cards_v2.py` | Decision card 10 yếu tố |
| `src/explain_v2.py` | Learning curve, per-class, Grad-CAM, calibration |
| `src/predict.py` | Load model, predict top-k, validate upload |
| `scripts/run_matrix.py` | Chạy ma trận run |
| `scripts/select_candidates.py` | Chọn top2 + control |
| `scripts/run_oof_analysis.py` | OOF |
| `scripts/freeze_v2.py` | Freeze winner |
| `scripts/acceptance_v2.py` | 12 điều kiện nghiệm thu |
| `scripts/repro_check_v2.py` | Tái lập |
| `scripts/training_cost_v2.py` | Chi phí huấn luyện |
| `scripts/report_v3_data.py` | Bảng/số liệu cho báo cáo v3 |
| `scripts/report_v3_figures.py` | 20 hình |
| `scripts/report_v3_meta.py` | Hardware, deployment, đính chính |
| `scripts/report_v3_manifest.py` | Manifest sha256 |
| `app.py` | Demo Streamlit |
| `tests/` | Test split, stats, config |

### 14.3. Ba câu hỏi để tự kiểm tra

1. Nếu ai hỏi “vì sao chia theo sign-group?”, bạn trả lời được bằng 95,22% và
   1-NN 90,5% không?
2. Nếu ai hỏi “vì sao chọn D2?”, bạn trả lời được OOF 97,04, X2 hơn control
   1,981 điểm, và tie-break ít params không?
3. Nếu ai hỏi “kết quả có đáng tin không?”, bạn trả lời được audit 15/15, OOF,
   lockbox one-shot, CI, Holm, tái lập 0,0 không?

Nếu trả lời được ba câu này, bạn đã nắm được cốt lõi của dự án.

---

## Kết luận

Tài liệu này giải thích mã nguồn theo lộ trình: dữ liệu → chia tập → mô hình →
train → chạy ma trận → thống kê → chọn mô hình → giải thích → demo → tái lập.
Mỗi bước đều trả lời ba câu: **làm gì, tại sao, kết quả ra sao**.

Điểm cần nhớ nhất:

- **Chia theo sign-group** để không rò rỉ frame.
- **Macro-F1/43** vì lớp hiếm quan trọng.
- **OOF** để chọn, **lockbox** để xác nhận, **test** chỉ benchmark phụ.
- **CI + Holm** để không kết luận quá mức.
- **Không đủ bằng chứng** là một kết quả hợp lệ.

Báo cáo đầy đủ: `docs/bao_cao/BAO_CAO_v3.md`. Tài nguyên: `outputs/results/v3/`.
