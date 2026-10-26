# BÁO CÁO BÀI TẬP LỚN HỌC SÂU
## Phân loại 43 lớp biển báo giao thông Đức (GTSRB) bằng CNN tự xây dựng

**Môn học:** Học sâu — 12INF028 · **Ngày hoàn thành:** 23/10/2026
**Người thực hiện:** sinh viên lớp 12INF028 (repo `Nguyen15idhue/GTSRB-CNN`)
**Môi trường chạy:** Windows 11 · Python 3.12 · PyTorch 2.11.0+cu128 · RTX 5070 12 GB · i9-14900KF / 64 GB

> **Kết quả cuối (test, đánh giá đúng một lần):** accuracy **98.56 %**, macro-F1 **0.9799**,
> top-3 **99.64 %** trên **12.630** ảnh test. Multi-seed trên validation sạch:
> accuracy **98.08 % ± 0.31 điểm %**,
> macro-F1 **0.9654 ± 0.0047**.
>
> **Cảnh báo:** đây là demo học thuật trên biển báo Đức; **không dùng cho quyết định giao thông thực tế**.
> Validation **random** bị rò rỉ frame nên chỉ dùng validation **stratified-group** và test để kết luận
> (mục 2.4).

## Tóm tắt

Báo cáo trình bày pipeline hoàn chỉnh cho GTSRB: xác minh dữ liệu, hai split có kiểm soát rò rỉ,
baseline (majority, HOG + LinearSVC), ba cấu hình CNN tự xây dựng, bảy thí nghiệm ablation,
chọn mô hình trên **validation sạch theo track**, đa seed, đánh giá test **một lần**, phân tích lỗi
định lượng, kiểm chứng OOD và demo Streamlit.

Ba kết luận chính:

1. **Rò rỉ frame là vấn đề thật và định lượng được:** 95,22 % ảnh validation random có frame train
   cách ±1 trong cùng track; 1-NN (không huấn luyện) đạt 90,50 % trên val random nhưng chỉ 31,00 %
   trên val group. Vì vậy val random chỉ dùng để tham chiếu, **không dùng để xếp hạng mô hình**.
2. **Trên validation sạch, CNN-B + augmentation (E3) tốt hơn CNN-C + GAP (E4)** 1,62 điểm %
   macro-F1 — đảo ngược kết luận nếu chỉ nhìn val random.
3. **Kết quả test ổn định và đã kiểm chứng độc lập** (14/14 check): accuracy 98.56 %,
   macro-F1 0.9799; chênh val–test nhỏ và không có dấu hiệu lỗi metric.

---
## Chương 1 — Giới thiệu

### 1.1. Bài toán

Cho một ảnh màu RGB chứa **một** biển báo giao thông Đức, dự đoán nhãn thuộc **43 lớp** (0–42).
Ảnh gốc có kích thước thay đổi; hệ thống resize về **64×64** và chuẩn hóa theo mean/std của train.

| Thành phần | Nội dung |
|---|---|
| Input | ảnh RGB, resize 64×64 → tensor 3×64×64 (12.288 giá trị) |
| Output | logits 43 chiều → xác suất (softmax khi suy luận), top-3 cho demo |
| Số lớp | 43 (single-label, phân loại đa lớp) |
| Đối tượng | sinh viên/nghiên cứu thị giác máy tính; công cụ gán nhãn/kiểm kê biển báo |
| Giá trị | tự động hóa gán nhãn kho ảnh lớn, hỗ trợ kiểm kê chất lượng biển báo |
| Giới hạn | chỉ phân loại ảnh đã chứa biển; không phát hiện biển trong cảnh; không nhận diện biển Việt Nam |

*Bảng 1.1. Thành phần bài toán*
### 1.2. Phạm vi

Có làm: phân loại 43 lớp, ablation, đa seed, đánh giá test một lần, phân tích lỗi, demo ảnh ngoài.
**Không** làm: detection, nhận diện biển Việt Nam, huấn luyện trên ảnh người dùng, triển khai thời gian thực.

### 1.3. Cảnh báo sử dụng

Mô hình là **nguyên mẫu học thuật**. Sai số trên ảnh ngoài phân phối còn lớn (Chương 5), ngưỡng từ chối
chỉ dựa trên confidence và **không đảm bảo** phát hiện ảnh không có biển. **Không dùng cho quyết định
giao thông thực tế.**

---
## Chương 2 — Dữ liệu

### 2.1. Nguồn và điều kiện sử dụng

- **GTSRB** — German Traffic Sign Recognition Benchmark, Stallkamp et al., *Neural Networks* 32 (2012) 323–332.
- Trang chính thức: `https://benchmark.ini.rub.de/gtsrb_dataset.html` (không phân giải DNS từ máy này
  ngày 22/10/2026; câu điều kiện sử dụng được chép từ bản lưu Wayback, ghi trong `data_report.json`).
- Nguyên văn điều kiện: *"The data is free to use. However, we cordially ask you to cite the following
  publication if you do: …"* → dữ liệu dùng cho nghiên cứu, **không** phải CC0.
- Train lấy từ archive chính thức `GTSRB_Final_Training_Images.zip`; test qua `torchvision.datasets.GTSRB`
  (bản train rút gọn 26.640 ảnh của torchvision **không** được dùng).

### 2.2. Quy mô thật (đếm lại, không dùng số công bố)

| Hạng mục | Giá trị |
|---|---|
| Ảnh train trên đĩa | 39.209 |
| Ảnh test | 12.630 |
| Ảnh train dùng để chia split | 39.201 (loại 8 ảnh trùng MD5 với test) |
| Số lớp | 43 (train 43/43, test 43/43) |
| Lớp ít nhất / nhiều nhất (train) | lớp 0 (210 ảnh) / lớp 2 (2.250 ảnh) |
| Tỷ lệ mất cân bằng max/min | 10.71 lần |
| Số lớp < 300 ảnh train | 9 |
| ROI hợp lệ | train 39.209/39.209; test 12.630/12.630 |
| Track trích được | 39.209/39.209 ảnh, 1.307 track |
| Ảnh lỗi khi mở | 0 |

*Bảng 2.1. Quy mô và chất lượng dữ liệu (đếm lại từ đĩa)*
![Hình 2.1. Phân bố số ảnh theo lớp (tô đỏ: lớp dưới 300 ảnh)](outputs/figures/01_class_distribution.png)

*Hình 2.1. Phân bố số ảnh theo lớp (tô đỏ: lớp dưới 300 ảnh)*

![Hình 2.2. Phân bố kích thước ảnh gốc](outputs/figures/02_image_sizes.png)

*Hình 2.2. Phân bố kích thước ảnh gốc*

![Hình 2.3. Lưới 43 ảnh mẫu (mỗi lớp một ảnh; crop ROI chỉ để hiển thị, không phải input model)](outputs/figures/03_sample_grid.png)

*Hình 2.3. Lưới 43 ảnh mẫu (mỗi lớp một ảnh; crop ROI chỉ để hiển thị, không phải input model)*

### 2.2.1. Bản chất ảnh đầu vào: frame video đã crop quanh biển

GTSRB được xây từ các **video quay đường phố Đức**. Mỗi file `.ppm` là một **frame đã được trích và crop quanh biển báo**, không phải frame video đầy đủ và cũng không phải ảnh chỉ còn đúng hình biển. Ảnh vẫn chứa một phần nền (trời, đường, cột, xe...), nên kích thước rất nhỏ và biến thiên: trung vị **43×43**, min **25×25**, max **266×232**.

- Tên file train có dạng `<track>_<frame>.ppm`: các frame cùng track là cùng một biển xuất hiện liên tiếp trong video, gần như giống nhau.
- `GT-*.csv` cung cấp **ROI** — hộp bao quanh biển bên trong ảnh; ROI sát biển hơn vùng ảnh đã crop.
- Test cũng là frame trích từ video và có ROI, nhưng tên file chỉ là số thứ tự nên **không có track ID**.

Trong project này:

- Input của model là **toàn bộ ảnh GTSRB** (đã crop quanh biển), chuyển RGB và resize 64×64; **không crop thêm theo ROI**.
- ROI chỉ dùng để vẽ EDA/lưới ảnh mẫu, tính `roi_area_ratio` và phân tích lỗi.
- Đây là bài toán **phân loại**, không phải phát hiện: model giả định ảnh đã chứa biển và chỉ dự đoán lớp 0–42. Crop ROI tự động là hướng mở rộng (E8), **chưa triển khai**.

Hệ quả: vì các frame cùng track gần như trùng nhau, chia ngẫu nhiên theo ảnh sẽ khiến val chứa "anh em sinh đôi" của train; đây là cơ chế chính gây rò rỉ frame ở mục 2.4.

### 2.3. Ba cách chia tập

| Split | Train / Val | Track train / val | Lớp ở val | Lớp vắng | Lớp ít ảnh nhất ở val |
|---|---|---|---|---|---|
| `random` (stratified 80/20) | 31.360 / 7.841 | 1.307 / 1.306 (giao 1306) | 43/43 | – | 42 |
| `group` (GroupShuffleSplit) | 31.341 / 7.860 | 1.045 / 262 (giao 0) | 39/43 | [0, 19, 34, 39] | 0 |
| **`stratified_group` (dùng cho GĐ6)** | **31.371 / 7.830** | **1.046 / 261 (giao 0)** | **43/43** | – | **30** |

*Bảng 2.2. Ba cách chia tập*
`stratified_group` được thiết kế để vừa **tách track** (không rò rỉ frame) vừa **đủ 43 lớp** ở val:
mỗi lớp góp `max(1, round(0,2 × số track của lớp))` track vào val. Đây là split dùng để chọn mô hình
và báo cáo multi-seed.

### 2.4. Rò rỉ frame: định lượng

| Cửa sổ frame | Val random | Val group |
|---|---|---|
| ±1 frame | **95,22 %** (7.466/7.841) | 0 % |
| ±2 frame | 99,60 % (7.810/7.841) | 0 % |
| ±5 frame | **100 %** (7.841/7.841) | 0 % |
| Khoảng cách frame gần nhất | trung vị **1** | không có |

*Bảng 2.3. Tỷ lệ ảnh val có frame train gần (rò rỉ)*
Con số **1.306** ở Bảng 2.2 là **số track** bị chồng, **không phải 1.306 ảnh trùng**. Không có ảnh nào trùng index giữa train/val (`overlap_index = 0`); cái bị trùng là **chuỗi frame gần nhau** trong cùng track. Cụ thể **7.466/7.841 ảnh val (95,22 %)** có frame train cách ±1 và **100 %** trong ±5. Vì vậy val random không đo tổng quát hóa; mức ~99,6–99,97 % là do model nhận ra lại track, không phải năng lực thật.

Bằng chứng độc lập bằng **1-NN** (ảnh xám 16×16, không huấn luyện): val random **90,50 %**;
val random nhưng **cấm** láng giềng cùng track **36,50 %**; val group **31,00 %**. Nghĩa là val random
bị "giải sẵn" bởi frame gần trùng. Hệ quả thực nghiệm: mọi mô hình đều đạt ~99,6–99,97 % trên val random
(Bảng 4.2) nên **không thể xếp hạng** mô hình trên split này. Đo Δ trực tiếp: E3 trên val group giảm
**1,46 điểm % accuracy** và **2,53 điểm % macro-F1** (trên các lớp có mặt) so với val random
(`leak_delta.json`).

### 2.5. Kiểm tra trùng lặp (MD5 + pHash)

- **MD5:** 8 ảnh train trùng 8 ảnh test (cùng nhãn 14, track `00023` của tập train) → loại khỏi train,
  giữ file trên đĩa để số đếm khớp công bố. Danh sách: `excluded_train_duplicates.csv`.
- **pHash (64-bit) + MAD/NCC:** 16 cặp pHash trùng tuyệt đối và 17.214 cặp Hamming ≤ 7 chỉ là **va chạm
  hash giữa biển cùng lớp**; 13 ảnh test (0,10 %) có ảnh train rất giống ở 64×64 nhưng khi so ở ảnh gốc
  (crop ROI → 128×128) NCC **cao nhất 0,949**, trung vị 0,811 → **không có cặp gần trùng được xác nhận**.
  Nếu loại cả 13 ảnh, accuracy thay đổi **≤ 0.103 điểm %**.

![Hình 2.4. Kiểm chứng 6 cặp nghi ngờ nhất ở ảnh gốc: cùng loại biển nhưng khác khung/nền/độ nét](outputs/figures/09_phash_highres.png)

*Hình 2.4. Kiểm chứng 6 cặp nghi ngờ nhất ở ảnh gốc: cùng loại biển nhưng khác khung/nền/độ nét*

### 2.6. Tiền xử lý và augmentation

- Input model là **toàn ảnh crop quanh biển** (mục 2.2.1), resize 64×64; **không crop ROI**. ROI chỉ dùng cho EDA và phân tích lỗi.
- Một hàm duy nhất `src/data.py::preprocess` dùng cho train/eval/test/demo: RGB → resize 64×64 bilinear →
  `[0,1]` → chuẩn hóa bằng mean/std tính **trên train**:
  mean `[0,340408, 0,312365, 0,321632]`, std `[0,275148, 0,264242, 0,270747]` (`data_stats.json`).
- Cache `uint8` 64×64 (610,7 MB, tạo 13,6 s) để epoch nhanh; `X_train.npy` là **pool 39.201 ảnh**.
- Augmentation **chỉ train**: RandomAffine (xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5),
  ColorJitter nhẹ (0,2/0,2/0,2/0,02), RandomPerspective 0,1 (p=0,3).
  **Không** lật ngang/dọc, không xoay 90/180 (đổi nghĩa biển); có kiểm tra tự động chặn `Flip`.

![Hình 2.5. Lưới 8 ảnh gốc / 8 ảnh sau augmentation — không lật, số trên biển giữ nguyên chiều](outputs/figures/04_augmentation.png)

*Hình 2.5. Lưới 8 ảnh gốc / 8 ảnh sau augmentation — không lật, số trên biển giữ nguyên chiều*

---
## Chương 3 — Phương pháp

### 3.1. Baseline

| Mô hình | Thiết lập | Vai trò |
|---|---|---|
| E0a majority | dự đoán lớp nhiều nhất trong train (lớp 2) | sàn tham chiếu |
| E0b HOG + LinearSVC | ảnh xám 64×64, HOG 9 hướng, cell 8×8, block 2×2, L2-Hys → 1.764 chiều; StandardScaler fit trên train; C ∈ {0,1; 1; 10} chọn theo val macro-F1; `dual=true` | baseline chuẩn GTSRB, không augmentation |

*Bảng 3.1. Hai baseline*
### 3.2. Kiến trúc CNN

![Hình 3.1. Ba cấu hình SmallCNN (input 3×64×64 → logits 43)](outputs/figures/10_architecture.png)

*Hình 3.1. Ba cấu hình SmallCNN (input 3×64×64 → logits 43)*

| Mô hình | Khối conv | BN | Dropout | GAP | Aug | Params thật |
|---|---|---|---|---|---|---|
| CNN-A | 3 (32/64/128) | không | 0 | không | không | 2.201.707 |
| **CNN-B (cuối)** | 3 (32/64/128) | có | 0,3 | không | có | **2.201.931** |
| CNN-C | 4 (32/64/128/256) | có | 0,4 | có | có | 399.947 |

*Bảng 3.2. Ba cấu hình CNN*
- 64×64 là cân bằng chi tiết/chi phí; 3–4 khối 3×3 đủ receptive field cho biển báo.
- BN ổn định huấn luyện; dropout + weight decay chống overfit với ~31k ảnh; GAP giảm 2,1 M tham số FC.
- Không dùng pretrained (yêu cầu "CNN tự xây dựng").

### 3.3. Thiết lập huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer / LR / WD | AdamW / 1e-3 / 1e-4 (CNN-A: WD 0) |
| Batch / epoch tối đa / patience | 128 / 40 / 7 |
| Scheduler | CosineAnnealingLR (T_max = 40) |
| Loss | CrossEntropyLoss (E7: class weight 1/√freq) |
| AMP | bật khi CUDA |
| Chọn checkpoint | **val macro-F1 (43 lớp)** |
| Seed | 42; multi-seed 42/123/2024 |

*Bảng 3.3. Thiết lập huấn luyện*
### 3.4. Chỉ số và kỷ luật đánh giá

- Accuracy, **macro-F1/43**, per-class precision/recall/F1, confusion matrix chuẩn hóa theo hàng, top-3.
- Macro-F1 luôn tính trên **đủ 43 lớp** (`labels=range(43)`); nếu split thiếu lớp thì báo thêm
  F1 trên lớp có mặt và trần lý thuyết — tránh so sánh sai.
- Test chỉ chạy **sau khi** `configs/final.yaml` được commit (`freeze`), có `--confirm-freeze`,
  ghi log, không ghi đè; mọi lựa chọn mô hình/epoch/ngưỡng đều trên val.

![Hình 3.2. Pipeline thực nghiệm](outputs/figures/10_pipeline.png)

*Hình 3.2. Pipeline thực nghiệm*

---
## Chương 4 — Kết quả

### 4.1. Learning curves

![Hình 4.1. Loss / accuracy / macro-F1 của 4 cấu hình chính (val random)](outputs/figures/10_curves_4runs.png)

*Hình 4.1. Loss / accuracy / macro-F1 của 4 cấu hình chính (val random)*

CNN-A hội tụ nhanh nhưng dao động hơn; E3/E4 mượt hơn nhờ BN + augmentation. Lưu ý val random bị
rò rỉ frame nên các đường đều bão hoà ~0,99 — hình chỉ dùng để kiểm tra hội tụ, không để so sánh.

### 4.2. Baseline và ablation (val random)

| Mô hình | Val acc | Val macro-F1 | Ghi chú |
|---|---|---|---|
| E0a majority | 0.057391 | 0.002524 | lớp 2 (1.800 ảnh train) |
| E0b HOG + LinearSVC | 0.965693 | 0.973994 | C=0,1; trích 51,3 s; fit 540,2 s |
| E1 CNN-A | 0.996301 | 0.995705 | không BN/dropout/aug |
| E2 CNN-B | 0.998342 | 0.998392 | + BN + dropout + WD |
| E3 CNN-B + aug | 0.999362 | 0.999471 | + augmentation |
| E4 CNN-C + GAP + aug | 0.999745 | 0.999763 | 4 khối + GAP |
| E5a lr 3e-4 (10 epoch) | 0.989542 | 0.993387 | so sánh xu hướng |
| E5b lr 3e-3 (10 epoch) | 0.771968 | 0.567247 | phân kỳ |
| E7 class weight | 0.998470 | 0.998914 | 1/√freq |

*Bảng 4.1. Baseline và ablation trên val random*
![Hình 4.2. So sánh val accuracy và val macro-F1 giữa các run](outputs/figures/05_experiment_comparison.png)

*Hình 4.2. So sánh val accuracy và val macro-F1 giữa các run*

Bảng chi tiết 7 run CNN (params, epoch dừng, thời gian):

| Run | Params | Best epoch | Val acc | Val macro-F1 | Train−Val (điểm %) | Thời gian (s) |
|---|---|---|---|---|---|---|
| e1_cnna | 2.201.707 | 22 | 0.996301 | 0.995705 | +0.37 | 83.2 |
| e2_cnnb | 2.201.931 | 33 | 0.998342 | 0.998392 | +0.12 | 120.8 |
| e3_cnnb_aug | 2.201.931 | 36 | 0.999362 | 0.999471 | -1.22 | 378.1 |
| e4_cnnc_gap_aug | 399.947 | 32 | 0.999745 | 0.999763 | -0.07 | 369.5 |
| e5a_lr3e4 | 2.201.931 | 10 | 0.989542 | 0.993387 | -6.33 | 107.4 |
| e5b_lr3e3 | 2.201.931 | 9 | 0.771968 | 0.567247 | -26.38 | 107.8 |
| e7_classweight | 2.201.931 | 21 | 0.998470 | 0.998914 | -3.52 | 270.4 |

*Bảng 4.2. Chi tiết 7 run CNN*
**Đọc bảng này đúng cách:** các chênh lệch rất nhỏ (E4 hơn E3 đúng **2 ảnh/7.841**) là **nhiễu**, không
phải bằng chứng mô hình tốt hơn. Vì vậy Giai đoạn 5 chỉ ghi nhận, không chốt mô hình trên split này.

### 4.3. Chọn mô hình trên validation sạch

Hai ứng viên sát nhau ở Giai đoạn 5 được chạy lại **trên `split_stratified_group`** (seed 42):

| Run | Params | Val acc | Val macro-F1 | Best epoch |
|---|---|---|---|---|
| E3 (CNN-B + aug) | 2.201.931 | **0,983525** | **0,970613** | 34 |
| E4 (CNN-C + GAP) | 399.947 | 0,976373 | 0,954453 | 34 |

*Bảng 4.3. Hai ứng viên trên val stratified_group*
E3 hơn E4 **1,62 điểm %** macro-F1 (> 0,3 điểm %) → chọn **E3**. Quy tắc "ưu tiên ít tham số" chỉ áp
dụng khi chênh < 0,3 điểm %, nên không chọn E4 dù chỉ bằng 18 % tham số. Đây là ví dụ rõ cho việc
**split quyết định kết luận**.

### 4.4. Multi-seed và test

| Seed | Val acc | Val macro-F1 | Best epoch |
|---|---|---|---|
| 42 | 98.35 % | 0.970613 | 34 |
| 123 | 97.75 % | 0.961541 | 29 |
| 2024 | 98.15 % | 0.964021 | 39 |
| **Mean ± std** | **98.08 % ± 0.31** | **0.965392 ± 0.004689** | – |

*Bảng 4.4. Kết quả 3 seed*
![Hình 4.3. Val accuracy / macro-F1 theo 3 seed (đường đỏ: mean, dải: ±1 std)](outputs/figures/10_multiseed.png)

*Hình 4.3. Val accuracy / macro-F1 theo 3 seed (đường đỏ: mean, dải: ±1 std)*

| Chỉ số test | Giá trị |
|---|---|
| Số ảnh | 12.630 |
| Accuracy | **98.56 %** |
| Macro-F1 / 43 lớp | **0.979894** |
| Top-3 accuracy | 99.64 % |
| Checkpoint sha256 | `9c4dd0d677d655e2…` |
| Commit freeze | `8f61c5a6cc` |

*Bảng 4.5. Kết quả test (đánh giá đúng một lần)*
![Hình 4.4. Confusion matrix test 43×43 (chuẩn hóa theo hàng)](outputs/figures/10_confusion_full.png)

*Hình 4.4. Confusion matrix test 43×43 (chuẩn hóa theo hàng)*

**Chênh val–test:** test **cao hơn** mean val 0,47 điểm % accuracy và 1,45 điểm % macro-F1. Đã kiểm tra
lại: accuracy/F1/top-3 tính lại từ `predictions.csv` khớp từng bit, nhãn khớp `GT-final_test.csv`,
sha256 khớp (`test_verification.json` 14/14 PASS). Phân phối ảnh val/test gần như trùng nhau (median ROI
0,5625 vs 0,5639; blur 0,0022 vs 0,0023; brightness 0,325 vs 0,338), nên **không suy diễn nguyên nhân**;
chênh lệch cỡ 1,5 lần std giữa seed và được ghi nhận, không phải lỗi metric.

---
## Chương 5 — Phân tích lỗi

### 5.1. Per-class

Chỉ **1 lớp** có F1 < 0,90. Lớp thấp nhất: **27 (Người đi bộ), F1 0,846,
support 60**. Với support nhỏ, 1 ảnh sai làm F1 giảm ~0,03, nên không kết luận "mô hình kém" cho lớp này.

![Hình 5.1. F1 theo lớp trên test (sắp tăng dần, kèm support)](outputs/figures/07_per_class_f1.png)

*Hình 5.1. F1 theo lớp trên test (sắp tăng dần, kèm support)*

### 5.2. Confusion và cặp nhầm

![Hình 5.2. Zoom 15 lớp F1 thấp nhất (chuẩn hóa theo hàng)](outputs/figures/07_confusion_zoom15.png)

*Hình 5.2. Zoom 15 lớp F1 thấp nhất (chuẩn hóa theo hàng)*

| Hạng | Cặp (thật → dự đoán) | Số ảnh |
|---|---|---|
| 1 | 27 → 11 | 15 |
| 2 | 2 → 1 | 12 |
| 3 | 8 → 5 | 11 |
| 4 | 22 → 25 | 10 |
| 5 | 12 → 13 | 10 |
| 6 | 8 → 3 | 9 |
| 7 | 39 → 33 | 7 |
| 8 | 5 → 7 | 7 |
| 9 | 3 → 5 | 7 |
| 10 | 30 → 11 | 6 |

*Bảng 5.1. Top 10 cặp nhầm*
![Hình 5.3. Top 10 cặp nhầm trên test](outputs/figures/10_top10_pairs_bar.png)

*Hình 5.3. Top 10 cặp nhầm trên test*

![Hình 5.4. Ảnh ví dụ cho từng cặp nhầm (crop ROI, đã phóng to)](outputs/figures/07_top_pairs.png)

*Hình 5.4. Ảnh ví dụ cho từng cặp nhầm (crop ROI, đã phóng to)*

### 5.3. Error rate theo nguyên nhân (định lượng)

| Biến | Nhóm thấp | Nhóm giữa | Nhóm cao | Chênh nhóm cao−thấp (CI95) | r² |
|---|---|---|---|---|---|
| Độ mờ (Laplacian, cao = nét) | 2.21 % | 1.02 % | 1.09 % | +1.19 điểm % (+0.65…+1.72) | 0.0003 |
| Độ sáng (mean V) | 1.76 % | 1.69 % | 0.88 % | +0.88 điểm % (+0.39…+1.37) | 0.0012 |
| Diện tích biển / ảnh | 2.28 % | 1.25 % | 0.79 % | +1.49 điểm % (+0.96…+2.01) | 0.0043 |

*Bảng 5.2. Error rate theo tertile*
![Hình 5.5. Error rate theo tertile của 3 biến (test, n = 12.630)](outputs/figures/07_error_rates.png)

*Hình 5.5. Error rate theo tertile của 3 biến (test, n = 12.630)*

**Diễn giải có kỷ luật:** cả ba biến có **effect size rất nhỏ** (r² từ 0,0003 đến 0,0043 — dưới 0,5 %
phương sai). p nhỏ chủ yếu do n = 12.630. Vì vậy chỉ nói **xu hướng**: ảnh mờ/tối/biển nhỏ có error rate
cao hơn, không khẳng định quan hệ nhân quả.

### 5.4. Mười hai mẫu sai (quan sát ≠ giả thuyết)

![Hình 5.6. 12 mẫu dự đoán sai trên test (crop ROI, đã phóng to)](outputs/figures/07_error_samples.png)

*Hình 5.6. 12 mẫu dự đoán sai trên test (crop ROI, đã phóng to)*

| # | Thật → dự đoán | Conf | Nhóm | Quan sát được | Giả thuyết |
|---|---|---|---|---|---|
| 1 | 37 → 35 | 0.994 | E-A mờ | Ảnh biển xanh tròn bị nhòe mạnh; chỉ còn mảng xanh đậm, mũi tên rẽ trái không đọc được (blur 0,00014 — nhóm mờ nhất). | Khi độ tương phản của mũi tên dưới ngưỡng, model dựa vào 'khối xanh tròn' và chọn lớp phổ biến nhất trong nhóm mũi tên (đi thẳng). |
| 2 | 30 → 11 | 0.990 | E-B tối | Ảnh chụp đêm; biển tam giác gần như đen, chỉ thấy viền mờ, không thấy hình bên trong (bright 0,145 — nhóm tối nhất). | Thiếu sáng làm mất chi tiết bên trong; model khớp phần viền tam giác với lớp 11 (đường ưu tiên) do tương phản hình dạng. |
| 3 | 12 → 13 | 0.998 | E-C biển nhỏ | Biển 'đường ưu tiên' bị cháy sáng: hình thoi gần trắng, viền vàng nhạt, không thấy viền đỏ (bright 0,586). | Overexposure làm mất viền/màu đỏ; hình thoi trắng dễ nhầm với biển 'nhường đường' (tam giác ngược màu trắng). |
| 4 | 12 → 13 | 1.000 | E-E lớp giống nhau | Cùng lớp 12, ảnh nét hơn (blur 0,0030) nhưng cũng cháy sáng, mất viền đỏ. | Lỗi do phân phối (biển cháy sáng) chứ không do mờ — cho thấy nguyên nhân E-B/E-C không loại trừ nhau. |
| 5 | 17 → 14 | 0.975 | E-A mờ | Biển 'cấm đi vào' bị nhòe; thanh trắng ngang gần như biến mất, chỉ còn đĩa đỏ (blur 0,000075). | Mất thanh trắng → còn đĩa đỏ tròn, gần với biển 'dừng lại' (14) về màu và hình bao. |
| 6 | 39 → 33 | 0.987 | E-B tối | Ảnh đêm, biển 'đi bên trái' rất tối; mũi tên sáng mờ, hướng không rõ (bright 0,127). | Mất chi tiết hướng mũi tên → nhầm trái/phải (lớp 33 'rẽ phải'), lỗi đối xứng điển hình. |
| 7 | 12 → 13 | 0.996 | E-C biển nhỏ | Biển 'đường ưu tiên' cháy sáng tương tự mẫu 3 (blur 0,0022; bright 0,576). | Cùng cơ chế overexposure; lặp lại ở 4/12 mẫu nên đây là nguyên nhân hệ thống, không phải cá biệt. |
| 8 | 2 → 1 | 0.999 | E-E lớp giống nhau | Biển '50' trong tối (bright 0,212), số mờ; nét chữ 5 và 3 khó phân biệt. | Độ tương phản chữ số thấp, hai lớp chỉ khác một nét → nhầm 50/30 (cặp nhầm 2→1 có 12 ảnh trên test). |
| 9 | 17 → 14 | 0.958 | E-A mờ | Biển 'cấm đi vào' nhòe, thanh trắng gần mất (blur 0,000096). | Cùng cơ chế mẫu 5; lặp lại cho thấy nhóm mờ là nguyên nhân E-A thật. |
| 10 | 39 → 33 | 0.937 | E-B tối | Biển 'đi bên trái' ban đêm, mũi tên mờ (bright 0,134). | Cùng cơ chế mẫu 6: mất hướng mũi tên do thiếu sáng. |
| 11 | 12 → 13 | 0.996 | E-C biển nhỏ | Biển 'đường ưu tiên' cháy sáng (bright 0,590). | Cùng cơ chế mẫu 3/4/7: mất màu/viền do overexposure. |
| 12 | 2 → 1 | 0.998 | E-E lớp giống nhau | Biển '50' có hoa văn moiré rõ, chữ số bị răng cưa (blur 0,0204 — cao nhất trong 12 mẫu). | Alias do hạ mẫu (resize 64×64) làm nét chữ số 5 giống 3; khác với nhóm mờ vì ảnh gốc nét nhưng bị giao thoa. |

*Bảng 5.3. Mười hai mẫu sai*
Bốn cơ chế lặp lại: **cháy sáng** làm mất viền đỏ của biển "đường ưu tiên" (4/12), **mờ** làm mất thanh
trắng của "cấm đi vào" (2/12), **thiếu sáng** làm mất hướng mũi tên (2/12), và **chữ số 50/30** khác nhau
một nét cộng moiré do resize (2/12). Các quan sát này lấy từ việc mở từng ảnh, không suy đoán.

### 5.5. OOD và ngưỡng từ chối

![Hình 5.7. Ảnh ngoài dataset (CC0/PD) — nhãn dự đoán hoặc "không xác định"](outputs/figures/07_ood.png)

*Hình 5.7. Ảnh ngoài dataset (CC0/PD) — nhãn dự đoán hoặc "không xác định"*

| Nhóm | n | Bị từ chối | Tỷ lệ (CI95 Wilson) |
|---|---|---|---|
| ood_sign_speed | 2 | 1 | 50 % (9–91 %) |
| ood_sign_text | 4 | 2 | 50 % (15–85 %) |
| no_sign | 4 | 3 | 75 % (30–95 %) |

*Bảng 5.4. OOD theo nhóm*
- τ = **0.9462** (phân vị 5 % max-prob trên val; acceptance in-distribution
  94.99 %), **không** chọn trên demo.
- Ngưỡng này là **cơ chế từ chối theo confidence**, không phải bộ phát hiện OOD: 1/4 ảnh đường vẫn bị
  nhận nhầm "Nhường đường" với confidence 0,996. Với n = 4–6, khoảng tin cậy rất rộng — không kết luận
  tỷ lệ chính xác.
- Ảnh biển nước ngoài (Mỹ/Hàn) top-1 đúng nhãn tương ứng nhưng 1/2 bị τ từ chối vì biển nhỏ; biển chữ
  Philippines bị gán nhãn Đức 2/4 lần → đúng bản chất domain shift.

---
## Chương 6 — Demo và kết luận

### 6.1. Demo

- `app.py` (Streamlit): upload JPG/JPEG/PNG ≤ 20 MB, hiển thị ảnh + top-3 tiếng Việt, cảnh báo
  "không xác định" khi max-prob < τ, xử lý 3 lỗi (file hỏng, quá lớn, thiếu checkpoint), footer cảnh báo.
- `src/predict.py` dùng đúng `src.data.preprocess` (không viết lại) và checkpoint cuối `final_s42`.
- AppTest: **0 exception**, có footer cảnh báo (`phase8_check.json`).

![Hình 6.1. Giao diện Streamlit chạy thật (Playwright, Chromium headless): trang chủ, kết quả đúng, từ chối ảnh không biển, lỗi file hỏng](outputs/figures/11_ui_overview.png)

*Hình 6.1. Giao diện Streamlit chạy thật (Playwright, Chromium headless): trang chủ, kết quả đúng, từ chối ảnh không biển, lỗi file hỏng*

App hiển thị **cảnh báo phạm vi** ngay đầu trang: chỉ 43 lớp biển Đức; biển Việt Nam (ví dụ cấm dừng/đỗ) không có lớp tương ứng nên sẽ bị trả “không xác định” hoặc đoán sai — có expander giải thích riêng. Giao diện được kiểm thử **end-to-end bằng Playwright** (`tests/e2e_app.py`): tự khởi động Streamlit,
upload ảnh thật, kiểm tra top-3, cảnh báo "không xác định", 3 loại lỗi, tab thông tin, ảnh mẫu và
trường hợp thiếu checkpoint. Kết quả **9/9 PASS**; thời gian từ upload đến khi hiện kết quả
**median 0,354 s** (2 lần đo, gồm upload + Streamlit rerun + render). Ảnh chụp:
`outputs/figures/11_ui_*.png`.

![Hình 6.2. Ảnh minh chứng 5 kịch bản demo (ảnh ghép từ kết quả chạy thật, không phải screenshot UI)](outputs/figures/08_scenarios.png)

*Hình 6.2. Ảnh minh chứng 5 kịch bản demo (ảnh ghép từ kết quả chạy thật, không phải screenshot UI)*

| Kịch bản | Kết quả | Conf | Bị từ chối |
|---|---|---|---|
| 1. Ảnh test GTSRB (chưa dùng train) | Cấm xe trên 3,5 tấn | 0.999997 | không |
| 2. Ảnh biển nước ngoài đã crop | Giới hạn tốc độ 30 km/h | 0.949504 | không |
| 3. Cảnh thật, biển nhỏ | Giới hạn tốc độ 20 km/h | 0.673364 | có |
| 4. Ảnh không có biển | Rẽ trái phía trước | 0.607856 | có |
| 4b. Ảnh không có biển (bị nhận nhầm) | Nhường đường | 0.99595 | không |
| 5. File lỗi / quá lớn | app báo lỗi thân thiện, không crash | – | không |

*Bảng 6.1. Kết quả 5 kịch bản demo*
**Thời gian:** model-only median **3.02 ms**; end-to-end trong Python
(bytes → validate → preprocess → forward → top-3) median **5.78 ms**,
p95 7.80 ms. **Chưa tính** upload mạng và render UI — không gọi
đây là "thời gian người dùng chờ".

![Hình 6.2. 11 ảnh ngoài dataset đã kiểm tra ẩn danh (không người/biển số)](outputs/figures/08_external_contact.png)

*Hình 6.2. 11 ảnh ngoài dataset đã kiểm tra ẩn danh (không người/biển số)*

### 6.2. Tái lập và kiểm thử

- `python -m pytest tests -q` → **11 passed** (split, macro-F1 43 lớp, checkpoint, validate upload,
  predictions/report integrity, log test 1 lần, threshold, manifest/license, class_names).
- Test đã chạy **đúng một lần**; `test_runs.log` 1 dòng; chạy lại bị từ chối trước khi ghi.
- Kiểm chứng độc lập test: **14/14 PASS** — 12.630 dòng/path duy nhất,
  nhãn khớp GT, metric và sha256 khớp.

- Clean-room (`.venv_test`, cài từ requirements + torch cu128):
  CUDA `True` · capability `(12, 0)` ·
  smoke train 2 epoch · evaluate val acc
  `0.983525` (kỳ vọng `0.983525`) ·
  AppTest `0 exception` (1.01 phút).

- Golden test 20 ảnh giữa hai môi trường: lệch xác suất lớn nhất
  **0.00e+00** (ngưỡng 1e-3), lệch nhãn 0.

### 6.3. Hạn chế (thật, không giấu)

1. **Val random bị rò rỉ frame** (95,22 % ±1); mọi so sánh trên split này chỉ có giá trị tham chiếu.
2. **Val group thiếu lớp** (0, 19, 34, 39) nếu dùng GroupShuffleSplit thuần; đã xử lý bằng
   `stratified_group`, nhưng split này vẫn là thiết kế riêng, chưa được chuẩn hóa trong tài liệu GTSRB.
3. **Sai số ngoài phân phối còn lớn**: biển nước ngoài/biển chữ có thể bị gán nhãn Đức với confidence cao;
   τ không phải bộ phát hiện OOD (1/4 ảnh không biển lọt).
4. **Chỉ 43 lớp biển Đức**, ảnh 64×64; không xử lý cảnh có nhiều biển hoặc biển quá nhỏ. Thử với biển Việt Nam (cấm dừng/đỗ) cho thấy app trả “không xác định” hoặc đoán sai — đúng như giới hạn này.
5. **Một lần test duy nhất** đúng kỷ luật, nhưng nghĩa là không có ước lượng độ bất định của chính test;
   multi-seed chỉ đo biến thiên huấn luyện trên val.
6. **mean/std** tính trên train của split random nhưng dùng cho mô hình cuối (split stratified-group) —
   lệch nhỏ, đã ghi nhận; nếu làm lại nên tính trên train của split cuối.
7. **UI đã có screenshot thật** (Playwright + Chromium headless, 9/9 kịch bản), nhưng **chưa có video demo**
   và chưa thử trên trình duyệt khác ngoài Chromium.
8. Giai đoạn 11–12 (slide, đóng gói/nộp) **chưa thực hiện** theo yêu cầu.

### 6.4. Hướng phát triển

Thu ảnh thật nhiều hơn (đặc biệt biển Việt Nam) để đo domain shift có kiểm soát; calibration xác suất
(temperature scaling) và OOD detection thực sự (energy/Mahalanobis); crop ROI tự động (E8);
tăng độ phân giải (96×96) và so sánh; distillation/pruning cho thiết bị yếu.

### 6.5. Kết luận

Pipeline đã đi đủ chuỗi dữ liệu → baseline → CNN → ablation → chọn mô hình trên val sạch → đa seed →
test một lần → phân tích lỗi → demo. Kết quả test **98.56 % accuracy / 0.9799 macro-F1** là con số
đã được kiểm chứng chéo (predictions, nhãn GT, sha256, log), và mọi hạn chế đã được nêu định lượng.

> **Mô hình chỉ phục vụ học tập, không dùng cho quyết định giao thông thực tế.**

---

## Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. *Man vs. computer: Benchmarking machine learning
   algorithms for traffic sign recognition.* Neural Networks 32 (2012) 323–332.
2. GTSRB dataset — `https://benchmark.ini.rub.de/gtsrb_dataset.html` (truy cập 22/10/2026, qua Wayback).
3. Torchvision datasets — `https://docs.pytorch.org/vision/main/datasets.html`.
4. PyTorch documentation — `https://pytorch.org/docs/stable/index.html`.
5. scikit-learn user guide — `https://scikit-learn.org/stable/`.
6. Streamlit documentation — `https://docs.streamlit.io/`.

---

## Phụ lục A — Per-class trên test

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 27 | 1.0000 | 0.7333 | 0.8462 | 60 |
| 30 | 0.9701 | 0.8667 | 0.9155 | 150 |
| 23 | 0.9281 | 0.9467 | 0.9373 | 150 |
| 22 | 1.0000 | 0.9000 | 0.9474 | 120 |
| 39 | 0.9881 | 0.9222 | 0.9540 | 90 |
| 20 | 0.9184 | 1.0000 | 0.9574 | 90 |
| 19 | 0.9375 | 1.0000 | 0.9677 | 60 |
| 11 | 0.9396 | 1.0000 | 0.9689 | 420 |
| 5 | 0.9670 | 0.9762 | 0.9716 | 630 |
| 29 | 0.9885 | 0.9556 | 0.9718 | 90 |
| 8 | 0.9908 | 0.9556 | 0.9729 | 450 |
| 3 | 0.9692 | 0.9800 | 0.9746 | 450 |
| 18 | 0.9921 | 0.9615 | 0.9766 | 390 |
| 33 | 0.9676 | 0.9952 | 0.9812 | 210 |
| 37 | 1.0000 | 0.9667 | 0.9831 | 60 |
| 25 | 0.9795 | 0.9938 | 0.9866 | 480 |
| 2 | 0.9906 | 0.9840 | 0.9873 | 750 |
| 7 | 0.9803 | 0.9956 | 0.9879 | 450 |
| 24 | 1.0000 | 0.9778 | 0.9888 | 90 |
| 40 | 1.0000 | 0.9778 | 0.9888 | 90 |
| 14 | 0.9783 | 1.0000 | 0.9890 | 270 |
| 21 | 0.9783 | 1.0000 | 0.9890 | 90 |
| 26 | 0.9783 | 1.0000 | 0.9890 | 180 |
| 1 | 0.9822 | 0.9972 | 0.9897 | 720 |
| 12 | 1.0000 | 0.9797 | 0.9898 | 690 |
| 31 | 0.9818 | 1.0000 | 0.9908 | 270 |
| 13 | 0.9849 | 0.9972 | 0.9910 | 720 |
| 17 | 1.0000 | 0.9833 | 0.9916 | 360 |
| 32 | 1.0000 | 0.9833 | 0.9916 | 60 |
| 36 | 0.9836 | 1.0000 | 0.9917 | 120 |
| 41 | 0.9836 | 1.0000 | 0.9917 | 60 |
| 15 | 0.9859 | 1.0000 | 0.9929 | 210 |
| 6 | 1.0000 | 0.9867 | 0.9933 | 150 |
| 35 | 0.9924 | 1.0000 | 0.9962 | 390 |
| 28 | 0.9934 | 1.0000 | 0.9967 | 150 |
| 4 | 1.0000 | 0.9939 | 0.9970 | 660 |
| 10 | 1.0000 | 0.9985 | 0.9992 | 660 |
| 0 | 1.0000 | 1.0000 | 1.0000 | 60 |
| 9 | 1.0000 | 1.0000 | 1.0000 | 480 |
| 16 | 1.0000 | 1.0000 | 1.0000 | 150 |
| 34 | 1.0000 | 1.0000 | 1.0000 | 120 |
| 38 | 1.0000 | 1.0000 | 1.0000 | 690 |
| 42 | 1.0000 | 1.0000 | 1.0000 | 90 |

*Bảng A.1. Per-class trên test*

## Phụ lục B — Artifact chính

| File | Nội dung |
|---|---|
| `outputs/results/test_report.json`, `test_final.json` | kết quả test duy nhất (acc, F1, per-class, confusion, sha256) |
| `outputs/results/predictions_test_final.csv` | 12.630 dòng dự đoán test (bản sao nguyên văn, kèm sha256) |
| `outputs/results/summary.csv` | bảng tổng hợp run (có cột `split`, `role`) |
| `outputs/results/multiseed.json` | mean ± std 3 seed |
| `outputs/results/error_report.json` | per-class, top cặp, tertile, 12 mẫu, OOD |
| `outputs/results/leak_delta.json` | định lượng rò rỉ frame |
| `outputs/results/phash_*.json`, `phash_pairs.csv` | audit gần trùng |
| `outputs/results/phase2..8_check.json` | checklist từng giai đoạn |
| `outputs/results/cleanroom.json`, `golden_compare.json` | tái lập môi trường sạch |

*Bảng B.1. Artifact chính*

## Phụ lục C — Lệnh chạy

```powershell
python -m src.data --download --report
python -m src.data --prepare --verify
python -m src.train --config configs/final.yaml --seed 42
python -m src.evaluate --run outputs/runs/final_s42 --split val
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze  # chỉ 1 lần
python -m src.summarize --figure --multiseed
python -m src.error_analysis --run outputs/runs/final_s42
python -m pytest tests -q
streamlit run app.py
```