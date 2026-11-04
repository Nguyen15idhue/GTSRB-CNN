# BÁO CÁO DỰ ÁN HỌC SÂU — GTSRB CNN

## Phân loại 43 lớp biển báo giao thông Đức bằng CNN tự xây dựng

| Thông tin | Giá trị |
|---|---|
| Học phần | 12INF028 — Học sâu |
| Người thực hiện | Nguyen15idhue |
| Giảng viên hướng dẫn | — (chưa cung cấp trong repo) |
| Thời gian thực hiện | 22/10/2026 – 04/11/2026 |
| Phiên bản báo cáo | 3.0 — bản làm lại sau đánh giá báo cáo v2 |
| Phiên bản mã nguồn | git commit `201b59d` |
| Ngày cập nhật số liệu | 04/11/2026 |
| Phạm vi | 111 run CNN + 1 baseline; split 3 tầng; OOF; lockbox; official test |
| Nguồn số liệu | `outputs/results/v3/`, `outputs/results/v2/`, `outputs/splits/v2/` |
| Kết quả chính | OOF macro-F1 **0,9704**; lockbox **0,9760**; test clean **0,9825** |

---

## Tóm tắt điều hành

Báo cáo này trình bày lại toàn bộ dự án phân loại 43 lớp biển báo giao thông Đức
(GTSRB) bằng CNN tự xây dựng, sau khi báo cáo v2 bị đánh giá là có nhiều số liệu
nhưng chưa tổ chức thành một hệ thống thông tin. Bản v3 **không chạy lại 111 run**
và **không mở lại lockbox**; mọi số liệu được lấy từ artifact đã có hoặc đo mới
trong phạm vi suy luận.

### Bảng 0.1. Kết quả chính

| Hạng mục | Giá trị |
|---|---|
| Số lớp | 43 |
| Tổng ảnh train ban đầu | 39.209 |
| Pool sau khi loại 8 ảnh trùng MD5 với test | 39.201 |
| Lockbox | 8.302 ảnh, 277 track, 238 sign-group |
| Dev | 30.899 ảnh, 1.030 track, 945 sign-group |
| Số run CNN | 111 |
| Baseline | 1 (HOG + LinearSVC) |
| Mô hình được chọn | D2: 4 khối conv 32/64/128/256, flatten, BN, dropout 0,3, 1.448.779 params |
| OOF accuracy | **97,9244%** (pooled 30.899 ảnh, trung bình 3 seed) |
| OOF macro-F1/43 | **97,0400% ± 0,1869** |
| Lockbox accuracy | **98,8758% ± 0,1010** |
| Lockbox macro-F1/43 | **97,5981% ± 0,6246** |
| Official test raw | 98,6989% / 98,0398% (accuracy / macro-F1) |
| Official test clean | **98,8692% / 98,2499%** |
| Tổng thời gian 111 run | 3,99 giờ epoch-loop + 8,2 phút baseline |

### Năm phát hiện chính

1. **Chia theo sign-group là bắt buộc.** Với split random, 95,22% ảnh validation
   có frame train cách không quá 1 frame; 1-NN đạt 90,5% mà không học gì. Split
   sign-group loại bỏ hoàn toàn hiện tượng này.
2. **BN và dropout là hai yếu tố có bằng chứng rõ.** BN +3,724 điểm, dropout
   +5,963 điểm, CI 95% đều không chứa 0. Weight decay và augmentation "all"
   **không đủ bằng chứng**.
3. **D2 thắng OOF nhưng không vượt trội X1.** D2 đạt 97,0400% macro-F1; X1 đạt
   96,2971%; chênh lệch chưa kết luận (CI chứa 0). D2 được chọn theo tie-break
   vì ít tham số hơn (1,45M so với 2,20M).
4. **Occlusion là điểm yếu lớn nhất.** Khi che 10–30% trung tâm, accuracy trung
   bình chỉ còn 17,63% (D2), thấp hơn nhiều so với blur, noise, JPEG.
5. **Lỗi tập trung vào ít track.** D2 có 144 ảnh lỗi trên screening val nhưng
   chỉ thuộc 23 track; không được đếm 144 ảnh lỗi như 144 ca độc lập.

### Ba lưu ý bắt buộc khi đọc số

- **Bốn tập đánh giá không tương đương:** screening val 6.180 ảnh, OOF 30.899 ảnh,
  lockbox 8.302 ảnh, official test 12.630 ảnh. Không so sánh trực tiếp các con số.
- **Không có kiểm chứng camera thật, thời tiết thật, máy cấu hình thấp.** Mọi
  tuyên bố hiệu năng chỉ có giá trị trong phạm vi GTSRB và máy train.
- **Demo Streamlit hiện vẫn dùng checkpoint legacy `final_s42` và ngưỡng cũ
  0,9462**, chưa dùng D2/v2. Báo cáo tách rõ "mô hình đã kiểm chứng" và "demo
  đã triển khai".

### Mục lục

| Chương | Nội dung |
|---|---|
| 1 | Tổng quan dự án |
| 2 | Dữ liệu và chia tập |
| 3 | Kiến trúc và môi trường |
| 4 | Thiết kế thực nghiệm |
| 5 | Kết quả thực nghiệm |
| 6 | Vì sao chọn D2 |
| 7 | Đánh giá từng lớp |
| 8 | Phân tích lỗi và nhầm lẫn |
| 9 | Robustness và khả năng giải thích |
| 10 | Hiệu quả vận hành và triển khai |
| 11 | Độ tin cậy và tái lập |
| 12 | Hạn chế |
| 13 | Hướng phát triển |
| 14 | Kết luận |
| 15 | Tài liệu tham khảo |
| Phụ lục A–M | Bảng 43 lớp, split, factorial, per-class, confusion, hardware, calibration, lỗi, đính chính, lệnh chạy, glossary |

---

## Chương 1. Tổng quan dự án

### 1.1. Bài toán

GTSRB (German Traffic Sign Recognition Benchmark) gồm **43 lớp biển báo giao
thông Đức**, ảnh là các frame video đã được cắt quanh biển. Nhiệm vụ của dự án:
huấn luyện một CNN **tự xây dựng, không dùng pretrained**, nhận ảnh RGB 64×64 và
trả về logits cho 43 lớp.

Bài toán có ba đặc điểm khó:

- **Mất cân bằng lớp:** lớp ít nhất 210 ảnh, lớp nhiều nhất 2.250 ảnh, tỷ lệ
  10,714 lần.
- **Ảnh trong cùng một track gần như trùng nhau:** mỗi track là một đoạn video
  khoảng 30 frame của cùng một biển. Chia ngẫu nhiên theo ảnh sẽ khiến validation
  bị rò rỉ frame.
- **Điều kiện ảnh thay đổi:** tối, mờ, chói, góc nhìn lạ, biển bị che.

Vì vậy báo cáo dùng **macro-F1/43** làm chỉ số chính, không chỉ accuracy, và chia
dữ liệu theo **sign-group** thay vì random.

### 1.2. Mục tiêu

| Mục tiêu | Chỉ tiêu đánh giá |
|---|---|
| Phân loại chính xác 43 lớp | Accuracy, macro-F1/43 |
| Phân biệt lớp ít dữ liệu | F1 từng lớp, per-class precision/recall |
| Hạn chế dự đoán sai nhưng tự tin | ECE, Brier score, confidence đúng/sai |
| Tổng quát hóa | OOF 3 fold × 3 seed, lockbox độc lập |
| Chịu được ảnh xấu | Corruption robustness 6 loại × 3 mức |
| Có thể tái lập | Seed, hash, checkpoint, test tự động |
| Hiệu quả tính toán | Latency, throughput, số tham số, RAM/VRAM |

### 1.3. Phạm vi và không thuộc phạm vi

**Trong phạm vi:**

- 43 lớp GTSRB, ảnh tĩnh.
- CNN tự xây dựng, không pretrained.
- Split 3 tầng theo sign-group, audit rò rỉ.
- Factorial 2⁴, kiến trúc, hyperparameter, augmentation.
- OOF, lockbox one-shot, official test raw/clean.
- Phân tích lỗi, robustness, calibration, Grad-CAM.
- Đo hiệu năng suy luận trên máy train.

**Không thuộc phạm vi:**

- Camera thật, video thời gian thực, thời tiết thật.
- Biển báo ngoài 43 lớp GTSRB (ví dụ biển Việt Nam).
- Huấn luyện lại trên máy cấu hình thấp.
- Tuyên bố mô hình đủ an toàn cho quyết định giao thông thực tế.

### 1.4. Đóng góp chính

1. **Split 3 tầng theo sign-group** thay cho split random bị rò rỉ frame.
2. **Factorial 2⁴** tách riêng BN, dropout, weight decay, augmentation.
3. **OOF 3 fold × 3 seed** để chọn mô hình, kèm CI cluster bootstrap theo track.
4. **Lockbox one-shot** mở một lần sau freeze, không dùng để chọn.
5. **Phân tích cơ chế:** per-class, confusion, Grad-CAM định lượng, calibration,
   robustness, error taxonomy.
6. **Đo vận hành:** latency theo batch/thiết bị, throughput, dung lượng mô hình,
   RAM/VRAM, thời gian đánh giá.

### 1.5. Cách đọc báo cáo: bốn tập đánh giá

| Tập | n | Vai trò | Dùng để kết luận? |
|---|---:|---|---|
| Screening train | 24.719 | Train cho factorial, kiến trúc, hyperparam, augmentation | Không |
| Screening val | 6.180 | Chọn cấu hình, early stopping, phân tích cơ chế | Không |
| OOF (3 fold × 3 seed) | 30.899 | Mỗi ảnh dev được dự đoán ngoài fold | Có |
| Lockbox | 8.302 | Test nội bộ độc lập, mở một lần sau freeze | Có |
| Official test raw | 12.630 | Benchmark phụ, không track-disjoint | Không |
| Official test clean | 11.879 | Benchmark phụ sau lọc 751 ảnh gần trùng | Không |

Quy tắc: **mỗi bảng chỉ dùng một tập**, tiêu đề bảng ghi rõ tập. Không đặt OOF
cạnh official test mà không ghi vai trò.

---

## Chương 2. Dữ liệu và chia tập

### 2.1. Nguồn dữ liệu

| Thông tin | Giá trị |
|---|---|
| Tên bộ dữ liệu | German Traffic Sign Recognition Benchmark (GTSRB) |
| Công bố | Stallkamp et al., Neural Networks 32 (2012) 323–332 |
| Đơn vị | Viện Neuroinformatik, Đại học Ruhr-Universität Bochum |
| Train gốc | 39.209 ảnh, 43 lớp |
| Test chính thức | 12.630 ảnh, 43 lớp |
| Định dạng | PPM (ảnh), CSV (nhãn + ROI) |
| Giấy phép | Dữ liệu miễn phí sử dụng, yêu cầu trích dẫn công bố gốc |
| Thời điểm dùng | 22/10/2026 |

Train được lấy từ archive chính thức `GTSRB_Final_Training_Images.zip` (39.209
ảnh), không dùng bản rút gọn 26.640 ảnh của torchvision. Test lấy qua
`torchvision.datasets.GTSRB`, nhãn đọc từ `GT-final_test.csv`.

### 2.2. Quy mô dữ liệu

**Bảng 2.1. Tóm tắt quy mô**

| Chỉ tiêu | Giá trị | Ghi chú |
|---|---:|---|
| Tổng ảnh train ban đầu | 39.209 | GTSRB Final_Training_Images |
| Ảnh train trùng MD5 với test | 8 | Đều lớp 14, track 00023 |
| Pool dùng chia split | 39.201 | 39.209 − 8 |
| Số lớp | 43 | 0–42 |
| Ảnh/lớp nhỏ nhất | 210 | Lớp 0 |
| Ảnh/lớp trung vị | 600 | — |
| Ảnh/lớp lớn nhất | 2.250 | Lớp 2 |
| Tỷ lệ mất cân bằng max/min | 10,714 | — |
| Số lớp dưới 300 ảnh | 9 | 0, 19, 24, 27, 29, 32, 37, 41, 42 |
| Ảnh test chính thức | 12.630 | — |
| Ảnh test gần trùng train | 751 | MAD < 5, cùng lớp |
| Ảnh test clean | 11.879 | 12.630 − 751 |
| Ảnh lockbox | 8.302 | 277 track, 238 sign-group |
| Ảnh dev | 30.899 | 1.030 track, 945 sign-group |
| Ảnh screening train / val | 24.719 / 6.180 | 824 / 206 track |
| Track toàn pool | 1.307 | 29–30 frame/track |
| Sign-group toàn pool | 1.183 | Gộp 39.603 cặp gần trùng |
| Ảnh trùng MD5 trong pool | 0 | SHA-256/MD5 |

Nguồn: `outputs/results/v3/tables/02b_dataset_summary.csv`,
`outputs/splits/v2/manifest.json`.

### 2.3. Phân bố 43 lớp

![Hình 2.1. Phân bố số ảnh theo lớp (đỏ: lớp dưới 300 ảnh)](outputs/figures/v3/01_class_distribution.png)

*Hình 2.1. 43 lớp xếp theo số ảnh; 9 lớp dưới 300 ảnh; 10 lớp lớn nhất chiếm khoảng 50% tổng ảnh.*

Điểm đáng chú ý:

- Lớp 0 (20 km/h) và lớp 19 (đường cong nguy hiểm bên trái) chỉ có 210 ảnh.
- Lớp 2 (50 km/h) có 2.250 ảnh, gấp 10,714 lần lớp ít nhất.
- 9 lớp dưới 300 ảnh: 0, 19, 24, 27, 29, 32, 37, 41, 42.
- Mất cân bằng này là lý do dùng **macro-F1/43** thay vì chỉ accuracy.

### 2.4. Chất lượng dữ liệu

![Hình 2.2. Phân bố độ nét, độ sáng, độ tương phản](outputs/figures/v3/16_data_quality.png)

*Hình 2.2. Laplacian variance (độ nét), độ sáng trung bình và độ tương phản trên pool 39.201 ảnh.*

**Bảng 2.2. Thống kê chất lượng ảnh**

| Chỉ số | Min | P10 | Trung vị | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Laplacian variance (độ nét) | 0,000014 | 0,000221 | 0,002396 | 0,016167 | 0,113549 |
| Độ sáng trung bình (0–1) | 0,026 | 0,104 | 0,294 | 0,587 | 0,971 |
| Độ tương phản (std xám) | 0,008 | 0,040 | 0,152 | 0,291 | 0,440 |
| Chiều rộng gốc (px) | 25 | 30 | 43 | 82 | 243 |
| Chiều cao gốc (px) | 25 | 30 | 43 | 81 | 225 |

**Bảng 2.3. Tính toàn vẹn dữ liệu**

| Kiểm tra | Kết quả |
|---|---|
| Ảnh đã quét | 51.839 (train + test) |
| Ảnh không đọc được | 0 |
| Ảnh ở chế độ RGB | 100% |
| Ảnh trùng MD5 trong pool | 0 |
| Ảnh train trùng MD5 với test | 8 (đã loại khỏi pool) |
| Cặp gần trùng được gộp sign-group | 39.603 |
| Ảnh test có ảnh train gần trùng | 751 (5,95%) |
| Ảnh tối (độ sáng < 0,10) | 3.596 (9,2%) |
| Ảnh chói (độ sáng > 0,80) | 558 (1,4%) |
| Ảnh dưới p10 độ nét | 3.920 (10%) |
| Ảnh dưới p10 tương phản | 3.920 (10%) |
| Nhãn kiểm tra thủ công | **Chưa** (xem mục 8.4) |

![Hình 2.3. Ví dụ các cực trị chất lượng ảnh](outputs/figures/v3/16b_data_quality_examples.png)

*Hình 2.3. Từ trái sang: tối nhất, sáng nhất, mờ nhất, nét nhất, tương phản thấp, trung vị.*

Kết luận: dữ liệu **không có ảnh hỏng**, nhưng có một tỷ lệ đáng kể ảnh tối/mờ.
Đây là yếu tố cần được tính đến khi phân tích lỗi, không phải lỗi của mô hình.

### 2.5. Tiền xử lý và augmentation

- Ảnh gốc kích thước thay đổi → `convert("RGB")` → resize bilinear 64×64.
- Cache `uint8` shape `(N, 64, 64, 3)` trong `.npy`, không commit.
- Mean/std tính **trên train của từng split**, lưu trong `manifest.mean_std`;
  không dùng thống kê toàn cục.
- Hàm `preprocess` duy nhất trong `src/data.py` dùng chung cho train/eval/demo.
- Augmentation hợp lệ: affine (xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5),
  color jitter (0,2/0,2/0,2/hue 0,02), perspective (0,1, p=0,3).
- **Cấm** horizontal flip, vertical flip, xoay 90/180; có test tự động chặn.

### 2.6. Chia tập 3 tầng

![Hình 2.4. Sơ đồ chia dữ liệu 3 tầng theo sign-group](outputs/figures/v3/02_split_diagram.png)

*Hình 2.4. Pool 39.201 → lockbox 8.302 / dev 30.899; dev → screening 24.719/6.180 và OOF 3 fold.*

**Bảng 2.4. Các tầng dữ liệu**

| Tầng | Quy tắc | Ảnh | Track | Sign-group | Seed |
|---|---|---:|---:|---:|---:|
| Pool | 39.209 − 8 ảnh trùng | 39.201 | 1.307 | 1.183 | — |
| Lockbox | 20% sign-group mỗi lớp | 8.302 | 277 | 238 | 2024 |
| Dev | Pool − lockbox | 30.899 | 1.030 | 945 | — |
| Screening train | 80% sign-group của dev | 24.719 | 824 | — | 42 |
| Screening val | 20% sign-group của dev | 6.180 | 206 | — | 42 |
| Fold 0 train / val | 2/3 – 1/3 sign-group của dev | 19.949 / 10.950 | — | — | 42 |
| Fold 1 train / val | — | 20.729 / 10.170 | — | — | 42 |
| Fold 2 train / val | — | 21.120 / 9.779 | — | — | 42 |
| Final train | Toàn bộ dev | 30.899 | 1.030 | 945 | 42/123/2024 |

**Vì sao dùng sign-group thay vì random?** Ba bằng chứng định lượng:

| Chỉ số | Val random | Val group/sign-group |
|---|---:|---:|
| % frame val có frame train cách ±1 | **95,22%** | 0% |
| Track train/val trùng | **1.306 / 1.307** | 0 |
| 1-NN (16×16 xám, không học) | **90,5%** | 31,0% |
| Val accuracy cùng model | 99,936% | 98,473% |
| Macro-F1/43 cùng model | 99,947% | 88,351%* |

*Group val cũ thiếu 4 lớp nên bị phạt cấu trúc 9,06 điểm; phần do leak thật khoảng 2,53 điểm.

Kết luận: val random **không đo được khả năng tổng quát**; nó đo mức độ gần nhau
của frame. Vì vậy mọi kết luận v3 dùng split theo sign-group.

### 2.7. Audit rò rỉ dữ liệu

**Bảng 2.5. Kết quả audit C1–C12**

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
| HASH | Hash 6 file split khớp manifest | Khớp | ✓ |
| MEANSTD | mean/std khớp lại từ cache | 0,0e+00 | ✓ |

Nguồn: `outputs/results/v2/audit/audit_report.json` (**15/15 PASS**).

### 2.8. Official test và clean set

Official test **không có track ID** và **không track-disjoint**. Audit gần trùng
test ↔ train (MAD < 5, cùng lớp, thumbnail 16×16 rồi kiểm chứng full 64×64):

| Mục | Giá trị |
|---|---:|
| Cặp gần trùng | 24.554 |
| Ảnh test có ít nhất một ảnh train gần trùng | 751 (5,95%) |
| Clean | 11.879 |
| Raw | 12.630 |

Clean **cao hơn** raw (98,8692 so với 98,6989), nên không có dấu hiệu thổi phồng.
Official test vẫn chỉ là **benchmark phụ**, không dùng để chọn mô hình hay kết luận
chính.

### 2.9. Tóm tắt chương 2

- 39.201 ảnh pool, 43 lớp, mất cân bằng 10,714 lần.
- 0 ảnh hỏng; 3.596 ảnh tối, 3.920 ảnh dưới p10 độ nét.
- Split 3 tầng theo sign-group; audit 15/15 PASS.
- Lockbox 8.302 ảnh, one-shot; official test 12.630 ảnh, chỉ benchmark phụ.

---

## Chương 3. Kiến trúc và môi trường

### 3.1. CNN hoạt động như thế nào

Một CNN phân loại ảnh gồm các thành phần:

- **Convolution 3×3:** trượt bộ lọc trên ảnh để học đặc trưng cục bộ (viền, màu,
  hình dạng).
- **Batch Normalization:** chuẩn hóa đầu ra từng kênh, giúp tối ưu ổn định hơn.
- **ReLU:** phi tuyến, giữ tín hiệu dương.
- **Max Pooling 2×2:** giảm kích thước không gian, tăng vùng tiếp nhận.
- **Flatten:** chuyển feature map thành vector.
- **Fully Connected:** kết hợp đặc trưng thành logits 43 lớp.
- **Dropout:** tắt ngẫu nhiên một phần neuron khi train để giảm overfit.

Mô hình trả **logits**, không softmax trong model; softmax chỉ dùng khi suy luận.

### 3.2. Các cấu hình mô hình

**Bảng 3.1. So sánh kiến trúc**

| Mô hình | Khối | Widths | BN | Dropout | GAP | FC hidden | Params |
|---|---|---:|---|---:|---|---:|---:|
| CNN-A (F00) | 3 | 32/64/128 | Không | 0 | Không | 256 | 2.201.707 |
| CNN-B (F07/F15) | 3 | 32/64/128 | Có | 0,3 | Không | 256 | 2.201.931 |
| CNN-C (D3) | 4 | 32/64/128/256 | Có | 0,4 | Có | — | 399.947 |
| **D2 (winner)** | 4 | 32/64/128/256 | Có | 0,3 | Không | 256 | **1.448.779** |
| D1 | 3 | 32/64/128 | Có | 0,3 | Có | — | 99.019 |
| W1 | 3 | 16/32/64 | Có | 0,3 | Không | 256 | 1.083.579 |
| W2 | 3 | 48/96/192 | Có | 0,3 | Không | 256 | 3.366.363 |

### 3.3. Kiến trúc D2

![Hình 3.1. Kiến trúc D2](outputs/figures/v3/03_architecture_d2.png)

*Hình 3.1. D2: 4 khối conv 32/64/128/256 + BN + ReLU + maxpool; flatten 4×4×256 = 4.096; FC 256; dropout 0,3; FC 43 logits.*

**Bảng 3.2. Thông số từng lớp của D2**

| Lớp | Đầu ra | Params |
|---|---|---:|
| Conv 3×3, 3→32 + BN | 32×64×64 | 928 |
| MaxPool 2×2 | 32×32×32 | 0 |
| Conv 3×3, 32→64 + BN | 64×32×32 | 18.560 |
| MaxPool 2×2 | 64×16×16 | 0 |
| Conv 3×3, 64→128 + BN | 128×16×16 | 73.984 |
| MaxPool 2×2 | 128×8×8 | 0 |
| Conv 3×3, 128→256 + BN | 256×8×8 | 295.424 |
| MaxPool 2×2 | 256×4×4 | 0 |
| Flatten | 4.096 | 0 |
| FC 4.096→256 + ReLU + Dropout 0,3 | 256 | 1.048.832 |
| FC 256→43 | 43 | 11.051 |
| **Tổng** | **43 logits** | **1.448.779** |

### 3.4. Môi trường phần cứng và phần mềm

**Bảng 3.3. Phần cứng**

| Thành phần | Giá trị |
|---|---|
| CPU | Intel Core i9-14900KF, 24 nhân, 32 luồng |
| GPU | NVIDIA GeForce RTX 5070, 11,94 GB VRAM, capability 12.0 |
| RAM | 63,84 GB |
| Ổ cứng | Lexar SSD NM710 2 TB (NVMe) |
| Hệ điều hành | Windows 11 Pro, build 26200, 64-bit |
| Driver GPU | 610.88 |
| CUDA / cuDNN | CUDA 12.8 / cuDNN 9.19 |

**Bảng 3.4. Phần mềm**

| Thư viện | Phiên bản |
|---|---|
| Python | 3.12.10 (CPython 64-bit) |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| NumPy | 2.5.2 |
| Pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |

### 3.5. Cấu hình huấn luyện

**Bảng 3.5. Giao thức huấn luyện cố định**

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (L1 test 1e-4) |
| Weight decay | 1e-4 (factorial test 0) |
| Batch size | 128 |
| Epochs | Tối đa 40 |
| Scheduler | Cosine annealing, không warmup |
| AMP | Bật khi CUDA |
| Early stopping | Patience 7 theo val macro-F1 |
| Label smoothing | 0 |
| Class weight | none (CW1 test inv_sqrt) |
| Input | 64×64 RGB (I2 test 96) |
| Mean/std | Theo train của từng split |
| Seed | 42, 123, 2024 |
| num_workers | 4–8 |
| Deterministic | Bật (`torch.use_deterministic_algorithms`, warn_only) |

Early stopping chỉ dùng val của split đang train; **không** dùng lockbox/test.

---

## Chương 4. Thiết kế thực nghiệm

### 4.1. Câu hỏi nghiên cứu

1. Chia dữ liệu thế nào để không rò rỉ frame?
2. BN, dropout, weight decay, augmentation có thực sự giúp không?
3. 3 khối hay 4 khối, GAP hay flatten, width nào?
4. Learning rate, class weight, input size nào phù hợp?
5. Augmentation thành phần nào có lợi, thành phần nào hại?
6. Mô hình nào thắng khi đánh giá bằng OOF 3 fold × 3 seed?
7. Kết quả OOF có được xác nhận trên lockbox độc lập không?
8. Mô hình yếu ở đâu: lớp nào, điều kiện ảnh nào, lỗi gì?

### 4.2. Baseline

- **B0 majority:** luôn dự đoán lớp nhiều nhất. Accuracy 2,9126%; macro-F1 0,1316%.
- **B1 HOG + LinearSVC:** HOG trên ảnh xám 64×64, 9 orientations, cell 8×8,
  block 2×2, L2-Hys → 1.764 chiều; `StandardScaler`; `LinearSVC` OvR;
  C ∈ {0,1; 1; 10} chọn theo macro-F1 trên screening val.
- Kết quả B1: C = 0,1; accuracy 87,2654%; macro-F1 85,9968%; 75.895 params;
  trích HOG 40,72 s; fit 452,74 s.
- B1 không refit trên dev, không đánh giá trên lockbox.

### 4.3. Factorial 2⁴

Cố định: 3 khối 32/64/128, FC 256, GAP off, input 64, batch 128, LR 1e-3,
40 epoch, patience 7, 3 seed. Bốn yếu tố bật/tắt:

| Yếu tố | Mức 0 | Mức 1 |
|---|---|---|
| BN | off | on |
| Dropout | 0 | 0,3 |
| Weight decay | 0 | 1e-4 |
| Augmentation | off | all |

16 tổ hợp F00–F15. Main effect = trung bình F1 của 8 cấu hình mức 1 trừ trung
bình 8 cấu hình mức 0. Interaction BN×Aug và DO×WD dùng contrast tích, kiểm
chứng bằng paired bootstrap theo track.

### 4.4. Thực nghiệm kiến trúc

- **Depth:** D1 (3 khối + GAP) so với D2 (4 khối + flatten).
- **GAP:** D3 (4 khối + GAP) so với D2 (4 khối + flatten).
- **Width:** W1 (16/32/64) và W2 (48/96/192) so với D2 (32/64/128/256).
- Mỗi cấu hình chạy 3 seed trên screening.

### 4.5. Hyperparameter

- **LR:** L1 = 1e-4 so với 1e-3.
- **Class weight:** CW1 = inv_sqrt(tần suất lớp).
- **Input size:** I2 = 96×96 (cache riêng, không ghi đè cache 64).

### 4.6. Augmentation và corruption

- **G1 affine:** xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5.
- **G2 color:** brightness/contrast/saturation 0,2, hue 0,02.
- **G3 perspective:** distortion 0,1, p=0,3.
- **G0:** không augmentation (F07).
- **Corruption:** blur, noise, brightness, contrast, JPEG, occlusion; mỗi loại
  3 mức; đánh giá trên screening val seed 42.

### 4.7. OOF, lockbox và official test

- **OOF:** chia sign-group của dev thành 3 fold cân bằng frame; mỗi ảnh dev được
  dự đoán đúng một lần mỗi seed; 3 cấu hình X1/X2/X3 × 3 seed × 3 fold = 27 run.
- **Lockbox:** 8.302 ảnh, mở một lần sau freeze; 3 seed final; không dùng để chọn.
- **Official test:** raw 12.630 và clean 11.879; chỉ benchmark phụ.

### 4.8. Metric và thống kê

- **Accuracy:** top-1.
- **Macro-F1/43:** `f1_score(labels=range(43), average="macro", zero_division=0)`.
  Lớp vắng mặt vẫn bị tính 0.
- **Top-3:** nhãn đúng nằm trong 3 dự đoán cao nhất.
- **ECE 15 bin:** trung bình có trọng số |accuracy − confidence| theo bin.
- **Brier đa lớp:** trung bình tổng bình phương xác suất so với one-hot.
- **CI 95%:** cluster bootstrap theo track, B = 2.000:

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
- **Phân rã phương sai:** ANOVA 2 chiều seed × fold, báo var_seed, var_fold,
  var_resid, ICC.

### 4.9. Luật quyết định (đăng ký trước)

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

### 4.10. Ma trận thực nghiệm và chi phí

**Bảng 4.1. Số run theo nhóm**

| Block | Cấu hình | Seed/fold | Run | Vai trò |
|---|---|---:|---:|---|
| Baseline B1 | 1 | 1 | 1 | Mốc cổ điển |
| F factorial | 16 | 3 | 48 | BN/DO/WD/Aug |
| D architecture | 5 | 3 | 15 | GAP/depth/width |
| H hyperparam | 3 | 3 | 9 | LR/class weight/input |
| G augmentation | 3 | 3 | 9 | affine/color/perspective |
| X confirmation | 3 | 3 fold × 3 seed | 27 | OOF + control |
| Final | 1 | 3 | 3 | Train dev, test lockbox |
| **Tổng** | | | **111 CNN + 1 baseline** | |

**Bảng 4.2. Chi phí huấn luyện theo nhóm**

| Nhóm | Run | Tổng epoch | Tổng thời gian (phút) | TB/run (s) | TB/epoch (s) | GPU mem max (MB) |
|---|---:|---:|---:|---:|---:|---:|
| D_architecture | 15 | 496 | 45,0 | 179,9 | 5,44 | 464,7 |
| FINAL | 3 | 78 | 7,9 | 158,2 | 6,08 | 248,3 |
| F_factorial | 48 | 1.387 | 86,4 | 108,0 | 3,74 | 318,9 |
| G_augmentation | 9 | 260 | 13,7 | 91,7 | 3,17 | 318,9 |
| H_hyperparam | 9 | 298 | 31,8 | 212,3 | 6,41 | 693,9 |
| X_confirmation | 27 | 892 | 54,6 | 121,3 | 3,67 | 318,9 |
| **Tổng CNN** | **111** | **3.411** | **239,5** | 129,5 | **4,21** | **693,9** |
| B1 HOG+SVC | 1 | — | 8,2 | 493,5 | — | — |

Tổng: **3,99 giờ epoch-loop** + 8,2 phút baseline = **4,13 giờ**. Phạm vi đo là
vòng epoch, không gồm nạp cache, dự đoán val cuối, phân tích, hình, báo cáo.

---

## Chương 5. Kết quả thực nghiệm

> Toàn bộ số trong chương này lấy từ `outputs/results/v3/`. Số in đậm là số dùng
> để kết luận. Bảng đầy đủ ở `outputs/results/v3/tables/`.

### 5.1. Baseline

**Bảng 5.1. Kết quả baseline trên screening val**

| Mô hình | Accuracy | Macro-F1/43 | Params | Thời gian fit |
|---|---:|---:|---:|---:|
| B0 majority | 2,9126% | 0,1316% | 0 | — |
| B1 HOG + LinearSVC (C=0,1) | 87,2654% | 85,9968% | 75.895 | 452,74 s |

CNN tốt nhất trên cùng screening val (D2) đạt **97,1584% macro-F1**, cao hơn B1
**11,16 điểm**. B1 không được refit trên dev và không đánh giá trên lockbox.

### 5.2. Factorial 2⁴

**Bảng 5.2. 16 cấu hình factorial (trung bình 3 seed, screening val)**

| ID | BN | DO | WD | Aug | Acc (%) | Macro-F1 (%) | Δ vs F00 (điểm %) [CI] | Gap | Time (s) |
|---|---|---|---|---|---|---:|---:|---:|---:|
| F00 | off | 0 | 0 | off | 90,092 ± 1,902 | 84,109 ± 3,710 | — | 9,87 | 52,6 |
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

![Hình 5.1. Hiệu ứng chính và tương tác của factorial](outputs/figures/v3/08_factor_effects.png)

*Hình 5.1. BN và dropout dương rõ; weight decay và augmentation không đủ bằng chứng; tương tác nhỏ.*

**Bảng 5.3. Hiệu ứng chính (điểm %)**

| Yếu tố | Δ | 95% CI | Kết luận |
|---|---:|---|---|
| BN | **+3,724** | [+1,767; +5,005] | Tốt hơn |
| Dropout 0,3 | **+5,963** | [+2,873; +7,147] | Tốt hơn |
| Weight decay 1e-4 | +0,058 | [−0,506; +0,551] | Không đủ bằng chứng |
| Augmentation all | +0,529 | [−1,204; +2,303] | Không đủ bằng chứng |
| BN × Aug | −0,279 | [−1,058; +0,551] | Không rõ |
| Dropout × WD | −0,121 | [−0,686; +0,548] | Không rõ |

**Đọc kỹ tương tác:** augmentation giúp khi chưa có regularization (F08 − F00 =
**+2,563**) nhưng hại khi đã đủ regularization (F15 − F07 = **−2,555**). Vì vậy
không được tuyên bố "augmentation giúp X điểm" chung chung.

### 5.3. Kiến trúc

![Hình 5.2. So sánh Accuracy và Macro-F1 các cấu hình](outputs/figures/v3/05_model_comparison.png)

*Hình 5.2. Trục phóng to 88–100; cột trái là số tham số; D2 và X2 được in đậm.*

**Bảng 5.4. Kết quả kiến trúc (screening val, trung bình 3 seed)**

| ID | Thay đổi so với D2 | Acc (%) | Macro-F1 (%) | Δ vs D2 (điểm %) [CI] | Params | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|---:|
| D1 | GAP, 3 khối | 74,234 ± 0,592 | 59,188 ± 0,764 | **−34,758** [−36,181; −22,916] | 99.019 | 213,2 | 40,0 |
| D2 | 4 khối, flatten | 98,004 ± 0,297 | **97,158 ± 0,610** | — | 1.448.779 | 132,8 | 23,3 |
| D3 | 4 khối + GAP + DO 0,4 | 96,575 ± 0,580 | 93,105 ± 0,906 | −0,841 [−2,724; +0,757] | 399.947 | 184,9 | 34,0 |
| W1 | 16/32/64 | 94,876 ± 0,916 | 90,686 ± 1,425 | −3,260 [−5,250; −0,735] | 1.083.579 | 181,9 | 33,7 |
| W2 | 48/96/192 | 97,104 ± 0,368 | 95,215 ± 1,230 | +1,268 [−0,585; +2,499] | 3.366.363 | 186,9 | 34,3 |

- **GAP 3 khối thất bại rõ:** 99k params, train acc 84,6% (seed 42), val trung bình
  74,2% sau 40 epoch → underfit vì thiếu dung lượng/tối ưu, không phải "GAP mất
  chi tiết" đơn thuần.
- **4 khối tốt hơn 3 khối** 3,212 điểm, CI không chứa 0 → bác bỏ giả thuyết H4.
- D3 tương đương D2 → chỉ dùng map E4 cũ, không kết luận cơ chế.
- Width nhỏ kém; width lớn +1,268 nhưng CI chứa 0 → chưa kết luận, params 3,37M.

### 5.4. Hyperparameter

**Bảng 5.5. Kết quả hyperparameter (screening val, trung bình 3 seed)**

| ID | Thay đổi | Acc (%) | Macro-F1 (%) | Δ vs F15 (điểm %) [CI] | Params | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|---:|
| L1 | LR 1e-4 | 91,888 ± 1,282 | 86,807 ± 1,615 | **−7,139** [−9,372; −3,353] | 2.201.931 | 160,5 | 29,0 |
| CW1 | class weight inv_sqrt | 95,960 ± 0,606 | 92,943 ± 0,475 | −1,004 [−2,195; +0,261] | 2.201.931 | 183,3 | 33,7 |
| I2 | input 96 | 94,213 ± 0,624 | 90,499 ± 1,429 | **−3,448** [−5,463; −0,984] | 4.823.371 | 293,1 | 36,7 |

- LR 1e-3 > 1e-4 rõ rệt (1e-4 không hội tụ trong 40 epoch).
- Class weight không cải thiện (CI chứa 0) → bỏ; đây là exploratory.
- Input 96 kém hơn 64 và params tăng 2,2× → giữ 64.

### 5.5. Augmentation và robustness

**Bảng 5.6. Augmentation thành phần (screening val, trung bình 3 seed)**

| ID | Profile | Acc (%) | Macro-F1 (%) | Δ vs G0 (điểm %) [CI] | Time (s) | Epochs |
|---|---|---:|---:|---|---:|---:|
| G1 | affine | 95,572 ± 0,190 | 92,251 ± 0,595 | **−4,250** [−6,972; −0,084] | 92,1 | 36,0 |
| G2 | color | 97,767 ± 0,242 | **97,417 ± 0,340** | +0,917 [−0,141; +1,842] | 118,7 | 28,3 |
| G3 | perspective | 96,375 ± 1,116 | 95,264 ± 1,988 | −1,237 [−2,706; +0,530] | 64,2 | 22,3 |

**Bảng 5.7. Corruption robustness (accuracy trung bình 3 mức, screening val, seed 42)**

| Config | Blur | Noise | Bright | Contrast | JPEG | Occlusion | TB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F07 (G0) | 95,05 | 86,27 | 97,13 | 93,49 | 94,00 | 23,11 | 81,51 |
| F15 (G5) | 93,73 | 81,00 | 96,19 | 92,77 | 91,22 | 26,43 | 80,22 |
| G1 affine | 91,35 | 75,84 | 94,75 | 86,81 | 88,38 | 24,24 | 76,90 |
| G2 color | 95,73 | 85,86 | 97,56 | 96,06 | 95,07 | 21,56 | **81,97** |
| G3 perspective | 94,06 | 84,85 | 96,78 | 93,65 | 93,49 | 21,44 | 80,71 |
| D2 (4 khối + all) | 93,88 | 83,59 | 97,10 | 92,92 | 92,78 | 17,63 | 79,65 |

![Hình 5.3. Corruption curves 6 loại × 3 mức](outputs/figures/v3/10_robustness.png)

*Hình 5.3. Color-only (G2) robust nhất; affine (G1) kém nhất; occlusion làm sập mọi mô hình.*

Kết luận: "all" không có lợi (clean −2,555 so với F07; corruption TB thấp hơn);
affine là thành phần hại; color trung tính (Δ +0,917, CI chứa 0) và corruption
+0,46 — dưới ngưỡng +0,5 nên **không đủ bằng chứng để giữ augmentation vì robustness**.

### 5.6. OOF 3 fold × 3 seed

**Bảng 5.8. Kết quả OOF (30.899 ảnh dev, trung bình 3 seed)**

| Config | Nguồn | OOF macro-F1 (%) | CI 95% |
|---|---|---:|---|
| X1 | G2 (3 khối + color) | 96,2971 ± 0,296 | [94,628; 97,020] |
| **X2** | **D2 (4 khối + all)** | **97,0400 ± 0,1869** | **[95,051; 97,942]** |
| X3 | F07 (control) | 95,0588 ± 0,568 | [93,113; 95,930] |

![Hình 5.4. OOF theo fold × seed và phân rã phương sai](outputs/figures/v3/09_oof_fold_seed.png)

*Hình 5.4. Fold là nguồn biến thiên chính (ICC fold 0,86–0,95); seed gần như không ảnh hưởng.*

**Bảng 5.9. So sánh cặp trên OOF**

| So sánh | Δ (điểm %) | 95% CI | Kết luận |
|---|---:|---|---|
| X1 − X2 | −0,743 | [−1,590; +0,274] | Chưa kết luận |
| X1 − X3 | **+1,238** | [+0,586; +2,071] | X1 tốt hơn control |
| X2 − X3 | **+1,981** | [+1,053; +2,961] | X2 tốt hơn control |

OOF có **278.091 dòng** = 30.899 ảnh × 3 seed × 3 config; mỗi ảnh dev được dự
đoán đúng một lần mỗi seed.

**Phân rã phương sai:** var_seed = 4,42e−06 (**0,9%**), var_fold = 4,20e−04
(**91,0%**), var_resid = 3,82e−05 (8,3%). Kết luận: báo cáo theo fold là bắt
buộc; multi-seed một mình không đủ.

### 5.7. Lockbox và official test

**Bảng 5.10. Kết quả trên ba tập độc lập hơn**

| Tập | n | Accuracy | Macro-F1/43 | Top-3 | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Lockbox** | 8.302 | **98,8758 ± 0,1010** | **97,5981 ± 0,6246** | 99,8474 | 0,005328 | 0,017862 |
| Official raw | 12.630 | 98,6989 ± 0,1625 | 98,0398 ± 0,4516 | 99,7677 | 0,005932 | 0,020196 |
| Official clean | 11.879 | 98,8692 ± 0,2359 | 98,2499 ± 0,5314 | 99,7671 | 0,005268 | 0,017925 |

- Lockbox top-3 99,8474%; ECE 0,005328; Brier 0,017862.
- Ba seed lockbox: accuracy 98,7593 / 98,9400 / 98,9280; F1 96,8770 / 97,9706 / 97,9468.
- Official clean **cao hơn** raw → ảnh gần trùng không thổi phồng.
- Official test không track-disjoint; không so trực tiếp với lockbox.

**Về bất định của lockbox:** artifact one-shot chỉ có 3 seed và aggregate
per-class, không có per-image predictions để bootstrap theo track. Vì vậy báo cáo
chỉ đưa **mean ± sd + range 3 seed**; riêng accuracy có thêm binomial CI
[98,649; 99,103] (n = 8.302). **Không tái sử dụng CI của OOF cho lockbox.**

### 5.8. Bảng tổng hợp cuối cùng

**Bảng 5.11. Bảng một chỗ cho người đọc**

| Tập | n | Accuracy | Macro-F1 | Vai trò | So sánh trực tiếp? |
|---|---:|---:|---:|---|---|
| Screening val (D2) | 6.180 | 98,0043 | 97,1584 | Chọn cấu hình | Không |
| OOF (X2) | 30.899 | 97,9244 | 97,0400 | Chọn winner | Không |
| Lockbox | 8.302 | 98,8758 | 97,5981 | Test nội bộ độc lập | Có (chuẩn chính) |
| Official test raw | 12.630 | 98,6989 | 98,0398 | Benchmark phụ | Không |
| Official test clean | 11.879 | 98,8692 | 98,2499 | Benchmark phụ | Không |

### 5.9. Tám so sánh chính (Holm)

**Bảng 5.12. Hiệu chỉnh Holm–Bonferroni**

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

---

## Chương 6. Vì sao chọn D2

### 6.1. Tiêu chí

1. OOF macro-F1/43 cao nhất trên 30.899 ảnh dev.
2. CI so với control không chứa 0.
3. Nếu hai ứng viên tương đương: ít tham số hơn, latency thấp hơn, ít
   augmentation hơn, ít epoch hơn.
4. Kết quả lockbox phải xác nhận, không được mâu thuẫn.

### 6.2. So sánh với các ứng viên

| Ứng viên | OOF macro-F1 | Δ vs control | Δ vs D2 | Params |
|---|---:|---:|---:|---:|
| X1 (G2, 3 khối + color) | 96,2971 | +1,238 | −0,743 | 2.201.931 |
| **X2 (D2, 4 khối + all)** | **97,0400** | **+1,981** | — | **1.448.779** |
| X3 (F07, control) | 95,0588 | — | −1,981 | 2.201.931 |

### 6.3. Bằng chứng và giới hạn

- X2 hơn control **+1,981 điểm**, CI [+1,053; +2,961] → không chứa 0.
- X1 − X2 = **−0,743**, CI [−1,590; +0,274] → **chưa kết luận**.
- X2 ít tham số hơn X1 **34%** (1,45M so với 2,20M) → thắng tie-break.
- Lockbox 97,5981% xác nhận OOF không bị thổi phồng.
- BN/dropout là bằng chứng mạnh nhất; depth có CI ủng hộ nhưng p Holm 0,373.

### 6.4. Điều không được nói

**Không được nói:** "D2 thắng nhờ augmentation all."

**Sự thật:** main effect augmentation +0,529, CI [−1,204; +2,303]; F15 − F07 =
−2,555 điểm trên clean; corruption trung bình của D2 (79,65) thấp hơn F07
(81,51). Augmentation "all" **không được chứng minh là có lợi**.

**Cách nói đúng:** "D2 thắng OOF nhờ cấu hình tổng thể 4 khối + BN/dropout;
bằng chứng mạnh nhất là BN/dropout, depth có CI ủng hộ nhưng p Holm 0,373.
Augmentation 'all' không được chứng minh là có lợi; đây là lựa chọn theo OOF
tổng thể và tie-break, không phải bằng chứng cho augmentation."

---

## Chương 7. Đánh giá từng lớp

### 7.1. Tổng quan

![Hình 7.1. F1 từng lớp trên lockbox](outputs/figures/v3/07_per_class_f1.png)

*Hình 7.1. 43 lớp xếp từ yếu nhất (trên) đến tốt nhất (dưới); thanh lỗi là độ lệch chuẩn 3 seed.*

**Bảng 7.1. 5 lớp yếu nhất và 5 lớp tốt nhất trên lockbox**

| Nhóm | Lớp | Tên | Support | F1 trung bình |
|---|---|---:|---|---:|
| Yếu nhất | 19 | Đường cong nguy hiểm bên trái | 30 | 0,7601 |
| Yếu nhất | 29 | Xe đạp qua đường | 60 | 0,7762 |
| Yếu nhất | 30 | Cảnh báo băng tuyết | 90 | 0,9252 |
| Yếu nhất | 28 | Trẻ em qua đường | 120 | 0,9276 |
| Yếu nhất | 23 | Đường trơn | 120 | 0,9381 |
| Tốt nhất | 34 | Rẽ trái phía trước | 90 | 1,0000 |
| Tốt nhất | 37 | Đi thẳng hoặc rẽ trái | 30 | 1,0000 |
| Tốt nhất | 38 | Đi bên phải | 360 | 1,0000 |
| Tốt nhất | 39 | Đi bên trái | 60 | 1,0000 |
| Tốt nhất | 42 | Hết cấm vượt cho xe trên 3,5 tấn | 60 | 1,0000 |

### 7.2. Quan hệ giữa số lượng ảnh và độ chính xác

- Tương quan Spearman giữa support và F1: **ρ = 0,126; p = 0,419** (n = 43 lớp).
- Tương quan Pearson: r = 0,277; p = 0,072.
- Kết luận: **không có quan hệ rõ ràng**. Lớp hiếm không phải lúc nào cũng kém;
  lớp 8 (120 km/h, 660 ảnh) có số lỗi ước tính cao thứ hai, trong khi lớp 37
  (30 ảnh) đạt F1 = 1,000.

### 7.3. Nhận xét

- Lớp 19 và 29 yếu vì **nhầm với lớp cùng hình dạng**: 19 → 23 (45% lớp thật),
  29 → 28 (22,8%).
- Lớp 30 (băng tuyết) nhầm với 11 (đường ưu tiên) — cùng tam giác đỏ.
- Lớp 28 (trẻ em) nhầm với 29 (xe đạp) — cùng tam giác, hình người/xe nhỏ.
- Các lớp biển xanh (33–40) đạt F1 cao vì màu và hình mũi tên khác biệt.
- Bảng đầy đủ 43 lớp: Phụ lục A và `outputs/results/v3/tables/07_per_class_lockbox.csv`.

---

## Chương 8. Phân tích lỗi và nhầm lẫn

### 8.1. Confusion matrix

![Hình 8.1. Confusion matrix lockbox, chuẩn hóa theo hàng](outputs/figures/v3/06_confusion_matrix.png)

*Hình 8.1. 43×43, trung bình 3 seed; màu đậm là tỷ lệ cao; panel phải là 10 cặp nhầm nhiều nhất.*

### 8.2. Các cặp nhầm nhiều nhất

**Bảng 8.1. Top 10 cặp nhầm trên lockbox**

| Thật | Dự đoán | Số ảnh (TB 3 seed) | % lớp thật | Cùng hình dạng | Cùng màu |
|---|---|---:|---:|---|---|
| 32 | 41 | 14,0 | 23,3% | Có | Có |
| 29 | 28 | 13,7 | 22,8% | Có | Có |
| 19 | 23 | 13,5 | 45,0% | Có | Có |
| 8 | 5 | 11,5 | 1,7% | Có | Có |
| 13 | 10 | 9,0 | 3,3% | Không | Có |
| 29 | 25 | 6,3 | 10,6% | Có | Có |
| 30 | 11 | 5,7 | 6,3% | Có | Có |
| 20 | 25 | 5,0 | 8,3% | Có | Có |
| 8 | 2 | 5,0 | 0,8% | Có | Có |
| 21 | 23 | 4,5 | 7,5% | Có | Có |

Trong 10 cặp trên, **9/10 cùng hình dạng** và **10/10 cùng màu**; tính cả 19 cặp
trong danh sách thì 17/19 cùng hình dạng. Đây là bằng chứng định lượng cho giả
thuyết "nhầm do hai biển có thiết kế giống nhau", không phải do mô hình nhìn nền.

### 8.3. Phân loại lỗi tự động

**Bảng 8.2. Phân nhóm heuristic 144 lỗi của D2 trên screening val**

| Nhóm | Số ảnh | Cách xác định |
|---|---:|---|
| Góc nhìn lạ / nhãn khó | 36 | Không thuộc các nhóm dưới |
| Confidence thấp | 32 | Confidence < 0,60 |
| Tự tin cao nhưng sai | 27 | Confidence ≥ 0,90 |
| Tối | 21 | Độ sáng < p5 |
| Mờ | 16 | Laplacian variance < p10 |
| Chói | 12 | Độ sáng > p95 |
| **Tổng** | **144** | 23 track |

Đây là **phân nhóm tự động theo percentile**, không phải kết quả kiểm tra bằng
mắt. Một ảnh có thể thỏa nhiều điều kiện; nhóm được gán theo thứ tự ưu tiên.

### 8.4. Kiểm tra thủ công: trạng thái trung thực

- Đã chuẩn bị **41 ảnh đại diện** (tối đa 2 ảnh/track, 23 track) trong
  `outputs/results/v3/tables/17b_manual_error_review_diverse.csv`.
- Hai contact sheet: `outputs/figures/v3/contact_sheets/error_sheet_01.png` và
  `error_sheet_02.png`.
- **Cột `verdict`, `cause`, `note` vẫn để trống.** Chưa có kết quả kiểm tra bằng
  mắt của con người.
- Vì vậy báo cáo **không kết luận** "bao nhiêu % lỗi do dữ liệu" hay "do mô
  hình". Các nhóm ở mục 8.3 chỉ là giả thuyết cần kiểm chứng.

### 8.5. Ví dụ dự đoán

![Hình 8.2. Ví dụ dự đoán đúng/sai theo confidence](outputs/figures/v3/15_prediction_examples.png)

*Hình 8.2. Bốn nhóm: đúng tự tin cao, đúng tự tin thấp, sai tự tin cao, sai tự tin thấp.*

Các ca sai tự tin cao thường là **nhầm giữa các biển giới hạn tốc độ** (80 → 120,
70 → 30) hoặc **biển tam giác cùng hình dạng** (11 → 30). Đây là lỗi phân biệt
lớp, không phải lỗi nhìn nền.

### 8.6. Lỗi do dữ liệu hay do mô hình?

| Loại | Bằng chứng | Kết luận |
|---|---|---|
| Dữ liệu tối/mờ | 3.596 ảnh tối, 3.920 ảnh dưới p10 độ nét | Góp phần, chưa định lượng |
| Nhãn sai | Chưa kiểm tra thủ công | **Chưa kết luận** |
| Nhầm lớp giống nhau | 9/10 cặp cùng hình dạng, 10/10 cùng màu | Có bằng chứng |
| Nhìn nền | Grad-CAM mass trong ROI 0,853 khi đúng, 0,733 khi sai | Không phải nguyên nhân chính |
| Mất cân bằng lớp | Support–F1 tương quan yếu (ρ = 0,126) | Không phải nguyên nhân chính |

---

## Chương 9. Robustness và khả năng giải thích

### 9.1. Robustness dưới corruption

Bảng 5.7 đã trình bày accuracy trung bình 3 mức cho 6 loại corruption. Điểm đáng
chú ý:

- **Occlusion là điểm yếu lớn nhất:** D2 chỉ còn **17,63%** accuracy khi che
  10–30% trung tâm; F07 23,11%; G2 21,56%.
- **Blur và contrast:** D2 93,88% và 92,92%; không tệ hơn nhiều so với clean.
- **Noise:** D2 83,59%, thấp hơn blur/contrast nhưng vẫn chấp nhận được.
- **JPEG:** D2 92,78%; ảnh nén không phải vấn đề lớn.
- **Brightness:** D2 97,10%; gần như không giảm.

Kết luận: mô hình **không bền với che khuất**, nhưng bền với thay đổi ánh sáng và
nén ảnh. Đây là hạn chế thật, không phải lỗi đánh giá: khi biển bị che mất phần
quyết định, mô hình không còn đủ thông tin.

### 9.2. Grad-CAM: mô hình nhìn vào đâu?

![Hình 9.1. Grad-CAM của D2 trên screening val](outputs/figures/v3/11_gradcam.png)

*Hình 9.1. 3 nhóm: đúng tự tin cao, đúng tự tin thấp, sai; viền trắng là ROI của biển.*

**Bảng 9.1. Tỷ lệ mass của Grad-CAM nằm trong ROI**

| Nhóm | n | Mass trong ROI (mean ± sd) | Median |
|---|---:|---:|---:|
| Đúng | 178 | 0,8529 ± 0,1445 | 0,8853 |
| Sai | 42 | 0,7332 ± 0,2456 | 0,8725 |
| Mức ngẫu nhiên | — | 0,5435 (diện tích ROI) | — |

Diễn giải: CAM tập trung trong ROI **nhiều hơn mức ngẫu nhiên** ở cả ca đúng và
sai. Ca sai vẫn nhìn vào biển, nên lỗi chủ yếu là **phân biệt lớp**, không phải
nhìn nền. Tuy nhiên, ca sai có mass thấp hơn và phương sai lớn hơn, cho thấy mô
hình bớt tập trung khi sai.

### 9.3. Calibration và confidence

![Hình 9.2. Reliability diagram, phân bố confidence, selective risk](outputs/figures/v3/12_calibration.png)

*Hình 9.2. Trái: reliability 15 bin kèm CI 95%; giữa: confidence đúng/sai; phải: accuracy theo coverage.*

**Bảng 9.2. Calibration trên screening val (D2, seed 42)**

| Chỉ số | Giá trị |
|---|---:|
| Số ảnh | 6.180 |
| Accuracy | 97,6699% |
| Confidence trung bình | 0,9809 |
| Confidence trung bình khi đúng | 0,9881 |
| Confidence trung bình khi sai | 0,6795 |
| Số dự đoán sai | 144 |
| Sai nhưng confidence ≥ 0,90 | 28 |
| Đúng nhưng confidence < 0,50 | 35 |
| ECE 15 bin | 0,008354 |
| Brier | 0,035747 |

Lockbox có ECE **0,005328** và Brier **0,017862** — mô hình cuối calibrated tốt
hơn trên tập độc lập. Tuy vậy, **28/144 lỗi có confidence ≥ 0,90**: không được
dùng confidence cao làm bằng chứng chắc chắn đúng.

### 9.4. Selective prediction và ngưỡng "không xác định"

Ngưỡng τ được chọn trên **OOF X2 seed 42** (30.899 ảnh, mỗi ảnh dự đoán ngoài fold),
không dùng lockbox/test:

| Chỉ số | Giá trị |
|---|---:|
| τ (phân vị 5% của max-prob) | **0,9590** |
| Coverage (tỷ lệ ảnh được chấp nhận) | 95,00% |
| Accuracy trên phần chấp nhận | **99,5503%** |
| Accuracy toàn bộ | 97,8090% |
| Tỷ lệ lỗi bị từ chối | **80,50%** |

Diễn giải: nếu chỉ trả lời khi confidence ≥ 0,9590, mô hình chấp nhận 95% ảnh và
đạt 99,55% accuracy trên phần đó; 80,5% lỗi bị lọc. Đây là cơ chế "không xác
định" hợp lý, nhưng **không phải bộ phát hiện OOD**: ảnh không có biển vẫn có
thể có confidence cao.

### 9.5. Precision–Recall

![Hình 9.3. Precision–Recall và AP từng lớp](outputs/figures/v3/13_precision_recall.png)

*Hình 9.3. Micro-AP 0,997; 5 lớp AP thấp nhất; 35/43 lớp đạt AP 1,000.*

- Micro-AP: **0,997**.
- AP thấp nhất: lớp 8 (0,973), lớp 16 (0,986), lớp 7 (0,991).
- 35/43 lớp đạt AP 1,000.
- Lưu ý: AP khác F1 vì AP tích hợp trên mọi ngưỡng; lớp 8 có AP thấp dù F1 0,986
  vì có một số ảnh confidence thấp.

---

## Chương 10. Hiệu quả vận hành và triển khai

### 10.1. Latency và throughput

![Hình 10.1. Latency và throughput CPU/GPU theo batch](outputs/figures/v3/14_inference_perf.png)

*Hình 10.1. Model-only FP32; ngôi sao là end-to-end 1 ảnh; batch 1 GPU không nhanh hơn CPU do overhead.*

**Bảng 10.1. Benchmark suy luận D2 (FP32, máy train)**

| Thiết bị | Batch | Latency median (ms) | P95 (ms) | Throughput (ảnh/s) | VRAM allocated (MB) |
|---|---:|---:|---:|---:|---:|
| CPU (8 thread) | 1 | 0,775 | 1,120 | 1.290 | — |
| CPU (8 thread) | 16 | 7,111 | 8,782 | 2.250 | — |
| CPU (8 thread) | 32 | 15,659 | 20,440 | 2.044 | — |
| CPU (8 thread) | 64 | 40,163 | 42,871 | 1.593 | — |
| GPU RTX 5070 | 1 | 0,863 | 1,392 | 1.159 | 16,0 |
| GPU RTX 5070 | 16 | 0,854 | 1,435 | 18.727 | 31,4 |
| GPU RTX 5070 | 32 | 0,854 | 1,204 | 37.491 | 48,2 |
| GPU RTX 5070 | 64 | 0,810 | 1,167 | **79.019** | 81,7 |

**Bảng 10.2. End-to-end 1 ảnh (uint8 → normalize → forward → softmax → top-3)**

| Thiết bị | Median (ms) | P95 (ms) | Ghi chú |
|---|---:|---:|---|
| CPU (8 thread) | 1,330 | 1,577 | Chưa gồm upload/render UI |
| GPU RTX 5070 | 1,243 | 2,274 | Chưa gồm upload/render UI |

**Phát hiện quan trọng:** GPU batch 1 **không nhanh hơn CPU** (0,86 so với 0,78 ms)
vì overhead launch/sync. GPU chỉ có lợi khi batch lớn. Vì vậy không được nói
"GPU nhanh hơn 100 lần" dựa trên throughput batch 64.

### 10.2. Kích thước mô hình và bộ nhớ

| Chỉ số | Giá trị |
|---|---:|
| Số tham số D2 | 1.448.779 |
| Kích thước checkpoint | 5.815.209 byte ≈ **5,55 MB** |
| Dung lượng FP32 lý thuyết | 5,53 MB |
| Peak VRAM batch 64 | 81,7 MB allocated / 126 MB reserved |
| Peak RAM process | ~1.060 MB |

### 10.3. Thời gian đánh giá

| Thiết bị | n | Batch | Thời gian | Ảnh/s |
|---|---:|---:|---:|---:|
| CPU | 6.180 | 256 | 4,93 s | 1.252 |
| GPU | 6.180 | 256 | 1,48 s | 4.179 |

Phạm vi: normalize uint8 → forward; không gồm đọc file PPM. Thời gian này **không
phải** thời gian phản hồi UI.

### 10.4. Chi phí huấn luyện

Đã trình bày ở Bảng 4.2: 111 run, 3.411 epoch, 239,5 phút epoch-loop, cộng 8,2
phút baseline = **4,13 giờ**. Không gồm nạp cache, dự đoán val cuối, phân tích,
hình, báo cáo. Đây là con số **thấp hơn** thời gian thực tế của toàn dự án.

### 10.5. Trạng thái triển khai

| Tính năng | Trạng thái | Bằng chứng | Khoảng trống |
|---|---|---|---|
| Giao diện web (Streamlit) | đã có | app.py; outputs/figures/11_ui_*.png | không có |
| Tải 1 ảnh lên và dự đoán top-3 | đã có | e2e_ui.json 9/9 check pass | chỉ 1 ảnh mỗi lần |
| Hiển thị tên tiếng Việt + confidence | đã có | app.py render_prediction; e2e_ui.json | không có |
| Cảnh báo 'không xác định' theo ngưỡng τ | đã có (ngưỡng cũ) | outputs/results/threshold.json τ=0.9462 chọn trên val split_random | chưa dùng τ v2 = 0.9590 chọn trên OOF; cần đổi khi cập nhật demo |
| Xử lý file lỗi / quá 20 MB | đã có | e2e_ui.json check loi_file_hong, loi_qua_20mb | không có |
| Chọn ảnh mẫu | đã có | e2e_ui.json check chon_anh_mau | không có |
| Nhiều ảnh / batch trên giao diện | chưa có | app.py dùng st.file_uploader đơn | cần phát triển nếu muốn demo batch |
| API cho hệ thống khác | chưa có | không có endpoint trong repo | ngoài phạm vi bài tập |
| Lưu lịch sử dự đoán | chưa có | không có DB/session state lưu lịch sử | ngoài phạm vi bài tập |
| Checkpoint v2 (D2/X2) | chưa tích hợp vào demo | app.py mặc định outputs/runs/final_s42 hoặc checkpoints/final_s42 (legacy) | demo đang dùng checkpoint cũ; báo cáo phải nói rõ |
| Golden test 20 ảnh giữa 2 môi trường | đã có | golden_compare.json max_prob_diff=0.0, pass=True | chạy trên checkpoint legacy |
| Kiểm thử trên máy cấu hình thấp | chưa có | chỉ đo CPU trên máy train (i9-14900KF) | cần máy khác hoặc giới hạn thread/ RAM |
| Kiểm thử camera thật / video | chưa có | external_test chỉ có ảnh tĩnh đã ẩn danh | ngoài phạm vi và cần thu thập dữ liệu |
| Tuyên bố an toàn cho giao thông thực tế | không | footer app.py và báo cáo đều ghi cảnh báo | không được tuyên bố |

### 10.6. Demo hiện tại dùng checkpoint nào?

- `app.py` tìm checkpoint theo thứ tự: biến môi trường `GTSRB_RUN_DIR` →
  `outputs/runs/final_s42/best.pt` → `checkpoints/final_s42/best.pt`.
- Cả hai đều là **checkpoint legacy** (E3/CNN-B, 2.201.931 params), không phải D2.
- Ngưỡng đang dùng là `outputs/results/threshold.json` với τ = 0,9462, chọn trên
  val `split_random` cũ.
- Đã chuẩn bị sẵn τ v2 = 0,9590 trong `outputs/results/v3/tables/threshold_v2_oof.json`
  và checkpoint `outputs/runs/v2_final_s42/best.pt`, nhưng **chưa tích hợp**.
- Vì vậy báo cáo tách rõ: **mô hình D2 đã kiểm chứng**, **demo chưa cập nhật**.

---

## Chương 11. Độ tin cậy và tái lập

### 11.1. Seed, cấu hình, hash

- Seed 42/123/2024; cùng split; cùng early stopping.
- Mỗi run ghi `config_sha256` (hash YAML) và `split_sha256` (hash split).
- Checkpoint `best.pt` là dict: `state_dict`, `config`, `class_names`, `epoch`,
  `val_macro_f1`, `git_commit`, `torch_version`, `config_sha256`, `split_sha256`.
- Final còn khớp `FREEZE.json`.

### 11.2. Chạy lại cùng seed

`scripts/repro_check_v2.py` chạy 2 lần 3 epoch cùng seed:

| Chỉ số | Sai khác tuyệt đối |
|---|---:|
| train_loss | 0,0 |
| val_loss | 0,0 |
| val_acc | 0,0 |
| val_macro_f1 | 0,0 |

Kết luận: **tái lập tuyệt đối** trong cùng môi trường. Không đảm bảo tái lập
tuyệt đối trên GPU/phiên bản thư viện khác.

### 11.3. Kiểm thử tự động

- `pytest`: **22 passed** (`tests/test_split_v2.py`, `test_stats_v2.py`,
  `test_config_v2.py`).
- `verify_41.json`: macro-F1 OOF tính lại lệch **0,0**; summary F15 lệch **3,1e−07**.
- `golden_compare.json`: 20 ảnh, sai khác xác suất **0,0**, pass.

### 11.4. Nghiệm thu 12/12

**Bảng 11.1. Mười hai điều kiện nghiệm thu**

| # | Điều kiện | Đạt | Bằng chứng |
|---:|---|---|---|
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

### 11.5. Audit dữ liệu

- 15/15 kiểm tra PASS (C1–C12 + HASH + MEANSTD + COUNT).
- 0 track/sign-group overlap giữa lockbox và dev.
- 0 ảnh test nằm trong train/lockbox.
- Mean/std khớp lại từ cache: 0,0e+00.

---

## Chương 12. Hạn chế

### 12.1. Dữ liệu

1. Chỉ dùng GTSRB; không có ảnh camera thật, thời tiết thật, biển Việt Nam.
2. Official test **không track-disjoint**; 751 ảnh gần trùng train.
3. 3.596 ảnh tối, 3.920 ảnh dưới p10 độ nét; chưa có kiểm tra nhãn thủ công.
4. 144 ảnh lỗi chỉ thuộc 23 track; không đếm như 144 ca độc lập.

### 12.2. Mô hình

1. Không dùng pretrained; dung lượng 1,45M params.
2. Occlusion làm accuracy rơi còn 17,63%.
3. Lớp 19 và 29 có F1 0,760 và 0,776; khó phân biệt với lớp cùng hình dạng.
4. Depth có CI ủng hộ nhưng p Holm 0,373; chưa đủ kết luận chắc chắn.
5. Augmentation "all" không được chứng minh là có lợi.

### 12.3. Đánh giá

1. Lockbox 8.302 ảnh / 277 track nhỏ hơn official test 12.630 ảnh.
2. Không có CI track-bootstrap cho lockbox (one-shot, không có per-image predictions).
3. Screening val 6.180 ảnh chỉ để chọn, không dùng kết luận.
4. Chưa có kiểm tra thủ công 41 ảnh lỗi.

### 12.4. Triển khai

1. Demo dùng checkpoint legacy và ngưỡng cũ.
2. Chưa đo trên máy cấu hình thấp.
3. Chưa có API, batch upload, lịch sử dự đoán.
4. Không tuyên bố mô hình đủ an toàn cho giao thông thực tế.

### 12.5. Bằng chứng chưa đủ

- Weight decay, width, class weight, augmentation: CI chứa 0.
- Depth: CI không chứa 0 nhưng Holm chưa đủ mạnh.
- Nguyên nhân lỗi: mới là heuristic, chưa kiểm tra thủ công.
- GPU/CPU utilization trong 111 run: không đo.

---

## Chương 13. Hướng phát triển

### Giai đoạn 1 — Cải thiện mô hình

- Tăng dữ liệu cho lớp 19, 29, 30, 28, 23.
- Thử augmentation có chọn lọc (color, perspective) thay vì "all".
- Xử lý occlusion: random erasing, cutout, hoặc dữ liệu che thật.
- Thử thêm kiến trúc (ResNet nhỏ, MobileNet tự huấn luyện).
- Kiểm tra nhiều split độc lập.

### Giai đoạn 2 — Tối ưu hiệu suất

- Pruning, knowledge distillation, quantization.
- Giảm dung lượng 5,55 MB xuống dưới 2 MB.
- Đo lại latency/throughput trên CPU và GPU sau tối ưu.
- Đo trên máy cấu hình thấp (laptop 4–8 GB RAM).

### Giai đoạn 3 — Hoàn thiện ứng dụng

- Cập nhật demo sang checkpoint D2 + τ = 0,9590.
- Thêm batch upload, API, lịch sử dự đoán.
- Tích hợp camera thật; thu thập ảnh thật đã ẩn danh.
- Kiểm thử với ảnh ngoài phân phối; bổ sung cảnh báo OOD.

---

## Chương 14. Kết luận

**1. Đã hoàn thành mục tiêu gì?**

Đã xây dựng pipeline hoàn chỉnh: dữ liệu → baseline → CNN tự xây dựng →
factorial → kiến trúc → hyperparameter → augmentation → OOF → lockbox →
official test → phân tích lỗi → robustness → calibration → Grad-CAM → đo vận hành.

**2. Mô hình đạt kết quả như thế nào?**

D2 đạt **OOF macro-F1 97,0400%**, **lockbox macro-F1 97,5981%**, **official test
clean 98,2499%**. Accuracy tương ứng 97,9244% / 98,8758% / 98,8692%.

**3. Bằng chứng nào chứng minh kết quả đáng tin cậy?**

Split 3 tầng theo sign-group, audit 15/15 PASS, OOF 3 fold × 3 seed, lockbox
one-shot, CI cluster bootstrap theo track, Holm cho 8 so sánh chính, tái lập
tuyệt đối cùng seed, 22 test pass, nghiệm thu 12/12.

**4. Những vấn đề nào chưa giải quyết?**

Occlusion (17,63%), lớp 19/29 yếu, chưa kiểm tra nhãn thủ công, chưa có camera
thật/máy yếu, demo chưa cập nhật D2, depth/width/weight decay/augmentation chưa
đủ bằng chứng.

**5. Công việc tiếp theo là gì?**

Cập nhật demo, kiểm tra thủ công 41 ảnh, thử augmentation chọn lọc, xử lý
occlusion, đo trên máy yếu, thu thập ảnh thật.

---

## Chương 15. Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. (2012). *Man vs. computer:
   Benchmarking machine learning algorithms for traffic sign recognition.*
   Neural Networks, 32, 323–332.
2. scikit-learn: `f1_score`, `classification_report`, `LinearSVC`.
3. Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
4. Holm, S. (1979). *A simple sequentially rejective multiple test procedure.*
5. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). *On Calibration of Modern
   Neural Networks.*
6. Hendrycks, D., Dietterich, T. (2019). *Benchmarking Neural Network Robustness
   to Common Corruptions and Perturbations.*
7. Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks
   via Gradient-based Localization.*
8. Bouthillier, X. et al. (2021). *Accounting for Variance in Machine Learning
   Benchmarks.*

---

## PHỤ LỤC

## Phụ lục A. Bảng đầy đủ 43 lớp trên lockbox

![Hình A.1. Ảnh mẫu 43 lớp](outputs/figures/v3/17_class_samples.png)

*Hình A.1. Mỗi lớp một ảnh mẫu, chọn ảnh có độ nét ≥ trung vị lớp và độ sáng gần trung vị; ảnh hiển thị là crop gốc, mô hình nhận 64×64.*

| ID | Tên biển | Support | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Giới hạn tốc độ 20 km/h | 30 | 0,9892 | 1,0000 | 0,9945 |
| 1 | Giới hạn tốc độ 30 km/h | 450 | 0,9948 | 0,9993 | 0,9970 |
| 2 | Giới hạn tốc độ 50 km/h | 450 | 0,9934 | 0,9956 | 0,9945 |
| 3 | Giới hạn tốc độ 60 km/h | 240 | 0,9986 | 0,9958 | 0,9972 |
| 4 | Giới hạn tốc độ 70 km/h | 750 | 0,9982 | 0,9978 | 0,9980 |
| 5 | Giới hạn tốc độ 80 km/h | 300 | 0,9617 | 1,0000 | 0,9804 |
| 6 | Hết giới hạn tốc độ 80 km/h | 90 | 1,0000 | 1,0000 | 1,0000 |
| 7 | Giới hạn tốc độ 100 km/h | 240 | 0,9850 | 1,0000 | 0,9924 |
| 8 | Giới hạn tốc độ 120 km/h | 660 | 0,9974 | 0,9747 | 0,9859 |
| 9 | Cấm vượt | 300 | 1,0000 | 1,0000 | 1,0000 |
| 10 | Cấm vượt cho xe trên 3,5 tấn | 240 | 0,9880 | 1,0000 | 0,9939 |
| 11 | Đường ưu tiên tại giao lộ tới | 270 | 0,9759 | 0,9988 | 0,9872 |
| 12 | Đường ưu tiên | 390 | 1,0000 | 0,9983 | 0,9991 |
| 13 | Nhường đường | 510 | 1,0000 | 0,9922 | 0,9961 |
| 14 | Dừng lại | 142 | 1,0000 | 1,0000 | 1,0000 |
| 15 | Cấm xe cộ | 120 | 1,0000 | 1,0000 | 1,0000 |
| 16 | Cấm xe trên 3,5 tấn | 90 | 1,0000 | 1,0000 | 1,0000 |
| 17 | Cấm đi vào | 210 | 1,0000 | 1,0000 | 1,0000 |
| 18 | Nguy hiểm chung | 240 | 1,0000 | 1,0000 | 1,0000 |
| 19 | Đường cong nguy hiểm bên trái | 30 | 0,9861 | 0,6556 | 0,7601 |
| 20 | Đường cong nguy hiểm bên phải | 60 | 0,9715 | 0,9222 | 0,9454 |
| 21 | Đường cong liên tiếp | 60 | 1,0000 | 0,8944 | 0,9438 |
| 22 | Đường gồ ghề | 90 | 1,0000 | 1,0000 | 1,0000 |
| 23 | Đường trơn | 120 | 0,9005 | 0,9833 | 0,9381 |
| 24 | Đường hẹp bên phải | 60 | 0,9891 | 1,0000 | 0,9945 |
| 25 | Công trường thi công | 330 | 0,9678 | 1,0000 | 0,9836 |
| 26 | Đèn tín hiệu giao thông | 90 | 0,9963 | 1,0000 | 0,9982 |
| 27 | Người đi bộ | 60 | 1,0000 | 1,0000 | 1,0000 |
| 28 | Trẻ em qua đường | 120 | 0,8850 | 0,9750 | 0,9276 |
| 29 | Xe đạp qua đường | 60 | 0,9917 | 0,6389 | 0,7762 |
| 30 | Cảnh báo băng tuyết | 90 | 0,9605 | 0,8926 | 0,9252 |
| 31 | Động vật hoang dã qua đường | 150 | 0,9826 | 1,0000 | 0,9912 |
| 32 | Hết mọi giới hạn tốc độ và cấm vượt | 60 | 0,9944 | 0,9056 | 0,9448 |
| 33 | Rẽ phải phía trước | 150 | 1,0000 | 1,0000 | 1,0000 |
| 34 | Rẽ trái phía trước | 90 | 1,0000 | 1,0000 | 1,0000 |
| 35 | Chỉ đi thẳng | 240 | 0,9918 | 1,0000 | 0,9959 |
| 36 | Đi thẳng hoặc rẽ phải | 90 | 1,0000 | 0,9667 | 0,9830 |
| 37 | Đi thẳng hoặc rẽ trái | 30 | 1,0000 | 1,0000 | 1,0000 |
| 38 | Đi bên phải | 360 | 1,0000 | 1,0000 | 1,0000 |
| 39 | Đi bên trái | 60 | 1,0000 | 1,0000 | 1,0000 |
| 40 | Vòng xuyến bắt buộc | 60 | 0,9787 | 1,0000 | 0,9891 |
| 41 | Hết cấm vượt | 60 | 0,9206 | 0,9944 | 0,9542 |
| 42 | Hết cấm vượt cho xe trên 3,5 tấn | 60 | 1,0000 | 1,0000 | 1,0000 |

Nguồn: `outputs/results/v3/tables/07_per_class_lockbox.csv` (trung bình 3 seed).

## Phụ lục B. Chi tiết các fold OOF

| Fold | Train | Val | Track train | Track val | Lớp val | Overlap track | Overlap group |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19.949 | 10.950 | 665 | 365 | 43 | 0 | 0 |
| 1 | 20.729 | 10.170 | 691 | 339 | 43 | 0 | 0 |
| 2 | 21.120 | 9.779 | 704 | 326 | 43 | 0 | 0 |

## Phụ lục C. Tổng hợp 31 cấu hình

| Config | Block | Accuracy (%) | Macro-F1 (%) | Params |
|---|---|---:|---:|---:|
| b1 | Baseline | 87,265 | 85,997 | 75.895 |
| f00 | F_factorial | 90,092 | 84,109 | 2.201.707 |
| f01 | F_factorial | 93,263 | 87,846 | 2.201.931 |
| f02 | F_factorial | 94,121 | 91,613 | 2.201.707 |
| f03 | F_factorial | 96,969 | 96,047 | 2.201.931 |
| f04 | F_factorial | 90,248 | 84,173 | 2.201.707 |
| f05 | F_factorial | 92,643 | 87,470 | 2.201.931 |
| f06 | F_factorial | 94,094 | 91,958 | 2.201.707 |
| f07 | F_factorial | 96,980 | 96,501 | 2.201.931 |
| f08 | F_factorial | 92,039 | 86,672 | 2.201.707 |
| f09 | F_factorial | 94,461 | 90,006 | 2.201.931 |
| f10 | F_factorial | 95,086 | 91,097 | 2.201.707 |
| f11 | F_factorial | 96,688 | 94,209 | 2.201.931 |
| f12 | F_factorial | 92,864 | 87,003 | 2.201.707 |
| f13 | F_factorial | 94,757 | 90,700 | 2.201.931 |
| f14 | F_factorial | 94,730 | 90,310 | 2.201.707 |
| f15 | F_factorial | 96,564 | 93,946 | 2.201.931 |
| d1 | D_architecture | 74,234 | 59,188 | 99.019 |
| d2 | D_architecture | 98,004 | 97,158 | 1.448.779 |
| d3 | D_architecture | 96,575 | 93,105 | 399.947 |
| w1 | D_architecture | 94,876 | 90,686 | 1.083.579 |
| w2 | D_architecture | 97,104 | 95,215 | 3.366.363 |
| l1 | H_hyperparam | 91,888 | 86,807 | 2.201.931 |
| cw1 | H_hyperparam | 95,960 | 92,943 | 2.201.931 |
| i2 | H_hyperparam | 94,213 | 90,499 | 4.823.371 |
| g1 | G_augmentation | 95,572 | 92,251 | 2.201.931 |
| g2 | G_augmentation | 97,767 | 97,417 | 2.201.931 |
| g3 | G_augmentation | 96,375 | 95,264 | 2.201.931 |
| x1 | X_confirmation | 97,194 | 96,297 | 2.201.931 |
| x2 | X_confirmation | 97,924 | 97,040 | 1.448.779 |
| x3 | X_confirmation | 96,111 | 95,059 | 2.201.931 |

Bảng đầy đủ 111 run: `outputs/results/v3/tables/20_experiment_index.csv`.

## Phụ lục D. Toàn bộ cặp nhầm trong top 19

| Thật | Dự đoán | Số ảnh (TB) | % lớp thật | Cùng hình dạng | Cùng màu |
|---:|---:|---:|---:|---|---|
| 32 | 41 | 14,0 | 23,3% | Có | Có |
| 29 | 28 | 13,7 | 22,8% | Có | Có |
| 19 | 23 | 13,5 | 45,0% | Có | Có |
| 8 | 5 | 11,5 | 1,7% | Có | Có |
| 13 | 10 | 9,0 | 1,8% | Không | Có |
| 29 | 25 | 6,3 | 10,6% | Có | Có |
| 30 | 11 | 5,7 | 6,3% | Có | Có |
| 20 | 25 | 5,0 | 8,3% | Có | Có |
| 8 | 2 | 5,0 | 0,8% | Có | Có |
| 21 | 23 | 4,5 | 7,5% | Có | Có |
| 8 | 7 | 4,5 | 0,7% | Có | Có |
| 36 | 35 | 4,0 | 4,4% | Có | Có |
| 19 | 31 | 4,0 | 13,3% | Có | Có |
| 4 | 8 | 3,0 | 0,4% | Có | Có |
| 23 | 30 | 3,0 | 2,5% | Có | Có |
| 30 | 23 | 2,0 | 2,2% | Có | Có |
| 30 | 5 | 2,0 | 2,2% | Không | Có |
| 20 | 30 | 2,0 | 3,3% | Có | Có |
| 28 | 30 | 2,0 | 1,7% | Có | Có |

## Phụ lục E. Thông số kiến trúc 8 cấu hình

| Config | Kiến trúc | Params | BN | Dropout | GAP | FC hidden |
|---|---|---:|---|---:|---|---:|
| f00 | 3 khối 32/64/128 | 2.201.707 | Không | 0,0 | Không | 256 |
| f07 | 3 khối 32/64/128 | 2.201.931 | Có | 0,3 | Không | 256 |
| f15 | 3 khối 32/64/128 | 2.201.931 | Có | 0,3 | Không | 256 |
| d1 | 3 khối 32/64/128 | 99.019 | Có | 0,3 | Có | 256 |
| d2 | 4 khối 32/64/128/256 | 1.448.779 | Có | 0,3 | Không | 256 |
| d3 | 4 khối 32/64/128/256 | 399.947 | Có | 0,4 | Có | 256 |
| w1 | 3 khối 16/32/64 | 1.083.579 | Có | 0,3 | Không | 256 |
| w2 | 3 khối 48/96/192 | 3.366.363 | Có | 0,3 | Không | 256 |

## Phụ lục F. Phần cứng và phần mềm

Xem Bảng 3.3 và Bảng 3.4. Nguồn máy đọc:
`outputs/results/v3/tables/14_hardware_software.json`.

## Phụ lục G. Calibration và robustness chi tiết

- Calibration: Bảng 9.2; nguồn `12_calibration_confidence.csv`.
- Robustness: Bảng 5.7; nguồn `25_corruption_summary.csv`.
- Selective risk: Hình 9.2; nguồn `threshold_v2_oof.json`.

## Phụ lục H. Phiếu rà soát lỗi

- Toàn bộ 144 lỗi: `17_manual_error_review.csv`.
- Bản rút gọn 41 ảnh / 23 track: `17b_manual_error_review_diverse.csv`.
- Contact sheet: `outputs/figures/v3/contact_sheets/error_sheet_01.png`,
  `error_sheet_02.png`.
- Trạng thái: **cột verdict/cause/note còn trống**, chưa có kiểm tra thủ công.

## Phụ lục I. Đính chính số liệu so với báo cáo v2

| Mục | Báo cáo cũ | Số đúng | Bằng chứng |
|---|---|---|---|
| OOF accuracy | 98,004% | 97,924% | 06c_oof_accuracy_check.json |
| Lockbox CI của macro-F1 | [95,0511; 97,9422] | mean 97,5981; sd 0,6246; range 3 seed [96,8770; 97,9706] | 06b_lockbox_uncertainty.csv |
| Số lỗi screening val | 190 lỗi / 6.180 ảnh | F15: 190 lỗi; D2: 144 lỗi / 23 track | 17_manual_error_review_summary.json |
| Latency | 0,93 ms/ảnh (không nêu batch/thiết bị) | GPU batch 1: 0,86 ms model-only; 1,24 ms end-to-end Python | 13_inference_benchmark.csv |
| Ngưỡng unknown của demo | τ = 0,9462 trên val split_random | τ v2 = 0,9590 trên OOF X2 | threshold_v2_oof.json; outputs/results/threshold.json |
| Checkpoint demo | final_s42 (legacy) | v2_final_s42 / X2 (D2, 1.448.779 params) | 15_deployment_status.csv |
| Thời gian huấn luyện | 4,13 giờ | 3,99 giờ epoch-loop (111 run) + 8,2 phút baseline | training_cost_totals.json |
| Depth: CI và Holm | +3,212 [+0,036; +4,440], p Holm 0,373 | giữ nguyên; báo 'có bằng chứng vừa, chưa đủ sau Holm' | main_comparisons.csv |

## Phụ lục J. Lệnh chạy lại

```powershell
# Dữ liệu và split
python -m src.data --download --report
python -m src.split_v2 --build
python -m src.audit_v2

# Baseline và factorial
python -m src.baseline_v2
python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024

# Kiến trúc, hyperparam, augmentation
python scripts/run_matrix.py --configs d1,d2,d3,w1,w2 --seeds 42,123,2024
python scripts/run_matrix.py --configs l1,cw1,i2 --seeds 42,123,2024
python scripts/run_matrix.py --configs g1,g2,g3 --seeds 42,123,2024

# OOF, freeze, final
python -m src.summarize_v2
python scripts/select_candidates.py
python scripts/run_matrix.py --configs x1,x2,x3 --seeds 42,123,2024 --folds 0,1,2
python scripts/run_oof_analysis.py --configs x1,x2,x3
python scripts/freeze_v2.py --winner x2 --source d2 --reason "..."
python scripts/run_matrix.py --configs final --seeds 42,123,2024

# Đánh giá
python -m src.final_eval_v2 --mode lockbox --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
python -m src.final_eval_v2 --mode test --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024

# Report pack v3
python scripts/report_v3_data.py --only all
python scripts/report_v3_figures.py --only all
python scripts/report_v3_meta.py
python scripts/report_v3_manifest.py
```

## Phụ lục K. Glossary

| Thuật ngữ | Định nghĩa | Công thức / ví dụ |
|---|---|---|
| Accuracy | Tỷ lệ ảnh dự đoán đúng trên tổng số ảnh. | số đúng / tổng; dễ bị lớp nhiều ảnh chi phối |
| Precision | Trong các ảnh được dự đoán là lớp X, bao nhiêu ảnh thực sự là X. | TP / (TP + FP) |
| Recall | Trong các ảnh thực sự là lớp X, mô hình nhận ra bao nhiêu. | TP / (TP + FN) |
| F1-score | Trung bình điều hòa của precision và recall. | 2PR / (P + R) |
| Macro-F1/43 | Trung bình F1 của 43 lớp, mỗi lớp trọng số 1/43. | lớp hiếm quan trọng ngang lớp nhiều ảnh |
| Weighted-F1 | F1 trung bình có trọng số theo số ảnh từng lớp. | giống accuracy ở chỗ lớp nhiều ảnh chi phối hơn |
| Top-3 accuracy | Tỷ lệ ảnh mà nhãn đúng nằm trong 3 dự đoán cao nhất. | dùng khi cần gợi ý thay vì chốt 1 nhãn |
| Track | Một đoạn video ngắn (~30 frame) của cùng một biển. | các frame gần như trùng nhau; không được chia khác tập |
| Sign-group | Nhóm các track được gộp vì ảnh gần trùng (MAD < 5). | đơn vị chia split để tránh rò rỉ |
| OOF (out-of-fold) | Mỗi ảnh dev được dự đoán bởi mô hình không train trên fold chứa ảnh đó. | giống thi thử mà đề không có trong bài ôn |
| Lockbox | Tập giữ riêng, chỉ mở một lần sau khi chốt mô hình. | đề thi cuối kỳ chưa từng xem |
| Official test | Tập test công bố của GTSRB, không có track ID. | benchmark phụ, không track-disjoint |
| CI 95% | Khoảng bất định của một ước lượng. | nếu lặp lại thí nghiệm nhiều lần, ~95% khoảng chứa giá trị thật |
| Holm–Bonferroni | Hiệu chỉnh p-value khi kiểm định nhiều giả thuyết. | giảm xác suất 'may mắn' khi so nhiều cặp |
| ECE | Sai lệch trung bình giữa confidence và accuracy theo bin. | 0 = calibrated hoàn hảo |
| Brier score | Sai số bình phương của xác suất dự đoán so với one-hot. | càng nhỏ càng tốt |
| Grad-CAM | Bản đồ nhiệt cho biết vùng ảnh ảnh hưởng tới dự đoán. | mô hình 'nhìn' vào đâu |
| Corruption | Biến đổi ảnh để kiểm tra độ bền: blur, noise, brightness, contrast, JPEG, occlusion. | điều kiện xấu ngoài đời |
| Occlusion | Che một phần trung tâm ảnh. | biển bị vật cản che |
| Throughput | Số ảnh xử lý được mỗi giây. | batch / latency |

## Phụ lục L. Chỉ mục artifact

| Nội dung | File |
|---|---|
| Bảng 43 lớp | `outputs/results/v3/tables/01_class_catalog.csv` |
| Split + hash + mean/std | `outputs/splits/v2/manifest.json` |
| Audit C1–C12 | `outputs/results/v2/audit/audit_report.json` |
| Baseline | `outputs/results/v2/baseline.json` |
| Factorial | `outputs/results/v3/tables/05_model_comparison.csv` |
| Kiến trúc | `outputs/results/v3/tables/23_architecture_spec.csv` |
| OOF | `outputs/results/v3/tables/11_oof_by_fold_seed.csv` |
| Lockbox | `outputs/results/v3/tables/21_lockbox_by_seed.csv` |
| Official test | `outputs/results/v3/tables/22_test_per_class_raw_clean.csv` |
| Per-class | `outputs/results/v3/tables/07_per_class_lockbox.csv` |
| Confusion | `outputs/results/v3/tables/10_confusion_lockbox_*.csv` |
| Calibration | `outputs/results/v3/tables/12_calibration_confidence.csv` |
| Grad-CAM | `outputs/results/v3/tables/16_gradcam_roi_mass.csv` |
| Benchmark | `outputs/results/v3/tables/13_inference_benchmark.csv` |
| Hardware | `outputs/results/v3/tables/14_hardware_software.json` |
| Deployment | `outputs/results/v3/tables/15_deployment_status.csv` |
| Nghiệm thu | `outputs/results/v2/ACCEPTANCE.json` |
| Hình | `outputs/figures/v3/` |
| Manifest pack | `outputs/results/v3/manifest.json` |
