# KẾ HOẠCH TỔNG QUAN — BÀI TẬP LỚN HỌC SÂU
## Phân loại biển báo giao thông Đức (GTSRB) bằng CNN tự xây dựng

| Thông tin | Giá trị |
|---|---|
| Môn học | Học sâu — 12INF028 |
| Đề tài | Phân loại biển báo giao thông (GTSRB) |
| Phiên bản tài liệu | 1.0 — bản chủ (master) |
| Ngày lập | 22/10/2026 |
| Trạng thái | Đã khóa phạm vi, sẵn sàng triển khai |
| Thay thế | `Cac_buoc_phan_loai_bien_bao_GTSRB.docx` và `Ke_hoach_hoan_thien_bai_tap_lon_GTSRB_CNN.docx` (hai file cũ trở thành tài liệu tham khảo; mọi điểm chưa thống nhất được chốt trong file này) |
| Hình thức thực hiện | **1 người thực hiện toàn bộ trên 1 máy**, sau đó nhóm trao đổi/học lại |
| Ngôn ngữ báo cáo | Tiếng Việt |

> **Nguyên tắc xuyên suốt:** mọi con số trong báo cáo và slide phải sinh ra từ lần chạy thật, lấy từ `outputs/results/summary.csv`. Không đưa số liệu dự kiến, số liệu "minh họa" hoặc kết quả chưa chạy vào báo cáo dưới bất kỳ hình thức nào.

---

## Mục lục

1. [Quyết định đã khóa](#1-quyết-định-đã-khóa)
2. [Phát biểu bài toán](#2-phát-biểu-bài-toán)
3. [Đối chiếu yêu cầu môn học](#3-đối-chiếu-yêu-cầu-môn-học)
4. [Dữ liệu GTSRB](#4-dữ-liệu-gtsrb)
5. [Chia dữ liệu và chống rò rỉ](#5-chia-dữ-liệu-và-chống-rò-rỉ)
6. [Tiền xử lý và augmentation](#6-tiền-xử-lý-và-augmentation)
7. [Baseline](#7-baseline)
8. [Kiến trúc CNN](#8-kiến-trúc-cnn)
9. [Thiết lập huấn luyện](#9-thiết-lập-huấn-luyện)
10. [Ma trận thực nghiệm](#10-ma-trận-thực-nghiệm)
11. [Logging, checkpoint và quy ước artifact](#11-logging-checkpoint-và-quy-ước-artifact)
12. [Đánh giá và chỉ số](#12-đánh-giá-và-chỉ-số)
13. [Phân tích lỗi](#13-phân-tích-lỗi)
14. [Demo dữ liệu mới](#14-demo-dữ-liệu-mới)
15. [Tái lập kết quả](#15-tái-lập-kết-quả)
16. [Kế hoạch 15 ngày (solo)](#16-kế-hoạch-15-ngày-solo)
17. [Rủi ro và đối phó](#17-rủi-ro-và-đối-phó)
18. [Cấu trúc báo cáo](#18-cấu-trúc-báo-cáo)
19. [Cấu trúc slide](#19-cấu-trúc-slide)
20. [Câu hỏi phản biện và trả lời gợi ý](#20-câu-hỏi-phản-biện-và-trả-lời-gợi-ý)
21. [Checklist nghiệm thu](#21-checklist-nghiệm-thu)
22. [Phụ lục](#22-phụ-lục)

---

## 1. Quyết định đã khóa

Các quyết định dưới đây **không thay đổi** trong suốt quá trình làm, trừ khi có lý do kỹ thuật được ghi lại rõ ràng:

| # | Hạng mục | Quyết định | Lý do |
|---|---|---|---|
| D1 | Phạm vi lớp | **43 lớp đầy đủ** (index 0–42) | GPU 12 GB thừa sức; giữ được phân tích mất cân bằng, confusion matrix và cặp nhầm có ý nghĩa |
| D2 | Kích thước ảnh | **64×64 RGB** | Cân bằng chi tiết/chi phí; 48×48 và 96×96 để dành cho ablation E6 |
| D3 | Môi trường | **Local: i9-14900KF, 64 GB RAM, RTX 5070 12 GB, Windows** | Không giới hạn giờ, kiểm soát phiên bản, tái lập dễ |
| D4 | Tổ chức code | **`src/` package + CLI + notebook cho EDA/hình** | Demo và train dùng chung hàm `preprocess`, tránh lệch pipeline |
| D5 | Nhân sự | **1 người làm toàn bộ** trên máy này | Ưu tiên P0, cắt P2, viết báo cáo song song từ ngày 2 |
| D6 | Tập test | **Split test chính thức 12.630 ảnh, đóng băng đến cuối** | Đúng yêu cầu "test độc lập"; tránh tune trên test |
| D7 | Chỉ số chính | **Accuracy + Macro-F1 + Confusion matrix** | Yêu cầu đề bài; macro-F1 phản ánh lớp hiếm |
| D8 | Tiêu chí chọn checkpoint | **Val Macro-F1** (early stopping patience 7) | Thống nhất một tiêu chí cho mọi run, tránh chọn theo cảm tính |
| D9 | Baseline | **Majority-class + HOG(64×64) + LinearSVC** | Yêu cầu "baseline đơn giản"; HOG+SVM là baseline chuẩn của GTSRB |
| D10 | Mô hình chính | **CNN-B: 3 khối conv + BN + Dropout + WD + augmentation nhẹ** | Tự xây dựng đúng yêu cầu, đủ mạnh, giải thích được từng thành phần |

### 1.1. Những điều không được làm

- Không dùng tập test để chọn mô hình, learning rate, số epoch hay ngưỡng quyết định.
- Không lật ngang/dọc ảnh biển báo (làm đổi nghĩa biển, ví dụ rẽ trái ↔ rẽ phải).
- Không commit `data/`, checkpoint, `.venv`, ảnh tự chụp chưa ẩn danh.
- Không đưa ảnh có mặt người/biển số xe rõ nét vào báo cáo, slide hay repo.
- Không tuyên bố mô hình đủ an toàn để dùng trong hệ thống giao thông thực tế.
- Không tái phân phối bộ dữ liệu gốc; chỉ trích dẫn nguồn.

---

## 2. Phát biểu bài toán

### 2.1. Bảng phát biểu

| Thành phần | Nội dung |
|---|---|
| **Input** | Ảnh màu RGB chứa **một** biển báo giao thông Đức; kích thước gốc thay đổi tùy ảnh, được resize về 64×64 |
| **Output** | Nhãn thuộc một trong **43 lớp** (0–42) + xác suất; demo hiển thị top-3 |
| **Số chiều đầu vào** | 64 × 64 × 3 = **12.288 biến** |
| **Số lớp** | 43 (phân loại đa lớp, single-label) |
| **Đối tượng sử dụng** | Sinh viên/nghiên cứu thị giác máy tính; người cần công cụ gán nhãn/kiểm kê biển báo |
| **Giá trị thực tế** | Tự động hóa gán nhãn biển báo từ kho ảnh lớn (tiết kiệm công sức), hỗ trợ kiểm kê/đánh giá chất lượng biển báo, tiền xử lý cho bản đồ số và bài toán trợ lái |
| **Giới hạn sử dụng** | Chỉ là nguyên mẫu học thuật; **không dùng cho quyết định giao thông thực tế**, không dùng cho xe tự hành |

### 2.2. Vì sao bài toán có ý nghĩa

- Biển báo là tín hiệu giao thông có cấu trúc (hình tròn/vuông/tam giác, màu đỏ/xanh/vàng) → bài toán thị giác máy tính kinh điển, phù hợp để học CNN.
- GTSRB có **mất cân bằng lớp mạnh** và **nhiều cặp lớp rất giống nhau** → buộc phải dùng macro-F1, per-class report và phân tích lỗi, đúng trọng tâm môn học.
- Kết quả có thể mở rộng sang bài toán thực tế: kiểm kê biển báo, phát hiện biển hư hỏng, gán nhãn dữ liệu bán tự động.

### 2.3. Ngoài phạm vi (ghi rõ trong báo cáo)

- Không giải quyết bài toán **phát hiện** biển báo trong ảnh cảnh rộng (detection) — chỉ phân loại ảnh đã chứa biển.
- Không xử lý ảnh có nhiều biển báo cùng lúc.
- Không nhận diện biển báo Việt Nam (khác hệ thống biển Đức) — chỉ dùng để minh họa domain shift.

---

## 3. Đối chiếu yêu cầu môn học

### 3.1. Yêu cầu chung (mục 2 của đề)

| # | Yêu cầu | Cách đáp ứng | Bằng chứng |
|---|---|---|---|
| 1 | Phát biểu input, output, đối tượng, giá trị thực tế | Mục 2 của tài liệu này; Chương 1 báo cáo | Ch1 |
| 2 | Nguồn dữ liệu, giấy phép, số mẫu, số lớp/biến, cách chia | Mục 4–5; `outputs/results/data_report.json` | Ch2 + phụ lục |
| 3 | ≥1 baseline đơn giản | Majority-class + HOG+SVM (E0a, E0b) | Bảng Ch4 |
| 4 | ≥1 mô hình học sâu chính, kiến trúc phù hợp | CNN tự xây dựng cho ảnh (E2/E3) | Ch3 + `best.pt` |
| 5 | Theo dõi train/val loss và metric | `metrics.csv` mỗi epoch + 4 hình learning curve | Ch4 |
| 6 | ≥1 thí nghiệm cải thiện/ablation | E1→E2→E3→E4 (bắt buộc), E5/E7 (nên) | Bảng ablation |
| 7 | Phân tích ≥10 mẫu dự đoán sai | Mục 13; bảng 10+ mẫu thật + phân tích 3 tầng | Ch5 |
| 8 | Demo dữ liệu chưa xuất hiện | Streamlit + 10–15 ảnh tự chụp ngoài dataset | Ch6 + `app.py` |
| 9 | Seed, config, requirements, README tái lập | `configs/`, `env.json`, `requirements.txt`, README | Phụ lục + repo |
| 10 | Không thu thập/công bố dữ liệu cá nhân nhạy cảm | Chỉ crop quanh biển; che mặt/biển số ảnh tự chụp | Ch2.5, Ch6 |

### 3.2. Yêu cầu riêng của đề tài GTSRB

| # | Yêu cầu | Cách đáp ứng | Trạng thái |
|---|---|---|---|
| R1 | CNN tự xây dựng | `src/models.py::SmallCNN` (không dùng pretrained) | Kế hoạch |
| R2 | Thử data augmentation và regularization | E3 (aug), E2 (BN+dropout+WD), E7 (class weight) | Kế hoạch |
| R3 | So sánh ≥2 cấu hình CNN | E1 vs E2 vs E3 vs E4 | Kế hoạch |
| R4 | Báo lỗi theo lớp và cặp biển dễ nhầm | Per-class P/R/F1 + top-10 confusion pairs | Kế hoạch |
| R5 | Demo ảnh ngoài dataset, không dùng cho quyết định thực tế | App + ảnh tự chụp + footer cảnh báo | Kế hoạch |
| R6 | Accuracy, Macro-F1, Confusion matrix | Mục 12 | Kế hoạch |
| R7 | Kiểm tra mất cân bằng lớp | EDA + biểu đồ phân bố + E7 | Kế hoạch |

### 3.3. Điểm mạnh/kế hoạch cũ và phần đã bổ sung

| Nội dung | Kế hoạch cũ | Bổ sung trong tài liệu này |
|---|---|---|
| Môi trường | "Dùng Colab hoặc máy cá nhân" | Chốt local RTX 5070 + PyTorch cu128 + cảnh báo Blackwell |
| Dữ liệu | Tải bằng torchvision | Thêm ROI, track ID, kiểm tra trùng lặp, tên lớp, license cần xác minh |
| Chia tập | Stratified 80/20 | Thêm **track-aware split** để đo mức lạc quan của val |
| Baseline | HOG+SVM | Thêm majority-class; quy tắc so sánh công bằng |
| Ablation | E2/E3/E4 | Thêm ma trận 10 run, ưu tiên P0/P1/P2, ngân sách thời gian |
| Đánh giá | Acc, F1, confusion | Thêm ngưỡng unknown, top-3, margin, bucket lỗi |
| Phân tích lỗi | Bảng 10 mẫu | Thêm định lượng theo độ mờ/độ sáng/diện tích biển |
| Demo | Streamlit + ảnh mới | Thêm 2 chế độ (đã crop / cảnh thật), xử lý unknown, ẩn danh |
| Tái lập | Seed + config + README | Thêm `env.json`, hash checkpoint, quy ước run, clean-room test |
| Quản lý | 3 tuần, chưa phân công | 15 ngày solo, có gate và definition of done |

---

## 4. Dữ liệu GTSRB

### 4.1. Nguồn và điều kiện sử dụng

| Hạng mục | Nội dung |
|---|---|
| Tên | German Traffic Sign Recognition Benchmark (GTSRB) |
| Bài báo gốc | Stallkamp, Schlipsing, Salmen, Igel, *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*, Neural Networks 32 (2012) 323–332 |
| Trang chính thức | `https://benchmark.ini.rub.de/gtsrb_dataset.html` |
| Cách tải khuyến nghị | `torchvision.datasets.GTSRB(root="data", split="train"/"test", download=True)` |
| Fallback | Tải thủ công từ trang chính thức hoặc mirror mà torchvision dùng; giải nén đúng cấu trúc `data/gtsrb/...` |
| Giấy phép | GTSRB được cung cấp **cho mục đích nghiên cứu**; **phải mở trang chính thức tại ngày truy cập và ghi lại đúng câu điều kiện sử dụng** vào Chương 2. Không tự suy diễn giấy phép. |
| Ngày truy cập cần ghi | Ghi ngày thực tế khi tải (dự kiến 22/10/2026) |
| Tái phân phối | Không đưa toàn bộ dataset vào repo/báo cáo; chỉ trích dẫn nguồn và dùng một số ảnh minh họa |

**Lưu ý thực tế:** trang `benchmark.ini.rub.de` có thể không truy cập được từ một số mạng (đã thử từ máy này ngày 22/10/2026). Khi đó:
1. Thử `download=True` của torchvision (dùng mirror riêng).
2. Nếu vẫn lỗi, tải thủ công và đặt đúng cấu trúc thư mục mà torchvision mong đợi.
3. Ghi lại cách tải thực tế + checksum file nén vào `outputs/results/data_report.json`.

### 4.2. Quy mô (số công bố — phải xác minh lại)

| Tập | Số ảnh công bố | Ghi chú |
|---|---|---|
| Train | 39.209 | 43 lớp, ảnh biến thiên kích thước |
| Test | 12.630 | Split chính thức, có file ground truth kèm ROI |
| Tổng | 51.839 | |

**Bắt buộc:** sau khi tải, đếm lại `len(train_ds)`, `len(test_ds)`, đếm theo từng lớp, ghi vào `data_report.json`. Số trong báo cáo là **số đếm thật**, không phải số công bố.

### 4.3. Cấu trúc dữ liệu và các đặc điểm phải kiểm tra

| Đặc điểm | Ảnh hưởng | Việc cần làm |
|---|---|---|
| Ảnh có **kích thước thay đổi** (khoảng 15×15 đến 250×250) | Không thể stack batch nếu không resize | Resize trong transform; ghi histogram kích thước gốc |
| Ảnh là **PPM**, torchvision trả `PIL.Image` RGB | Cần decode đúng | Kiểm tra mode ảnh, chuyển RGB |
| Có **ROI (bounding box)** trong file CSV GT | Cho phép crop biển, tính diện tích biển | Dùng cho E8 và phân tích lỗi |
| Ảnh trích từ **video**, frame liên tiếp gần trùng | Rò rỉ train/val nếu chia ngẫu nhiên | Xem mục 5.2 |
| **Mất cân bằng lớp** (~180–2.250 ảnh/lớp, tỷ lệ ~12:1) | Accuracy bị chi phối bởi lớp nhiều | Macro-F1, per-class report, E7 |
| **Không có tên lớp** trong torchvision | Dễ lệch nhãn khi demo | Tạo `configs/class_names.json` dùng chung |
| Có **nhãn nhiễu/ảnh mơ hồ** (biển bị che, nhiều biển, quá tối) | Trần accuracy và lỗi khó tránh | Nêu trong Ch5, tách "lỗi model" và "lỗi dữ liệu" |
| Nền ảnh có thể chứa người/xe | Vấn đề riêng tư khi đưa vào báo cáo | Crop quanh biển khi hiển thị |

### 4.4. Kiểm tra dữ liệu bắt buộc (D1)

- [ ] Đếm tổng số ảnh và số ảnh/lớp; lưu CSV.
- [ ] Vẽ biểu đồ cột số ảnh/lớp, sắp xếp tăng dần, tô màu lớp dưới 300 ảnh.
- [ ] Histogram kích thước ảnh gốc (width, height) và tỷ lệ khung.
- [ ] Kiểm tra ảnh lỗi/không đọc được, mode ảnh khác RGB.
- [ ] Hash toàn bộ ảnh (MD5/SHA1) → phát hiện ảnh trùng trong train.
- [ ] Đối chiếu hash train ↔ test → phát hiện trùng lặp giữa hai tập (nếu có, ghi rõ và loại khỏi train).
- [ ] Hiển thị lưới ảnh mẫu 5×9 (mỗi lớp 1 ảnh) kèm mã + tên lớp.
- [ ] Đọc thử 20 file CSV GT (train + test) để xác nhận cột ROI và ClassId khớp với nhãn torchvision.
- [ ] Ghi tất cả vào `data_report.json` + 3 hình vào `outputs/figures/`.

### 4.5. `class_names.json`

Cấu trúc:

```json
{
  "0": {"en": "Speed limit 20", "vi": "Giới hạn tốc độ 20 km/h", "shape": "circle", "color": "red"},
  "1": {"en": "Speed limit 30", "vi": "Giới hạn tốc độ 30 km/h", "shape": "circle", "color": "red"}
}
```

Bảng đầy đủ 43 lớp ở **Phụ lục A**. File này là nguồn duy nhất cho nhãn; dùng ở train, evaluate, confusion matrix, app.

---

## 5. Chia dữ liệu và chống rò rỉ

### 5.1. Chia tập

| Tập | Số ảnh (dự kiến) | Tỷ lệ | Mục đích | Quy tắc |
|---|---|---|---|---|
| Train | 31.360 | 80% của 39.201 | Cập nhật trọng số | Augmentation chỉ áp dụng ở đây |
| Validation | 7.841 | 20% của 39.201 | Chọn mô hình, early stopping, chọn ngưỡng unknown | Không cập nhật trọng số từ nhãn val |
| Test | 12.630 | Split chính thức | Đánh giá cuối cùng | Chỉ chạy **một lần** sau khi đóng băng config |

> **Cập nhật 22/10/2026:** GTSRB có 8 ảnh train trùng MD5 với test (nhãn 14; 8 ảnh train thuộc **track `00023` của tập train** — file test không có metadata track) nên 8 ảnh đó bị **loại khỏi tập huấn luyện**. File vẫn giữ đủ 39.209 trên đĩa để số đếm khớp công bố, nhưng chỉ chia split trên 39.201 ảnh (`usable_for_train=1` trong `outputs/results/train_meta.csv`); danh sách loại trừ ở `outputs/results/excluded_train_duplicates.csv`.

Cách chia: `sklearn.model_selection.train_test_split(..., stratify=labels, test_size=0.2, random_state=42)`.

Kiểm tra sau khi chia:
- [ ] Mỗi lớp đều có mặt trong val (43/43).
- [ ] Tỷ lệ từng lớp ở val xấp xỉ tỷ lệ ở train (±1%).
- [ ] Lớp hiếm nhất vẫn có ≥30 ảnh val (nếu không, ghi rõ và thận trọng khi đọc F1 lớp đó).

### 5.2. Rò rỉ do frame gần trùng (bắt buộc xử lý)

GTSRB được trích từ video; các frame liên tiếp trong cùng một track gần như giống nhau. Chia ngẫu nhiên khiến val chứa "anh em sinh đôi" của ảnh train → **val accuracy bị thổi phồng**, dẫn đến chọn sai mô hình.

**Kế hoạch:**
1. Trích **track ID** từ tên file (phần số đầu tiên; kiểm tra thực tế trên đĩa).
2. Tạo 2 bộ split:
   - `split_random`: stratified ngẫu nhiên (dùng cho báo cáo chính, so sánh được với các nghiên cứu dùng cách này).
   - `split_group`: chia theo track (GroupShuffleSplit, nhóm = track ID) — **val sạch hơn**.
3. Chạy mô hình chính trên **cả hai** split, báo cáo chênh lệch val accuracy:

```
Δ = acc_val(random) − acc_val(group)
```

Nếu Δ lớn (≥1–2%), đó là bằng chứng định lượng cho hiện tượng rò rỉ, một nhận xét rất có giá trị trong Ch5.

**Nếu không trích được track ID:** ghi rõ trong hạn chế, dùng split_random và so sánh val–test để ước lượng mức lạc quan.

### 5.3. Kỷ luật tập test (cơ chế thực thi, không chỉ khẩu hiệu)

- `evaluate.py --split test` yêu cầu cờ `--confirm-freeze` và ghi log mỗi lần chạy.
- Chỉ chạy test **sau** khi chốt `configs/final.yaml` và lưu `hash(best.pt)` + `git commit`.
- Kết quả test lưu vào `outputs/results/test_final.json`, **không ghi đè** (nếu chạy lại, tạo file `test_final_v2.json` và giải thích lý do trong báo cáo).
- Trong báo cáo ghi rõ: "Tập test được đánh giá đúng 1 lần cho mô hình cuối."

---

## 6. Tiền xử lý và augmentation

### 6.1. Chuẩn hóa dùng chung

| Bước | Giá trị | Ghi chú |
|---|---|---|
| Chuyển màu | RGB | Bắt buộc, kể cả ảnh grayscale |
| Resize | 64×64, interpolation bilinear | Dùng cùng tham số cho train/val/test/demo |
| Tensor | `ToTensor()` → [0,1] | |
| Normalize | mean/std **tính từ train** | Lưu vào `configs/final.yaml`; không dùng số của ImageNet |

Hàm `preprocess()` nằm **duy nhất** trong `src/data.py`, được import bởi train/evaluate/predict/app. Không viết lại ở notebook hay app.

### 6.2. Cache dữ liệu (tăng tốc)

- Decode + resize **một lần** toàn bộ 39.201 ảnh train (đã trừ 8 ảnh trùng test) và 12.630 ảnh test thành mảng `uint8`:
  - 64×64×3: train ≈ 480 MB, test ≈ 155 MB → giữ trong RAM (64 GB) hoặc lưu `.npy`.
- Augmentation thực hiện trên tensor/PIL sau khi lấy từ cache.
- Kỳ vọng: 1 epoch 31.360 ảnh, batch 128 (245 iteration) ≈ **10–20 giây** trên RTX 5070 với AMP (ước tính, đo thực tế ở epoch đầu).

### 6.3. Augmentation (chỉ train)

| Phép biến đổi | Tham số | Lý do |
|---|---|---|
| RandomAffine | degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5 | Mô phỏng góc chụp/lệch camera |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02 | Mô phỏng ánh sáng; hue nhỏ vì màu là tín hiệu ngữ nghĩa |
| RandomPerspective | distortion_scale=0.1, p=0.3 | Mô phỏng biển bị nghiêng |
| **Không** dùng | HorizontalFlip, VerticalFlip, Rotation 90/180 | Làm đổi nghĩa biển báo |

**Lưu ý:** augmentation mạnh có thể làm giảm accuracy trên GTSRB vì biển báo có hình dạng chuẩn. Mức trên là "nhẹ"; nếu E3 kém hơn E2, đó là kết quả hợp lệ và phải báo cáo trung thực, kèm giả thuyết.

### 6.4. Xử lý mất cân bằng

- Bắt buộc: dùng **macro-F1** làm chỉ số chính, báo cáo per-class.
- E7 (nên làm): so sánh `CrossEntropyLoss` thường vs **class weight = 1/√(tần suất)** (tránh over-weight lớp hiếm quá mức). Báo cáo đánh đổi accuracy ↔ macro-F1.
- Không dùng WeightedRandomSampler cho mô hình chính (làm thay đổi phân phối, khó so sánh); chỉ dùng trong E7 như một biến thể.

---

## 7. Baseline

### 7.1. E0a — Majority class (sàn tham chiếu)

- Dự đoán mọi ảnh là lớp nhiều nhất trong train.
- Mục đích: chứng minh accuracy cao không đồng nghĩa mô hình tốt khi dữ liệu mất cân bằng.
- Kỳ vọng: accuracy ≈ 5–6%, macro-F1 ≈ 0,002.

### 7.2. E0b — HOG + LinearSVC

| Thành phần | Thiết lập |
|---|---|
| Ảnh đầu vào | 64×64 RGB → grayscale |
| HOG | `orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys'` |
| Số chiều đặc trưng | ≈ 1.764 |
| Bộ phân loại | `LinearSVC(C=1.0, max_iter=5000)` (one-vs-rest, 43 lớp) |
| Chuẩn hóa | `StandardScaler` fit trên train |
| Chọn C | Thử {0.1, 1, 10} trên **val**, chọn theo val macro-F1 (ghi lại bảng thử) |
| Thời gian dự kiến | Trích đặc trưng 2–4 phút; huấn luyện 5–15 phút |

**Quy tắc so sánh công bằng với CNN:**
- Cùng `split_random` (cùng index train/val/test).
- Cùng metric, cùng cách tính.
- Ghi rõ HOG dùng grayscale và **không** augmentation — đây là baseline, không phải đối thủ được tối ưu.
- Đo thời gian train và thời gian suy luận/ảnh.

### 7.3. Bảng baseline cần điền

| Mô hình | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Thời gian train | Thời gian inference/ảnh |
|---|---|---|---|---|---|---|
| Majority | | | | | | |
| HOG + LinearSVC | | | | | | |

---

## 8. Kiến trúc CNN

### 8.1. Ba cấu hình

#### CNN-A — baseline "chưa tối ưu"

```
Input 3×64×64
├─ Conv3×3(3→32)  + ReLU + MaxPool2  → 32×32×32
├─ Conv3×3(32→64) + ReLU + MaxPool2  → 64×16×16
├─ Conv3×3(64→128)+ ReLU + MaxPool2  → 128×8×8
├─ Flatten → 8192
├─ FC(8192→256) + ReLU
└─ FC(256→43)
```
- Không BN, không dropout, không weight decay, không augmentation.
- Tham số ≈ **2,2 M**.

#### CNN-B — mô hình chính

```
Input 3×64×64
├─ Conv3×3(3→32)  + BN + ReLU + MaxPool2
├─ Conv3×3(32→64) + BN + ReLU + MaxPool2
├─ Conv3×3(64→128)+ BN + ReLU + MaxPool2
├─ Flatten → 8192
├─ Dropout(0.3) → FC(8192→256) + ReLU
├─ Dropout(0.3) → FC(256→43)
```
- Thêm BN, dropout 0.3, weight decay 1e-4, augmentation nhẹ.
- Tham số ≈ **2,2 M**.

#### CNN-C — sâu hơn, ít tham số hơn

```
Input 3×64×64
├─ Conv3×3(3→32)  + BN + ReLU + MaxPool2   → 32×32×32
├─ Conv3×3(32→64) + BN + ReLU + MaxPool2   → 64×16×16
├─ Conv3×3(64→128)+ BN + ReLU + MaxPool2   → 128×8×8
├─ Conv3×3(128→256)+ BN + ReLU + MaxPool2  → 256×4×4
├─ GlobalAvgPool → 256
├─ Dropout(0.4) → FC(256→43)
```
- Tham số ≈ **0,5 M**; GAP giảm overfit.
- Đây là lập luận kiến trúc: **sâu hơn nhưng ít tham số hơn** nhờ GAP.

### 8.2. Bảng so sánh kiến trúc (điền params thật bằng script)

| Mô hình | Số khối conv | BN | Dropout | GAP | Aug | Params | Ghi chú |
|---|---|---|---|---|---|---|---|
| CNN-A | 3 | Không | Không | Không | Không | ~2,2 M | Mốc so sánh |
| CNN-B | 3 | Có | 0,3 | Không | Có | ~2,2 M | Mô hình chính |
| CNN-C | 4 | Có | 0,4 | Có | Có | ~0,5 M | So sánh độ sâu |

**Bắt buộc:** in `sum(p.numel() for p in model.parameters())` và ghi vào báo cáo; không dùng số ước tính.

### 8.3. Vì sao chọn kiến trúc này

- Ảnh 64×64, đối tượng có cấu trúc hình học rõ → 3–4 khối conv 3×3 đủ receptive field.
- BN ổn định huấn luyện, cho phép lr cao hơn.
- Dropout + WD chống overfit khi train chỉ ~31k ảnh.
- GAP ở CNN-C giảm tham số tầng FC (8192×256 = 2,1 M) → giảm overfit.
- Không dùng pretrained vì đề yêu cầu **CNN tự xây dựng**.

---

## 9. Thiết lập huấn luyện

### 9.1. Siêu tham số mặc định (dùng cho mọi run, trừ khi run đó thay đổi có chủ đích)

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (E5 thử 3e-4 và 3e-3) |
| Weight decay | 1e-4 (CNN-A: 0) |
| Batch size | 128 |
| Số epoch tối đa | 40 |
| Early stopping | patience 7, theo **val macro-F1** |
| LR scheduler | CosineAnnealingLR (T_max = số epoch) |
| Loss | CrossEntropyLoss (E7: có class weight) |
| AMP | Bật (`torch.amp.autocast`) |
| Seed | 42 (multi-seed: 42, 123, 2024) |
| Checkpoint | `best.pt` theo val macro-F1 |

### 9.2. Tất định (determinism)

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# DataLoader
g = torch.Generator()
g.manual_seed(seed)
DataLoader(..., generator=g, worker_init_fn=seed_worker, num_workers=4,
           persistent_workers=True, pin_memory=True)
```

Ghi chú trong báo cáo: một số phép toán CUDA + AMP có thể không tất định tuyệt đối; chênh lệch giữa các lần chạy cùng seed thường <0,2%.

### 9.3. Ghi chú Windows

- `num_workers > 0` bắt buộc có `if __name__ == "__main__":` trong entrypoint.
- Kích hoạt venv: `Set-ExecutionPolicy -Scope Process Bypass` rồi `.\.venv\Scripts\Activate.ps1`.
- Đường dẫn dùng `pathlib.Path`, không hard-code ổ đĩa.

### 9.4. Quy trình một run

1. Đọc config YAML → validate key bắt buộc.
2. Set seed → build model → load train/val.
3. Mỗi epoch: train → eval val → ghi 1 dòng CSV → cập nhật scheduler → lưu `best.pt` nếu val macro-F1 tốt hơn.
4. Early stopping → kết thúc → lưu `config.yaml`, `env.json`, `curves.png`.
5. Không đụng đến test.

---

## 10. Ma trận thực nghiệm

### 10.1. Danh sách run

| ID | Cấu hình | Mục đích | Ưu tiên | Ước tính |
|---|---|---|---|---|
| E0a | Majority class | Sàn tham chiếu | P0 | <1 phút |
| E0b | HOG + LinearSVC | Baseline bắt buộc | P0 | 10–20 phút |
| E1 | CNN-A (plain, không aug/BN/dropout) | Mốc so sánh | P0 | ~10 phút |
| E2 | CNN-B (BN + dropout + WD, không aug) | Tác động regularization | P0 | ~12 phút |
| E3 | CNN-B + augmentation | **Mô hình chính (ứng viên)** | P0 | ~15 phút |
| E4 | CNN-C (4 khối + GAP + aug) | Tác động độ sâu/kiến trúc | P0 | ~15 phút |
| E5a | CNN-B + aug, lr=3e-4, 10 epoch | Tác động LR | P1 | ~5 phút |
| E5b | CNN-B + aug, lr=3e-3, 10 epoch | Tác động LR | P1 | ~5 phút |
| E6a | CNN-B + aug, 48×48 | Tác động độ phân giải | P2 | ~10 phút |
| E6b | CNN-B + aug, 96×96 | Tác động độ phân giải | P2 | ~25 phút |
| E7 | CNN-B + aug + class weight 1/√freq | Cân bằng lớp → macro-F1 | P1 | ~15 phút |
| E8 | CNN-B + aug + **ROI crop** | So sánh full image vs ROI | P1 | ~15 phút |
| E9 | ResNet18 fine-tune | "Trần" tham khảo (không phải mô hình chính) | P2 | ~20 phút |
| E10 | 3 seed cho mô hình thắng (42, 123, 2024) | mean ± std | P1 | 2 run thêm |

**Tối thiểu để đạt yêu cầu:** E0a, E0b, E1, E2, E3, E4.
**Khuyến nghị:** thêm E5, E7, E8, E10.
**Tổng thời gian GPU ước tính cho P0 + P1:** 2–3 giờ (chưa kể đọc hình/viết nhận xét).

### 10.2. Quy tắc so sánh

- Mỗi lần chỉ thay **một nhóm yếu tố**; nếu buộc phải gộp (ví dụ E2 gộp BN + dropout + WD), ghi rõ là "gói regularization" và không kết luận riêng cho từng yếu tố.
- Cùng seed, cùng số epoch tối đa, cùng early stopping, cùng split.
- Báo cáo kèm **params**, **thời gian/epoch**, **epoch dừng**.
- Chênh lệch <0,3% tuyệt đối: coi là **tương đương**, không tuyên bố "tốt hơn" nếu chưa có multi-seed.
- Kết luận cuối dựa trên **test** của mô hình đã chốt, không dựa trên val.

### 10.3. Bảng kết quả tổng hợp (mẫu — điền sau khi chạy)

| ID | Mô hình | Params | Aug | Epoch (best) | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Thời gian |
|---|---|---|---|---|---|---|---|---|---|
| E0a | Majority | 0 | – | – | | | | | |
| E0b | HOG+SVC | – | Không | – | | | | | |
| E1 | CNN-A | | Không | | | | | | |
| E2 | CNN-B | | Không | | | | | | |
| E3 | CNN-B | | Có | | | | | | |
| E4 | CNN-C | | Có | | | | | | |
| E5a | CNN-B, lr 3e-4 | | Có | | | | | | |
| E5b | CNN-B, lr 3e-3 | | Có | | | | | | |
| E7 | + class weight | | Có | | | | | | |
| E8 | + ROI crop | | Có | | | | | | |

---

## 11. Logging, checkpoint và quy ước artifact

### 11.1. Cấu trúc một run

```
outputs/runs/e3_cnnb_aug_s42/
├── config.yaml        # toàn bộ siêu tham số của run
├── env.json           # python, torch, cuda, gpu, driver, os
├── metrics.csv        # 1 dòng / epoch
├── best.pt            # state_dict + config + class_names + mean/std
├── curves.png         # loss & acc & macro-F1 theo epoch
└── test_report.json   # chỉ sinh khi evaluate test (run cuối)
```

### 11.2. Schema `metrics.csv`

```
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
1,0.001000,2.8451,0.3120,1.1024,0.6410,0.5987,14.2,1024
```

Quy tắc:
- Ghi **sau mỗi epoch**, flush ngay (không giữ trong RAM) để mất điện vẫn còn dữ liệu.
- `val_macro_f1` tính bằng `sklearn.metrics.f1_score(average="macro", zero_division=0)`.
- Không ghi đè file của run cũ.

### 11.3. `summary.csv` — nguồn số liệu duy nhất cho báo cáo

Cột: `run_id, model, params, aug, epochs_run, best_epoch, val_acc, val_macro_f1, test_acc, test_macro_f1, train_time_s, infer_ms_per_img, seed, notes`.

Quy trình: script `src/summarize.py` quét `outputs/runs/*/` → sinh `outputs/results/summary.csv`. **Mọi bảng trong báo cáo/slide copy từ file này**, không gõ tay.

### 11.4. Đặt tên run

`<exp_id>_<model>_<feature>_s<seed>` — ví dụ: `e3_cnnb_aug_s42`, `e4_cnnc_gap_aug_s42`, `e0b_hog_svc_s42`.

---

## 12. Đánh giá và chỉ số

### 12.1. Chỉ số

| Chỉ số | Công thức / công cụ | Vai trò |
|---|---|---|
| Accuracy | `accuracy_score` | Chỉ số phụ, dễ so sánh với công bố |
| Macro-F1 | `f1_score(average="macro")` | **Chỉ số chính** — mỗi lớp trọng số bằng nhau |
| Per-class P/R/F1 + support | `classification_report(output_dict=True)` | Phát hiện lớp yếu |
| Confusion matrix | `confusion_matrix(normalize="true")` | Cặp nhầm |
| Top-3 accuracy | tự tính từ logits | Trải nghiệm demo |
| Độ tin cậy + margin | `p1`, `p1 − p2` | Phân tích lỗi, ngưỡng unknown |

Ghi rõ trong báo cáo: vì sao macro-F1 quan trọng hơn accuracy khi dữ liệu mất cân bằng (lớp ít mẫu vẫn được tính ngang lớp nhiều mẫu).

### 12.2. Ngưỡng "không xác định" (unknown)

- Model luôn trả về 1 trong 43 lớp → ảnh không có biển vẫn bị gán nhãn với confidence cao.
- Chọn ngưỡng `τ` trên **val**: giữ lại ~95% ảnh in-distribution (percentile 5 của max-prob).
- Báo cáo: acceptance rate trên ảnh in-distribution, rejection rate trên ảnh không phải biển báo.
- Đây là phần bắt buộc của demo để trung thực về giới hạn mô hình.

### 12.3. Hình cần có

1. Train/val loss theo epoch (4 run chính trên cùng một hình hoặc 2×2).
2. Train/val accuracy theo epoch.
3. Val macro-F1 theo epoch.
4. Confusion matrix chuẩn hóa theo hàng (43×43) — bản đầy đủ + bản zoom 15 lớp yếu nhất.
5. F1 theo lớp (bar chart, sắp tăng dần, chú thích support).
6. Top-10 cặp nhầm (bar chart ngang).
7. Error rate theo tertile độ mờ / độ sáng / diện tích biển (3 subplot).
8. Lưới 10+ ảnh dự đoán sai kèm nhãn thật/dự đoán/confidence.

---

## 13. Phân tích lỗi

### 13.1. Đầu ra bắt buộc

`outputs/runs/<final>/predictions.csv`:

```
path,true_id,true_name,pred_id,pred_name,prob_true,prob_pred,margin,top3,
blur_laplacian,brightness_mean,roi_area_ratio,is_correct
```

Sinh tự động bởi `src/evaluate.py`; **không** sửa tay.

### 13.2. Ba tầng phân tích

**Tầng 1 — Theo lớp**
- Bảng per-class P/R/F1 + support, sắp theo F1 tăng dần.
- Nhận xét: lớp F1 thấp có tương quan với support thấp không? Có lớp nào F1 thấp dù support cao (dấu hiệu nhãn nhiễu/khó)?

**Tầng 2 — Theo nguyên nhân (định lượng)**
- Tính cho mỗi ảnh test: `blur_laplacian` (phương sai Laplacian trên ảnh xám), `brightness_mean` (mean kênh V), `roi_area_ratio` (từ ROI trong CSV GT).
- Chia mỗi biến thành 3 nhóm (tertile) → bảng error rate theo nhóm.
- Kết luận bằng số: ví dụ "nhóm ảnh mờ nhất có error rate gấp 3,2 lần nhóm nét nhất".
- Tách rõ **quan sát được** (số liệu) và **giả thuyết** (ví dụ "có thể do biển bị che một phần").

**Tầng 3 — Cặp nhầm**
- Top 10 cặp `(true → pred)` nhiều nhất kèm số lượng và 3 ảnh ví dụ mỗi cặp.
- Dự kiến (phải kiểm chứng bằng ma trận thật): các cặp giới hạn tốc độ liền kề (20/30/50/80/100/120), cặp rẽ trái/phải, keep left/right, road work/children crossing.

### 13.3. Bảng 10+ mẫu sai (mẫu để điền)

| # | Ảnh | Nhãn thật | Dự đoán | Confidence | Margin | Quan sát được | Giả thuyết |
|---|---|---|---|---|---|---|---|
| 1 | `figures/err/01.png` | | | | | | |
| 2 | | | | | | | |
| … | | | | | | | |
| 10 | | | | | | | |

**Yêu cầu:** ảnh phải lấy từ lần chạy thật; nhận xét phải dựa trên việc xem ảnh, không suy đoán chung chung.

### 13.4. Phân tích ngoài phân phối (OOD)

- Bộ 10–15 ảnh tự chụp: biển Việt Nam, ảnh cảnh đường có biển nhỏ, ảnh không có biển.
- Báo cáo: dự đoán, confidence, đúng/sai so với kỳ vọng.
- Kết luận: model học đặc trưng biển Đức; ảnh biển Việt Nam chỉ nên dùng minh họa domain shift, **không** kết luận model "nhận diện biển Việt Nam".

### 13.5. Taxonomy lỗi (dùng khi viết nhận xét)

| Loại | Mô tả | Cách nhận biết |
|---|---|---|
| E-A | Ảnh mờ / rung | Laplacian thấp |
| E-B | Thiếu sáng / ngược sáng | brightness thấp/cao bất thường |
| E-C | Biển quá nhỏ trong khung | roi_area_ratio thấp |
| E-D | Bị che một phần | quan sát ảnh |
| E-E | Hai lớp giống nhau về hình/màu | cặp nhầm lặp lại |
| E-F | Nhãn nhiễu / ảnh mơ hồ | ảnh không rõ ngay cả với người |
| E-G | Lỗi OOD | ảnh ngoài GTSRB |

---

## 14. Demo dữ liệu mới

### 14.1. Yêu cầu chức năng

- Streamlit, upload JPG/JPEG/PNG.
- Kiểm tra file hợp lệ, chuyển RGB, resize 64×64, normalize **bằng đúng hàm `preprocess()` của train**.
- Hiển thị: ảnh, top-3 nhãn + xác suất, cảnh báo nếu max-prob < τ.
- Xử lý lỗi: file hỏng, ảnh quá lớn (>20 MB), checkpoint không tồn tại.
- Footer cố định: *"Demo học thuật trên GTSRB (biển báo Đức). Không dùng cho quyết định giao thông thực tế."*

### 14.2. Hai chế độ

| Chế độ | Input | Kỳ vọng | Mục đích |
|---|---|---|---|
| **A. Ảnh đã crop biển** | Ảnh biển chiếm phần lớn khung | Accuracy cao, gần test | Chứng minh pipeline đúng |
| **B. Ảnh cảnh thật** | Biển nhỏ trong khung lớn | Có thể sai nhiều | Minh họa giới hạn + domain shift |

Chế độ B phải hiển thị **cảnh báo cần crop** và (nếu có) kết quả của bước detect đơn giản.

### 14.3. Bộ ảnh ngoài dataset

- 10–15 ảnh nhóm tự chụp (biển công cộng), ưu tiên biển giới hạn tốc độ/cấm để so sánh với hệ Đức.
- Không chụp người, không chụp biển số xe rõ nét.
- `external_test/manifest.csv`: `filename, nguồn, ngày chụp, mô tả, đã ẩn danh (Y/N), nhãn kỳ vọng`.
- Kiểm tra từng ảnh: crop quanh biển; nếu có người/biển số → blur trước khi đưa vào repo/báo cáo.

### 14.4. Kịch bản demo khi bảo vệ

1. Mở app, upload 1 ảnh test chưa từng dùng để train → đúng, top-3.
2. Upload 1 ảnh tự chụp đã crop → kết quả + nhận xét.
3. Upload 1 ảnh cảnh thật (biển nhỏ) → sai → giải thích domain shift.
4. Upload 1 ảnh không có biển → model vẫn gán nhãn; bật ngưỡng unknown → "không xác định".
5. Nêu rõ: đây là demo học thuật, không dùng thực tế.

---

## 15. Tái lập kết quả

### 15.1. Cấu trúc repo

```
BTLHocSau/
├── data/                          # .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                  # đọc/validate YAML
│   ├── data.py                    # load, split, cache, preprocess, loader
│   ├── models.py                  # SmallCNN, build_model
│   ├── baseline.py                # majority, HOG+SVC
│   ├── train.py                   # train_one
│   ├── evaluate.py                # metrics, predictions.csv, confusion
│   ├── error_analysis.py          # bucket, top pairs, grid
│   ├── predict.py                 # load_model, preprocess, predict_topk
│   └── summarize.py               # sinh summary.csv
├── configs/
│   ├── default.yaml
│   ├── e1_cnna.yaml … e8_roi.yaml
│   ├── class_names.json
│   └── final.yaml                 # config đã đóng băng
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results.ipynb
│   └── 03_error_analysis.ipynb
├── outputs/
│   ├── runs/
│   ├── figures/
│   └── results/
├── external_test/                 # ảnh tự chụp đã ẩn danh + manifest.csv
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── (report.docx, presentation.pptx)
```

### 15.2. Lệnh chuẩn

```powershell
# Cài đặt
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Kiểm tra GPU (phải thấy (12, 0) với RTX 5070)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"

# Dữ liệu
python -m src.data --download --report

# Train / evaluate
python -m src.train --config configs/e3_cnnb_aug.yaml --seed 42
python -m src.evaluate --run outputs/runs/e3_cnnb_aug_s42 --split val
python -m src.evaluate --run outputs/runs/e3_cnnb_aug_s42 --split test --confirm-freeze

# Tổng hợp + demo
python -m src.summarize
streamlit run app.py
```

### 15.3. `requirements.txt` (pin sau khi cài thành công)

```
torch>=2.7
torchvision>=0.22
numpy
pandas
scikit-learn
scikit-image
matplotlib
seaborn
streamlit
pyyaml
tqdm
pillow
imagehash
```

Lưu ý: `torch`/`torchvision` phải cài từ index cu128, không cài từ PyPI mặc định (có thể ra bản CPU).

### 15.4. `env.json`

```json
{
  "python": "3.12.x",
  "torch": "2.x.x+cu128",
  "torchvision": "0.2x.x",
  "cuda": "12.8",
  "gpu": "NVIDIA GeForce RTX 5070 12GB",
  "driver": "610.88",
  "os": "Windows ...",
  "cpu": "i9-14900KF",
  "ram_gb": 64
}
```

### 15.5. Clean-room test (bắt buộc trước khi nộp)

- [ ] Tạo venv mới tinh, cài đúng `requirements.txt`.
- [ ] Chạy README từ đầu: tải data → train 1 epoch → evaluate → app.
- [ ] Không có đường dẫn tuyệt đối nào trong code.
- [ ] Xóa `outputs/runs/` tạm và chạy lại 1 run ngắn để chắc script không phụ thuộc file cũ.

---

## 16. Kế hoạch 15 ngày (solo)

> Mỗi ngày có **definition of done (DoD)** — chỉ chuyển sang ngày tiếp theo khi DoD đạt.

| Ngày | Công việc | DoD |
|---|---|---|
| **D1** | Cài Python 3.12 + venv + PyTorch cu128; `git init`; tải GTSRB; EDA + hash + `data_report.json` | `torch.cuda.is_available()=True`, capability (12,0), smoke test matmul OK; có 3 hình EDA + số đếm thật |
| **D2** | Cache `.npy`; split random + group; DataLoader; vòng train có CSV/curves/checkpoint; `env.json` | 1 epoch chạy <60 s; `metrics.csv` đúng schema; curves sinh tự động |
| **D3** | E0a + E0b | Bảng baseline hoàn chỉnh (acc, macro-F1, thời gian) |
| **D4** | E1 + E2 | 2 run xong, đủ log + checkpoint + curves |
| **D5** | E3 + E4 | 4 cấu hình CNN so sánh được trên cùng bảng |
| **D6** | E5a/E5b + E7; chọn mô hình theo val macro-F1 | `configs/final.yaml` được tạo, có lý do chọn |
| **D7** | Multi-seed (E10) + **evaluate test 1 lần** | `test_final.json`, hash checkpoint, commit hash |
| **D8** | Per-class report, confusion matrix, top cặp nhầm, bucket lỗi | 4 hình + 3 bảng cho Ch4–5 |
| **D9** | Chọn + viết 10 mẫu sai; E8 (ROI) nếu kịp | Bảng 10 mẫu hoàn chỉnh, có nhận xét |
| **D10** | App Streamlit + ảnh tự chụp + ẩn danh + ngưỡng unknown | App chạy, `manifest.csv` đủ, thử 5 kịch bản |
| **D11** | `requirements.txt`, README, `summary.csv`, clean-room test | Venv sạch chạy lại được toàn bộ |
| **D12** | Viết Ch1–Ch3 | Bản nháp có số liệu thật |
| **D13** | Viết Ch4–Ch6 + phụ lục | Bản nháp đầy đủ |
| **D14** | Slide 12 trang | Slide khớp số liệu trong `summary.csv` |
| **D15** | Buffer + tổng duyệt + chuẩn bị phản biện | Checklist mục 21 đạt 100% |

**Quy tắc solo:**
- Viết báo cáo **song song** từ D2 (Chương 2), không dồn cuối.
- Mọi run chạy **unattended** bằng script; trong lúc GPU chạy thì viết văn/ làm hình.
- Nếu trễ: cắt E6/E9 trước, **không cắt** phân tích lỗi, demo, tái lập.
- Mỗi ngày commit ít nhất 1 lần với message rõ ràng.

---

## 17. Rủi ro và đối phó

| # | Rủi ro | Khả năng | Ảnh hưởng | Đối phó |
|---|---|---|---|---|
| R1 | Không cài được PyTorch đúng CUDA cho RTX 5070 (Blackwell) | Cao (máy chưa có Python) | Chặn toàn bộ | Làm ngay D1; dùng index cu128; kiểm tra capability (12,0); fallback Colab nếu bí |
| R2 | Tải GTSRB lỗi (trang chính thức không truy cập được) | Trung bình | Chậm D1 | Thử `download=True`; tải thủ công/mirror; ghi nguồn thực tế |
| R3 | Val cao – test thấp | Cao | Kết luận sai | Track-aware split; báo cáo Δ; không tune test |
| R4 | Demo ảnh thật sai nhiều | Rất cao | Mất điểm demo | ROI/crop + ngưỡng unknown + giải thích domain shift |
| R5 | Kết luận từ 1 seed | Cao | Phản biện | 3 seed cho mô hình thắng, báo mean±std |
| R6 | Hết thời gian viết báo cáo | Cao (solo) | Nặng | Viết từ D2; template từ D1; cắt P2 |
| R7 | Số liệu báo cáo lệch slide | Trung bình | Trừ điểm | Chỉ dùng `summary.csv` |
| R8 | Vi phạm riêng tư ảnh tự chụp | Thấp nhưng nghiêm trọng | Vi phạm yêu cầu môn | Crop biển, blur mặt/biển số, ghi manifest |
| R9 | Mất kết quả do ghi đè | Trung bình | Mất thời gian | Mỗi run 1 thư mục; không ghi đè; commit thường xuyên |
| R10 | Overfit do augmentation quá mạnh | Trung bình | Giảm accuracy | Aug nhẹ; so sánh E2 vs E3; báo cáo trung thực |

---

## 18. Cấu trúc báo cáo

### Chương 1 — Giới thiệu
- Bối cảnh và động lực (biển báo trong giao thông, tự động hóa).
- Phát biểu bài toán: input, output, 43 lớp, 12.288 chiều.
- Đối tượng sử dụng, giá trị thực tế, **giới hạn sử dụng**.
- Mục tiêu và phạm vi (không làm detection, không nhận diện biển Việt Nam).
- Cấu trúc báo cáo.

### Chương 2 — Dữ liệu
- Nguồn, bài báo gốc, ngày truy cập, điều kiện sử dụng.
- Số mẫu thật, số lớp, phân bố lớp, biểu đồ mất cân bằng.
- Đặc điểm ảnh, ROI, track, nhãn nhiễu.
- Tiền xử lý, chuẩn hóa, augmentation.
- Cách chia train/val/test + bảng số lượng + seed.
- Kiểm tra chất lượng dữ liệu (hash, trùng lặp, ảnh lỗi).

### Chương 3 — Phương pháp
- Baseline majority + HOG/SVM (công thức, tham số).
- Kiến trúc CNN-A/B/C (sơ đồ, bảng params, lý do).
- Loss, optimizer, scheduler, early stopping, checkpoint rule.
- Thiết lập thực nghiệm, seed, môi trường, phần cứng.
- Chỉ số đánh giá và vì sao chọn macro-F1.

### Chương 4 — Kết quả
- Learning curves (loss, acc, macro-F1) cho 4 run chính.
- Bảng so sánh đầy đủ (baseline + CNN + ablation).
- Phân tích ablation: yếu tố nào giúp bao nhiêu.
- Kết quả test cuối cùng + multi-seed mean±std.
- Nhận xét: mô hình nào thắng, vì sao, có phù hợp trực giác không.

### Chương 5 — Phân tích lỗi
- Confusion matrix + top cặp nhầm.
- Per-class F1 + support.
- Error rate theo độ mờ/độ sáng/diện tích biển.
- 10+ mẫu sai chi tiết (bảng + ảnh).
- Phân tích OOD (ảnh Việt Nam, ảnh không có biển).
- Nguyên nhân quan sát được vs giả thuyết; giới hạn mô hình.

### Chương 6 — Demo và kết luận
- Kiến trúc app, cách dùng, 2 chế độ.
- Kịch bản demo + kết quả thật.
- Tái lập: lệnh chạy, môi trường.
- Hạn chế (domain shift, không có unknown class, nhãn nhiễu).
- Hướng phát triển (detection + classification, transfer learning, mobile, biển Việt Nam).

### Tài liệu tham khảo
- GTSRB paper + trang dữ liệu.
- Torchvision, PyTorch, scikit-learn docs.
- 1–2 bài về CNN cho traffic sign (có thể dùng làm bối cảnh).

### Phụ lục
- A: bảng 43 lớp.
- B: config cuối.
- C: schema `metrics.csv` + `predictions.csv`.
- D: lệnh chạy.
- E: bảng kết quả đầy đủ theo lớp.

---

## 19. Cấu trúc slide

| # | Slide | Nội dung |
|---|---|---|
| 1 | Tiêu đề | Tên đề tài, môn, nhóm, ngày |
| 2 | Bài toán | Input/output, 43 lớp, ví dụ ảnh, giá trị thực tế |
| 3 | Dữ liệu | GTSRB, 39.209/12.630, phân bố lớp, mất cân bằng |
| 4 | Pipeline | Sơ đồ: data → split → aug → model → eval → demo |
| 5 | Baseline | Majority + HOG/SVM, kết quả |
| 6 | Kiến trúc CNN | Sơ đồ A/B/C, bảng params |
| 7 | Thiết lập huấn luyện | Loss, optimizer, schedule, early stopping, seed |
| 8 | Learning curves | Loss/acc/F1 train vs val |
| 9 | Kết quả & ablation | Bảng so sánh, nhận xét từng yếu tố |
| 10 | Confusion & lỗi | Ma trận, top cặp nhầm, 3 ảnh sai tiêu biểu |
| 11 | Demo | Ảnh mới, top-3, ca đúng/sai, ngưỡng unknown |
| 12 | Kết luận | Đóng góp, hạn chế, hướng phát triển, cảnh báo sử dụng |

---

## 20. Câu hỏi phản biện và trả lời gợi ý

| Câu hỏi | Trả lời gợi ý |
|---|---|
| Vì sao không dùng transfer learning? | Đề yêu cầu CNN tự xây dựng; transfer learning chỉ là thí nghiệm tham khảo (E9) để so sánh trần, không phải mô hình chính |
| Vì sao chọn 64×64? | Cân bằng chi tiết/chi phí; E6 so sánh 48/96 để trả lời bằng số |
| Macro-F1 khác accuracy thế nào? | Accuracy tính trung bình theo mẫu → lớp nhiều chi phối; macro-F1 trung bình theo lớp → lớp hiếm được tính ngang |
| Vì sao val cao hơn test? | Val cùng track/nguồn với train (frame gần trùng); test là phiên ghi khác → phân phối khác. Đã đo bằng Δ random vs group split |
| BatchNorm giúp gì? | Chuẩn hóa phân phối đầu vào từng layer → ổn định, cho phép lr cao, giảm phụ thuộc khởi tạo |
| Vì sao không lật ngang? | Lật biển "rẽ trái" thành "rẽ phải" → đổi nhãn, tạo nhãn sai |
| Làm sao biết overfit? | Train loss giảm, val loss tăng; khoảng cách train–val acc tăng; theo dõi curves |
| Ảnh không có biển báo thì sao? | Model luôn trả 1 trong 43 lớp; dùng ngưỡng τ để từ chối, đã đo acceptance/rejection |
| Dùng được ở Việt Nam chưa? | Chưa: khác hệ biển, khác bố cục/màu; chỉ minh họa domain shift |
| 99% có đáng tin? | Test cùng phân phối GTSRB, biển chiếm phần lớn ảnh; ảnh thực tế khó hơn nhiều — đã chứng minh bằng demo |
| Tại sao chọn AdamW? | Kết hợp Adam + weight decay tách biệt, ổn định cho CNN nhỏ; E5 kiểm tra độ nhạy lr |
| Ablation có công bằng? | Cùng split, seed, epoch, early stopping; mỗi run chỉ đổi nhóm yếu tố được nêu |
| Seed có đảm bảo tái lập tuyệt đối? | Gần đúng; AMP/CUDA có thể lệch nhỏ <0,2%, đã ghi trong hạn chế |
| Xử lý mất cân bằng thế nào? | Macro-F1 làm chỉ số chính; E7 thử class weight; báo cáo đánh đổi |
| Nếu có thêm thời gian? | Detection + classification, mobile, biển Việt Nam, calibration, ensemble |

---

## 21. Checklist nghiệm thu

### 21.1. Theo yêu cầu môn học

- [ ] Bài toán phát biểu rõ, có giới hạn sử dụng.
- [ ] Nguồn dữ liệu, điều kiện sử dụng, số mẫu/lớp, cách chia được ghi rõ **bằng số thật**.
- [ ] Có baseline và CNN, so sánh trên cùng quy trình.
- [ ] Có log train/val + learning curves, không chỉ 1 accuracy cuối.
- [ ] Có ablation và giải thích tác động.
- [ ] Có kết quả test độc lập sau khi chốt mô hình.
- [ ] Có ≥10 mẫu sai được phân tích từ dữ liệu thật.
- [ ] Demo chạy với ảnh chưa dùng trong huấn luyện.
- [ ] Seed, config, requirements, README tái lập được.
- [ ] Báo cáo và slide khớp số liệu thực tế.
- [ ] Không có dữ liệu cá nhân nhạy cảm; ảnh tự chụp đã ẩn danh.

### 21.2. Theo yêu cầu riêng đề tài

- [ ] CNN tự xây dựng (không pretrained cho mô hình chính).
- [ ] Có augmentation + regularization.
- [ ] So sánh ≥2 cấu hình CNN.
- [ ] Báo lỗi theo lớp + cặp dễ nhầm.
- [ ] Demo ảnh ngoài dataset + cảnh báo không dùng thực tế.
- [ ] Accuracy, macro-F1, confusion matrix đầy đủ.
- [ ] Kiểm tra mất cân bằng lớp.

### 21.3. Artifact phải có khi nộp

- [ ] `src/` chạy được + `app.py`.
- [ ] `configs/final.yaml` + `class_names.json`.
- [ ] `outputs/runs/*/best.pt` (mô hình cuối) + `metrics.csv` + `curves.png`.
- [ ] `outputs/results/summary.csv`, `test_final.json`, `data_report.json`.
- [ ] `outputs/figures/` đủ 8 hình ở mục 12.3.
- [ ] `external_test/` + `manifest.csv` (đã ẩn danh).
- [ ] `requirements.txt`, `README.md`, `env.json`.
- [ ] `report.docx`, `presentation.pptx`.

---

## 22. Phụ lục

### Phụ lục A — 43 lớp GTSRB

| ID | Tên tiếng Anh | Tên tiếng Việt | Số ảnh train (điền từ dữ liệu) |
|---|---|---|---|
| 0 | Speed limit 20 | Giới hạn tốc độ 20 km/h | |
| 1 | Speed limit 30 | Giới hạn tốc độ 30 km/h | |
| 2 | Speed limit 50 | Giới hạn tốc độ 50 km/h | |
| 3 | Speed limit 60 | Giới hạn tốc độ 60 km/h | |
| 4 | Speed limit 70 | Giới hạn tốc độ 70 km/h | |
| 5 | Speed limit 80 | Giới hạn tốc độ 80 km/h | |
| 6 | End of speed limit 80 | Hết giới hạn tốc độ 80 km/h | |
| 7 | Speed limit 100 | Giới hạn tốc độ 100 km/h | |
| 8 | Speed limit 120 | Giới hạn tốc độ 120 km/h | |
| 9 | No passing | Cấm vượt | |
| 10 | No passing > 3.5 t | Cấm vượt cho xe trên 3,5 tấn | |
| 11 | Right-of-way at next intersection | Đường ưu tiên tại giao lộ tới | |
| 12 | Priority road | Đường ưu tiên | |
| 13 | Yield | Nhường đường | |
| 14 | Stop | Dừng lại | |
| 15 | No vehicles | Cấm xe cộ | |
| 16 | Vehicles > 3.5 t prohibited | Cấm xe trên 3,5 tấn | |
| 17 | No entry | Cấm đi vào | |
| 18 | General caution | Nguy hiểm chung | |
| 19 | Dangerous curve left | Đường cong nguy hiểm bên trái | |
| 20 | Dangerous curve right | Đường cong nguy hiểm bên phải | |
| 21 | Double curve | Đường cong liên tiếp | |
| 22 | Bumpy road | Đường gồ ghề | |
| 23 | Slippery road | Đường trơn | |
| 24 | Road narrows on right | Đường hẹp bên phải | |
| 25 | Road work | Công trường thi công | |
| 26 | Traffic signals | Đèn tín hiệu giao thông | |
| 27 | Pedestrians | Người đi bộ | |
| 28 | Children crossing | Trẻ em qua đường | |
| 29 | Bicycles crossing | Xe đạp qua đường | |
| 30 | Beware of ice/snow | Cảnh báo băng tuyết | |
| 31 | Wild animals crossing | Động vật hoang dã qua đường | |
| 32 | End speed/passing limits | Hết mọi giới hạn tốc độ và cấm vượt | |
| 33 | Turn right ahead | Rẽ phải phía trước | |
| 34 | Turn left ahead | Rẽ trái phía trước | |
| 35 | Ahead only | Chỉ đi thẳng | |
| 36 | Go straight or right | Đi thẳng hoặc rẽ phải | |
| 37 | Go straight or left | Đi thẳng hoặc rẽ trái | |
| 38 | Keep right | Đi bên phải | |
| 39 | Keep left | Đi bên trái | |
| 40 | Roundabout mandatory | Vòng xuyến bắt buộc | |
| 41 | End of no passing | Hết cấm vượt | |
| 42 | End of no passing > 3.5 t | Hết cấm vượt cho xe trên 3,5 tấn | |

### Phụ lục B — Config mẫu (`configs/e3_cnnb_aug.yaml`)

```yaml
experiment:
  id: e3_cnnb_aug
  seed: 42
  notes: "CNN-B + BatchNorm + Dropout0.3 + WD1e-4 + augmentation nhe"

data:
  root: data
  image_size: 64
  num_classes: 43
  val_ratio: 0.2
  split: random          # random | group
  cache: true
  num_workers: 4
  mean: [0.0, 0.0, 0.0]  # dien sau khi tinh tu train
  std:  [1.0, 1.0, 1.0]

augmentation:
  enabled: true
  rotation: 10
  translate: 0.1
  scale_min: 0.9
  scale_max: 1.1
  shear: 5
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.02
  perspective: 0.1
  perspective_p: 0.3

model:
  name: cnn_b
  widths: [32, 64, 128]
  use_bn: true
  dropout: 0.3
  gap: false
  fc_hidden: 256

train:
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 128
  max_epochs: 40
  patience: 7
  scheduler: cosine
  amp: true
  monitor: val_macro_f1
  class_weight: none      # none | inv_sqrt_freq

output:
  dir: outputs/runs
```

### Phụ lục C — Schema `predictions.csv`

```
path,true_id,true_name,pred_id,pred_name,prob_true,prob_pred,margin,
top3,blur_laplacian,brightness_mean,roi_area_ratio,is_correct
```

- `top3`: chuỗi `"id:prob|id:prob|id:prob"`.
- `blur_laplacian`: phương sai Laplacian trên ảnh xám 64×64.
- `brightness_mean`: mean kênh V (HSV).
- `roi_area_ratio`: `(x2-x1)*(y2-y1)/(W*H)` từ CSV GT; NA nếu không có.

### Phụ lục D — Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks 32 (2012) 323–332.
2. GTSRB dataset — `https://benchmark.ini.rub.de/gtsrb_dataset.html` (kiểm tra lại ngày truy cập).
3. Torchvision datasets — `https://docs.pytorch.org/vision/main/datasets.html`
4. PyTorch — `https://pytorch.org/docs/stable/index.html`
5. scikit-learn — `https://scikit-learn.org/stable/`
6. Streamlit — `https://docs.streamlit.io/`

### Phụ lục E — Lịch sử thay đổi tài liệu

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 1.0 | 22/10/2026 | Bản chủ đầu tiên: khóa phạm vi 43 lớp @ 64×64, local RTX 5070, `src/` + CLI + notebook, solo 15 ngày; bổ sung track-aware split, ROI, ngưỡng unknown, phân tích lỗi định lượng, ma trận thực nghiệm, checklist nghiệm thu |
