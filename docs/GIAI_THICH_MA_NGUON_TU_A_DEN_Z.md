# Giải thích mã nguồn GTSRB-CNN từ A đến Z

> **Dành cho:** người mới nhập môn học sâu, chưa quen PyTorch/pipeline huấn luyện.
> **Mục tiêu:** đọc xong có thể (1) hiểu repo làm gì, (2) giải thích từng bước cho người khác,
> (3) tự chạy lại và kiểm tra kết quả, (4) biết chỗ nào dễ sai và vì sao thiết kế như vậy.
>
> **Nguyên tắc tài liệu:** mọi con số đều lấy từ artifact thật trong `outputs/results/`
> hoặc từ README/AGENTS của repo. Không có số liệu minh họa.
>
> **Phiên bản tài liệu:** 25/10/2026. Trạng thái repo: Giai đoạn 1–10 hoàn tất.

---

## Mục lục

1. [Cách dùng tài liệu](#1-cách-dùng-tài-liệu)
2. [Bài toán và sản phẩm cuối](#2-bài-toán-và-sản-phẩm-cuối)
3. [Bức tranh lớn: pipeline 10 bước](#3-bức-tranh-lớn-pipeline-10-bước)
4. [Kiến thức nền tối thiểu](#4-kiến-thức-nền-tối-thiểu)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Cấu trúc repo và vai trò từng file](#6-cấu-trúc-repo-và-vai-trò-từng-file)
7. [Giai đoạn 1 — Tải, kiểm tra, EDA dữ liệu](#7-giai-đoạn-1--tải-kiểm-tra-eda-dữ-liệu)
8. [Giai đoạn 2 — Cache, chia tập, chuẩn hóa, augmentation](#8-giai-đoạn-2--cache-chia-tập-chuẩn-hóa-augmentation)
9. [Hệ thống cấu hình YAML](#9-hệ-thống-cấu-hình-yaml)
10. [Mô hình CNN](#10-mô-hình-cnn)
11. [Baseline — mốc so sánh](#11-baseline--mốc-so-sánh)
12. [Huấn luyện](#12-huấn-luyện)
13. [Đánh giá val/test](#13-đánh-giá-valtest)
14. [Phân tích lỗi, ngưỡng từ chối, OOD](#14-phân-tích-lỗi-ngưỡng-từ-chối-ood)
15. [Suy luận và demo Streamlit](#15-suy-luận-và-demo-streamlit)
16. [Tổng hợp kết quả và multi-seed](#16-tổng-hợp-kết-quả-và-multi-seed)
17. [Kiểm thử và tái lập](#17-kiểm-thử-và-tái-lập)
18. [Đọc kết quả thật của repo](#18-đọc-kết-quả-thật-của-repo)
19. [Runbook chạy từ A đến Z](#19-runbook-chạy-từ-a-đến-z)
20. [Câu hỏi thường gặp và bẫy](#20-câu-hỏi-thường-gặp-và-bẫy)
21. [Kịch bản thuyết trình 10 phút](#21-kịch-bản-thuyết-trình-10-phút)
22. [Phụ lục: schema, đường dẫn, bản đồ code](#22-phụ-lục-schema-đường-dẫn-bản-đồ-code)

---

## 1. Cách dùng tài liệu

### 1.1. Ba cách đọc

| Bạn đang cần | Đọc phần |
|---|---|
| Hiểu tổng quan để bảo vệ bài | 2, 3, 18, 21 |
| Hiểu code để sửa/chạy lại | 4 → 17, 19 |
| Debug khi có lỗi | 20, 22 |
| Kiểm tra số liệu trước khi báo cáo | 18, 22 |

### 1.2. Quy ước trong tài liệu

Mỗi bước kỹ thuật được trình bày theo 4 mục:

- **Khái niệm:** thuật ngữ đó nghĩa là gì, vì sao cần.
- **Code làm gì:** hàm/file nào, thao tác theo thứ tự nào.
- **Kết quả:** file nào được sinh ra, con số nào kiểm tra được.
- **Tự kiểm tra:** lệnh hoặc câu hỏi để chắc bạn đã hiểu.

Ký hiệu đường dẫn code dạng `src/data.py:1167` = file `src/data.py`, dòng 1167.
Số dòng có thể lệch nếu file bị sửa; khi đó tìm theo tên hàm.

### 1.3. Điều quan trọng nhất cần nhớ

Repo này không chỉ là “một file train CNN”. Nó là một **quy trình nghiên cứu nhỏ**:

```text
dữ liệu có kiểm chứng
  → chia tập không rò rỉ
  → baseline
  → CNN + ablation
  → chọn trên val
  → đóng băng cấu hình
  → test đúng một lần
  → phân tích lỗi
  → demo
  → báo cáo
```

Nếu chỉ nhớ một câu: **mọi con số phải truy được về một file trong `outputs/`, và test chỉ được dùng một lần.**

---

## 2. Bài toán và sản phẩm cuối

### 2.1. Bài toán

Cho một ảnh biển báo giao thông Đức, dự đoán ảnh thuộc một trong **43 lớp**.

- Đầu vào: ảnh RGB, được đưa về **64 × 64**.
- Đầu ra: vector **43 logits**; lớp dự đoán là vị trí có logit lớn nhất.
- Dữ liệu: **GTSRB** (German Traffic Sign Recognition Benchmark).
  - Train: **39.209** ảnh.
  - Test: **12.630** ảnh.
  - 43 lớp, mất cân bằng mạnh: lớp ít nhất có 210 ảnh train, lớp nhiều nhất 2.250 ảnh.

### 2.2. Sản phẩm cuối

| Thành phần | File | Dùng để làm gì |
|---|---|---|
| Mô hình cuối | `checkpoints/final_s42/best.pt` hoặc `outputs/runs/final_s42/best.pt` | Suy luận |
| Báo cáo | `docs/bao_cao/BAO_CAO.md` / `.docx` | Nộp bài |
| Demo | `app.py` | Upload ảnh, xem top-3 |
| Bảng kết quả | `outputs/results/summary.csv` | Nguồn số liệu duy nhất |
| Kết quả test | `outputs/results/test_final.json` | Test đúng một lần |
| Phân tích lỗi | `outputs/results/error_report.json` | Chương 5 báo cáo |
| Hình | `outputs/figures/01…11_*.png` | Báo cáo/slide |

### 2.3. Kết quả cốt lõi (số thật)

| Hạng mục | Kết quả | Nguồn |
|---|---|---|
| Mô hình | CNN-B tự xây, **2.201.931** tham số | `configs/final.yaml` |
| Split cuối | `stratified_group`: **31.371 train / 7.830 val** | `split_stratified_group.json` |
| Val (seed 42) | accuracy **0,983525** · macro-F1 **0,970613** | `summary.csv` |
| Multi-seed val | accuracy **0,980843 ± 0,003052** · macro-F1 **0,965392 ± 0,004689** | `multiseed.json` |
| Test (một lần) | accuracy **0,985590** · macro-F1 **0,979894** · top-3 **0,996437** | `test_final.json` |
| Baseline HOG+SVC | val accuracy **0,965693** · macro-F1 **0,973994** | `summary.csv` |
| Lỗi đáng chú ý | lớp 27 (Người đi bộ) F1 **0,846154**; cặp nhầm 27→11 **15 ảnh** | `error_report.json` |
| Ngưỡng từ chối | τ = **0,9461585**, giữ **94,99 %** ảnh val | `threshold.json` |

> **Cảnh báo đọc số:** val **random** ở giai đoạn ablation đạt gần 99,9 % vì bị **rò rỉ frame**
> (95,22 % ảnh val có frame train cách ±1 trong cùng track). Kết luận cuối phải dùng
> **val stratified_group** và **test**. Đây là bài học phương pháp luận quan trọng nhất của repo.

---

## 3. Bức tranh lớn: pipeline 10 bước

### 3.1. Sơ đồ

```text
┌──────────────────────────────┐
│ 1. Tải & xác minh dữ liệu    │  src/data.py
│    39.209 train / 12.630 test│
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 2. EDA + phát hiện trùng lặp │  data_report.json, train_meta.csv
│    loại 8 ảnh train trùng test│  còn 39.201 ảnh để chia
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 3. Cache 64×64 uint8 (.npy)  │  data/cache/
│ 4. Chia train/val (JSON)     │  split_random/group/stratified_group
│ 5. Tính mean/std trên train  │  data_stats.json
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 6. Baseline                  │  majority, HOG + LinearSVC
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 7. Train CNN + ablation      │  E1, E2, E3, E4, E5, E7
│    chọn checkpoint theo val  │  outputs/runs/<run_id>/
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 8. Chọn mô hình trên val sạch│  cand_e3 / cand_e4
│    đóng băng final.yaml      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 9. Test đúng một lần         │  test_final.json + predictions.csv
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│10. Phân tích lỗi + demo      │  error_report.json, app.py
└──────────────────────────────┘
```

### 3.2. Vì sao phải chia thành nhiều giai đoạn?

Vì mỗi giai đoạn trả lời một câu hỏi khác nhau:

| Câu hỏi | Giai đoạn | Không được lẫn với |
|---|---|---|
| Dữ liệu có đúng không? | 1–2 | Chọn mô hình |
| Mô hình học được chưa? | 3 | Đánh giá cuối |
| Có tốt hơn baseline không? | 4–5 | Test |
| Mô hình nào tốt nhất? | 6 | Test |
| Chất lượng thật là bao nhiêu? | 6 | Chọn lại mô hình |
| Nó sai ở đâu, vì sao? | 7 | Tune trên test |

---

## 4. Kiến thức nền tối thiểu

Phần này đủ để đọc code. Nếu đã biết, bỏ qua.

### 4.1. Ảnh trong máy tính

- Ảnh màu là ma trận `H × W × 3` (cao, rộng, 3 kênh đỏ-xanh lá-xanh dương).
- Mỗi điểm ảnh (pixel) là số **0–255** (`uint8`).
- PyTorch dùng tensor `C × H × W` (kênh trước), nên phải `permute(2, 0, 1)`.
- Chuẩn hóa: `(x/255 - mean) / std` để các kênh có phân bố gần 0, giúp train ổn định.

### 4.2. Tensor, batch, epoch

| Khái niệm | Nghĩa |
|---|---|
| Tensor | Mảng nhiều chiều, chạy được trên GPU |
| Batch | Nhóm ảnh đưa vào model một lần, ví dụ 128 ảnh |
| Epoch | Một lượt đi qua toàn bộ tập train |
| Step/iteration | Một lần cập nhật trọng số trên một batch |
| Logits | Điểm thô 43 lớp, chưa qua softmax |
| Softmax | Biến logits thành xác suất cộng lại bằng 1 |

### 4.3. CNN

**Tích chập (convolution)** là một bộ lọc nhỏ (3×3) trượt khắp ảnh để tìm đặc trưng: cạnh, góc, màu, hình tròn, chữ số.

Một khối conv trong repo:

```text
Conv2d(3×3, padding=1) → BatchNorm → ReLU → MaxPool(2)
```

- `padding=1`: giữ nguyên kích thước sau conv.
- `BatchNorm`: chuẩn hóa đầu ra từng kênh, giúp học nhanh và ổn định.
- `ReLU`: bỏ giá trị âm, tạo phi tuyến.
- `MaxPool(2)`: lấy max trong ô 2×2, giảm một nửa chiều cao/rộng.

Với 3 lần pool: `64 → 32 → 16 → 8`.

### 4.4. Overfitting và regularization

- **Overfitting:** model học thuộc train, điểm train cao nhưng val thấp.
- **Dropout:** tắt ngẫu nhiên một phần neuron khi train, buộc model không phụ thuộc một nơ-ron.
- **Weight decay:** phạt trọng số quá lớn.
- **Augmentation:** tạo ảnh biến thể nhẹ từ ảnh train.

Ba thứ này có trong CNN-B. CNN-A là mốc “chưa tối ưu” để so sánh.

### 4.5. Hàm mất mát và tối ưu

- **CrossEntropyLoss:** đo khoảng cách giữa phân bố dự đoán và nhãn đúng.
- **AdamW:** bộ tối ưu cập nhật trọng số theo gradient, có weight decay tách riêng.
- **Learning rate (lr):** độ lớn bước cập nhật. Quá lớn → phân kỳ; quá nhỏ → chậm.
- **Cosine scheduler:** giảm lr theo đường cosin từ đầu đến cuối train.
- **AMP (Automatic Mixed Precision):** dùng float16 cho một số phép toán trên GPU để nhanh và ít VRAM hơn, vẫn giữ float32 cho chỗ nhạy cảm.

### 4.6. Metric

Với 43 lớp:

- **Accuracy** = số dự đoán đúng / tổng số ảnh.
- **Precision** của lớp c: trong các ảnh model nói là c, bao nhiêu phần đúng.
- **Recall** của lớp c: trong các ảnh thật là c, model bắt được bao nhiêu.
- **F1** = trung bình điều hòa của precision và recall.
- **Macro-F1** = trung bình F1 của **43 lớp**, mỗi lớp trọng số như nhau.
- **Confusion matrix** = bảng 43×43: hàng là nhãn thật, cột là nhãn dự đoán.

Vì sao dùng macro-F1? Vì accuracy có thể đẹp dù model bỏ rơi lớp hiếm. Macro-F1 phạt nặng lớp hiếm dự đoán kém.

Repo luôn tính macro-F1 trên **đủ 43 lớp**:

```python
f1_score(y_true, y_pred, average="macro", labels=list(range(43)), zero_division=0)
```

Nếu val thiếu lớp, lớp đó F1 = 0 và bị tính vào trung bình, không bị sklearn lặng lẽ bỏ qua.

### 4.7. Chia tập và rò rỉ dữ liệu

- **Train:** học.
- **Validation (val):** chọn mô hình, chọn epoch, chọn ngưỡng.
- **Test:** chỉ dùng **một lần** để báo cáo chất lượng cuối.

**Rò rỉ (leakage)** xảy ra khi thông tin của test/val lọt vào train. Ví dụ GTSRB có nhiều frame liên tiếp của cùng một video/track. Nếu chia ngẫu nhiên theo ảnh, frame 1 vào train và frame 2 vào val → model đã “thấy” gần như cùng ảnh. Val sẽ cao giả tạo.

Cần phân biệt rõ: **1.306 là số track bị chồng, không phải 1.306 ảnh trùng**. `overlap_index = 0` nghĩa là không có ảnh nào nằm ở cả hai tập; nhưng **7.466/7.841 ảnh val (95,22 %)** có frame train cách ±1 trong cùng track. Vì vậy val random đạt ~99,9 % là do nhận ra lại track, không phải tổng quát hóa.

Repo xử lý bằng split theo **track** (nhóm frame cùng nguồn), chi tiết ở mục 8.2.

### 4.8. Seed và tái lập

Seed là số khởi tạo cho random. Cùng seed + cùng code + cùng dữ liệu + cùng phiên bản thư viện → kết quả gần như/bằng nhau.

Repo seed Python, NumPy, PyTorch, CUDA, DataLoader; bật chế độ deterministic. Đây là lý do `phase3_determinism.json` có `max_abs_diff = 0.0`.

### 4.9. Vài cú pháp Python/PyTorch cần biết

| Cú pháp | Nghĩa |
|---|---|
| `with open(...) as fh:` | Mở file, tự đóng khi xong |
| `Path(...) / "x"` | Ghép đường dẫn an toàn |
| `dict.get("k", default)` | Lấy khóa, thiếu thì trả mặc định |
| `@dataclass` | Tự sinh `__init__` cho lớp dữ liệu |
| `@torch.no_grad()` | Không tính gradient, dùng khi đánh giá |
| `model.train()` / `model.eval()` | Bật/tắt dropout và BatchNorm ở chế độ train |
| `tensor.to(device)` | Chuyển tensor sang GPU/CPU |
| `logits.argmax(dim=1)` | Chọn lớp có điểm cao nhất |
| `np.bincount(y, minlength=43)` | Đếm số ảnh mỗi lớp |
| `json.dump(..., ensure_ascii=False)` | Ghi JSON giữ tiếng Việt |

---

## 5. Cài đặt môi trường

### 5.1. Khái niệm

Môi trường ảo (`.venv`) là một Python riêng, cài thư viện riêng, không làm hỏng Python hệ thống.

PyTorch phải cài từ index **cu128** vì GPU RTX 5070 là Blackwell, compute capability `(12, 0)`. Cài từ PyPI mặc định dễ ra bản CPU hoặc CUDA cũ.

### 5.2. Lệnh

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

### 5.3. Kết quả mong đợi

```text
2.11.0+cu128 True (12, 0)
```

Nếu ra `False` hoặc `(12, 0)` không đúng, dừng lại, cài lại torch trước khi làm bước khác.

### 5.4. Chỉ muốn demo, không train

```powershell
pip install -r requirements-demo.txt
streamlit run app.py
```

Checkpoint demo có sẵn ở `checkpoints/final_s42/best.pt`.

### 5.5. Tự kiểm tra

- `torch.cuda.is_available()` là `True`?
- `torch.cuda.get_device_capability()` là `(12, 0)`?
- `.venv` đã activate? Prompt có `(.venv)`.

---

## 6. Cấu trúc repo và vai trò từng file

### 6.1. Cây thư mục rút gọn

```text
GTSRB-CNN/
├── AGENTS.md                 quy tắc làm việc, hằng số, điều cấm
├── README.md                 tổng quan + kết quả + lệnh chạy
├── app.py                    demo Streamlit
├── requirements.txt          thư viện đầy đủ
├── requirements-demo.txt     chỉ thư viện suy luận
├── configs/
│   ├── default.yaml          cấu hình nền
│   ├── e1…e7.yaml            từng thí nghiệm
│   ├── final.yaml            cấu hình đã đóng băng
│   └── class_names.json      43 lớp: en, vi, shape, color
├── src/
│   ├── config.py             hằng số, YAML, seed, device, env
│   ├── data.py               tải, EDA, cache, split, preprocess, loader
│   ├── models.py             SmallCNN A/B/C
│   ├── baseline.py            majority + HOG/LinearSVC
│   ├── train.py               vòng train + early stopping + artifact
│   ├── evaluate.py            metric, confusion, predictions, test một lần
│   ├── error_analysis.py      per-class, top cặp nhầm, tertile, τ, OOD
│   ├── predict.py             load model, top-k, từ chối unknown
│   └── summarize.py           summary.csv + multi-seed
├── tests/
│   ├── test_pipeline.py       11 pytest
│   ├── golden_test.py          so xác suất giữa 2 môi trường
│   ├── e2e_app.py             test UI Playwright
│   ├── export_demo_images.py  xuất ảnh demo
│   └── package_demo.py        đóng gói bundle
├── outputs/
│   ├── runs/<run_id>/         artifact từng lần train (không commit)
│   ├── figures/               hình 01…11
│   └── results/               JSON/CSV tổng hợp (commit)
├── external_test/             ảnh ngoài dataset + manifest.csv
├── checkpoints/final_s42/     checkpoint demo được commit
└── docs/                      kế hoạch, báo cáo, hướng dẫn
```

### 6.2. Luồng import

```text
config.py  ←── data.py ←── train.py ←── evaluate.py
   ↑             ↑            ↑             ↑
models.py ───────┘            │             │
   ↑                          │             │
baseline.py ──────────────────┘             │
                                             │
predict.py ←── error_analysis.py ────────────┘
   ↑
app.py
```

- `config.py` không import module nội bộ nào khác → nền tảng.
- `data.py` import `config.py`.
- `train.py` import `config`, `data`, `models`.
- `evaluate.py` import `config`, `data`, `models`.
- `predict.py` import `config`, `data.preprocess`, `models`.
- `app.py` chỉ import `src.predict` + `src.config`; **không tự viết lại preprocess**.

### 6.3. Tự kiểm tra

Mở `src/data.py`, tìm `def preprocess`. Hỏi: app.py có gọi hàm này không?  
Đáp án: có, qua `src.predict`, và `predict.py:103` gọi `preprocess(...)`.

---

## 7. Giai đoạn 1 — Tải, kiểm tra, EDA dữ liệu

Mục tiêu: chắc chắn dữ liệu đúng **trước khi** train. Nếu dữ liệu sai, mọi con số sau đó vô nghĩa.

### 7.1. Khái niệm: EDA

EDA (Exploratory Data Analysis) = khám phá dữ liệu trước khi mô hình hóa:

- Có bao nhiêu ảnh?
- Mỗi lớp bao nhiêu ảnh?
- Kích thước ảnh có đồng nhất không?
- Có ảnh hỏng, ảnh trùng, nhãn sai không?
- Có bao nhiêu track/video?

### 7.2. Code làm gì

Hàm `download_gtsrb` — `src/data.py:119`:

1. Dùng `torchvision.datasets.GTSRB(split="test", download=True)` cho test (12.630 ảnh).
2. Gọi `_ensure_official_train()` — `src/data.py:159`.
   - Lý do: torchvision tải train từ `GTSRB-Training_fixed.zip`, bản rút gọn **26.640 ảnh**, không khớp công bố 39.209.
   - Repo tải archive chính thức `GTSRB_Final_Training_Images.zip` (MD5 `f33fd80ac59bff73c82d25ab499e03a3`), giải nén, thay thư mục train của torchvision, kiểm tra đủ 39.209.
3. Kiểm tra sự tồn tại của thư mục ảnh train/test và `GT-final_test.csv`.

Sau đó `run_report` — `src/data.py:1561`:

1. `index_train_images`: duyệt từng thư mục lớp `<class>/`, lấy `(path, label)`.
2. `read_test_annotations`: đọc `GT-final_test.csv` (phân tách `;`).
3. `scan_split`: mở từng ảnh, `verify()`, lấy width/height/mode, MD5, track id, ROI.
4. `count_duplicates`: đếm trùng MD5 trong train, trong test, giữa train-test.
5. `summarize_tracks`: thống kê track.
6. Ghi `train_meta.csv`, `excluded_train_duplicates.csv`, vẽ 3 hình EDA, ghi `data_report.json`.

### 7.3. Các khái niệm con

**MD5** là “dấu vân tay” của file. Hai file cùng MD5 gần như chắc chắn giống hệt từng byte. MD5 chỉ bắt trùng tuyệt đối, không bắt gần trùng.

**ROI** (Region of Interest) là hình chữ nhật quanh biển báo trong ảnh gốc, cho trong annotation: `Roi.X1, Roi.Y1, Roi.X2, Roi.Y2`.

**Track ID** là phần số đầu tiên trong tên file train, ví dụ `00023_00012.ppm` → track `00023`. Nhiều frame liên tiếp của cùng một track là các ảnh gần giống nhau.

### 7.3.1. Ảnh GTSRB thực chất là gì?

Cần phân biệt **3 tầng**:

```text
Video gốc ngoài đường
   → chọn/cắt frame
Ảnh GTSRB: frame đã crop quanh biển (vẫn có nền)
   → project xử lý
Ảnh model nhìn: resize 64×64, không crop ROI thêm
```

- GTSRB được xây từ **video quay đường phố Đức**; mỗi file `.ppm` là một **frame đã trích và crop quanh biển báo**.
- Đây **không phải** frame video đầy đủ 1920×1080, cũng **không phải** ảnh chỉ còn đúng hình biển: vẫn có nền (trời, đường, cột, xe...).
- Kích thước nhỏ và biến thiên: trung vị **43×43**, min **25×25**, max **266×232** — nếu là frame đầy đủ thì phải lớn hơn nhiều.
- **ROI** là hộp bao quanh biển bên trong ảnh; ROI sát biển hơn vùng ảnh đã crop.
- **Track** là một lần biển xuất hiện liên tiếp trong video, gồm ~30 frame gần giống nhau. Tên file train `<track>_<frame>.ppm`; test chỉ có tên số nên **không có track ID**.

Trong project này:

- Model nhận **toàn bộ ảnh GTSRB** (đã crop quanh biển), convert RGB, resize 64×64; **không crop thêm theo ROI**.
- ROI chỉ dùng cho EDA, lưới ảnh mẫu, `roi_area_ratio` và phân tích lỗi.
- Đây là **phân loại**, không phải phát hiện: model giả định ảnh đã chứa biển và chỉ dự đoán lớp 0–42. E8 (crop ROI) chỉ là hướng mở rộng, chưa làm.

Hệ quả: các frame cùng track gần như trùng nhau; chia ngẫu nhiên theo ảnh sẽ đưa "anh em sinh đôi" vào cả train và val, gây rò rỉ frame.

### 7.4. Kết quả thật

Từ `outputs/results/data_report.json`:

| Thông tin | Giá trị |
|---|---|
| Train | 39.209 |
| Test | 12.630 |
| Lớp | 43 |
| Train sau khi loại trùng | **39.201** |
| Ảnh train bị loại | **8** |
| Lý do loại | 8 ảnh train trùng MD5 với 8 ảnh test, đều nhãn 14 (Stop) |

8 ảnh này bị đánh dấu `usable_for_train = 0` trong `train_meta.csv`; file ảnh vẫn còn trên đĩa để số đếm khớp công bố 39.209.

Artifact:

```text
outputs/results/data_report.json
outputs/results/train_meta.csv
outputs/results/excluded_train_duplicates.csv
outputs/figures/01_class_distribution.png
outputs/figures/02_image_sizes.png
outputs/figures/03_sample_grid.png
```

### 7.5. Vì sao phải loại 8 ảnh trùng?

Nếu ảnh test xuất hiện trong train, model có thể “học thuộc” ảnh đó. Test sẽ không còn độc lập. Dù chỉ 8/12.630 ảnh, nguyên tắc là nguyên tắc: test phải sạch.

### 7.6. Tự kiểm tra

Mở `data_report.json`, xem:

- `counts.train` = 39209?
- `counts.train_pool_after_exclusion` = 39201?
- `counts.excluded_duplicates` = 8?

Lệnh chạy lại (không tải lại nếu đã có dữ liệu):

```powershell
python -m src.data --report
python -m src.data --report --limit 200   # smoke test nhanh
```

---

## 8. Giai đoạn 2 — Cache, chia tập, chuẩn hóa, augmentation

Đây là phần quan trọng nhất về mặt phương pháp. Nếu hiểu sai phần này, mọi kết quả đều đáng ngờ.

### 8.1. Cache ảnh — `src/data.py:870`

**Khái niệm:** đọc và resize 39.201 + 12.630 ảnh mỗi epoch rất chậm. Cache là đọc một lần, lưu mảng `uint8` đã resize.

**Code làm gì:**

```text
với mỗi split (train, test):
  tạo mảng rỗng (N, 64, 64, 3) uint8
  với từng ảnh:
    mở ảnh → convert RGB → resize 64×64 bilinear → ghi vào mảng
  lưu X_*.npy, y_*.npy, roi_*.npy, paths_*.txt
```

**Kết quả:**

```text
data/cache/X_train.npy   (39201, 64, 64, 3) uint8  ← pool để chia split
data/cache/y_train.npy   (39201,)
data/cache/X_test.npy    (12630, 64, 64, 3) uint8
data/cache/cache_meta.json
```

**Lưu ý cực quan trọng:** `X_train.npy` là **pool 39.201 ảnh**, chưa phải tập train 31.371. Tập train/val thật là chỉ số trong `split_*.json`. Đừng nói “train có 39.201 ảnh”.

**Không crop ROI:** cache dùng **toàn bộ ảnh GTSRB** (đã crop quanh biển) resize 64×64; ROI chỉ lưu riêng trong `roi_*.npy` để phân tích, không dùng làm input.

**Vì sao `uint8`?** Ảnh gốc 0–255. Lưu `uint8` tiết kiệm 4 lần so với `float32`. Khi đưa vào model mới đổi sang float và chuẩn hóa.

### 8.2. Ba cách chia tập — `src/data.py:954`

#### (a) `random`

Dùng `train_test_split(test_size=0.2, stratify=labels, random_state=42)`.

- Kết quả: **31.360 train / 7.841 val**.
- Ưu: giữ tỷ lệ lớp, đơn giản.
- Nhược: **rò rỉ frame**. Track có thể xuất hiện ở cả train và val.
- Thực tế: **1.306 track** bị chồng giữa train/val, nhưng **0 ảnh trùng index**; 7.466/7.841 ảnh val (95,22 %) có frame train cách ±1.
- Vì vậy val random đạt ~99,9 % và **không dùng để kết luận cuối**.

#### (b) `group`

Dùng `GroupShuffleSplit` theo `track_key`.

- Kết quả: **31.341 train / 7.860 val**.
- Track train/val rời nhau hoàn toàn (`overlap_track = 0`).
- Nhược: có lớp **không có mặt ở val**: 0, 19, 34, 39.
- Vì thiếu lớp, macro-F1 43 lớp bị kéo xuống và trần lý thuyết < 1.

#### (c) `stratified_group` — split cuối

Đây là split chính của Giai đoạn 6. Code — `src/data.py:980`:

```text
với từng lớp c:
  lấy danh sách track của lớp c
  k = max(1, round(0.2 × số track))
  chọn ngẫu nhiên k track (seed cố định) cho val
  các track còn lại cho train
```

- Track không bao giờ nằm ở cả hai tập.
- Mỗi lớp có ít nhất 1 track ở val (vì `max(1, ...)`).
- Kết quả: **31.371 train / 7.830 val**, **43/43 lớp ở val**, `overlap_track = 0`.

Bảng so sánh:

| Split | Train | Val | Đủ 43 lớp ở val | Track chồng | Dùng để |
|---|---:|---:|---|---:|---|
| `random` | 31.360 | 7.841 | Có | 1.306 | Ablation Giai đoạn 5 |
| `group` | 31.341 | 7.860 | Không (thiếu 0,19,34,39) | 0 | Chẩn đoán rò rỉ |
| `stratified_group` | 31.371 | 7.830 | Có | 0 | **Chọn & báo cáo cuối** |

**Bài học:** “val sạch” và “val đủ lớp” đôi khi xung đột. `stratified_group` là giải pháp dung hòa.

### 8.3. Mean/std — `src/data.py:1104`

**Khái niệm:** chuẩn hóa ảnh:

```text
x_norm = (x / 255 - mean) / std
```

`mean`/`std` tính trên **train**, không tính trên val/test (tránh rò rỉ thống kê).

Code tính theo từng kênh, dùng tổng và tổng bình phương để không phải nạp toàn bộ vào RAM:

```text
mean_c = sum(x_c) / N_pixel
var_c  = sum(x_c²) / N_pixel - mean_c²
std_c  = sqrt(var_c)
```

Kết quả thật — `outputs/results/data_stats.json`:

```json
{
  "mean": [0.340408, 0.312365, 0.321632],
  "std":  [0.275148, 0.264242, 0.270747],
  "num_images": 31360,
  "split": "random"
}
```

**Nói đúng nguồn:** mean/std được tính **một lần** trên train của `split_random` (31.360 ảnh) rồi đóng băng, dùng cho mọi split và cho demo. Đây là hằng số tiền xử lý toàn cục 3 số, không chứa nhãn; nhưng khi giải thích phải nói đúng là “tính trên train random”, không nói “tính trên train stratified_group”.

### 8.4. Hàm `preprocess` duy nhất — `src/data.py:1167`

Đây là “hợp đồng” quan trọng nhất giữa train và demo.

**Không crop ROI:** input là **toàn ảnh GTSRB đã crop quanh biển**; `preprocess` không nhận ROI và không cắt thêm. ROI chỉ dùng cho EDA/phân tích lỗi.

**Code làm gì:**

```text
input: PIL / ndarray / đường dẫn
  → convert RGB
  → resize 64×64 bilinear
  → uint8
  → float32 / 255
  → HWC → CHW
  → (x - mean) / std
output: tensor (3, 64, 64)
```

**Vì sao chỉ một hàm?** Nếu train dùng resize khác demo, model nhận ảnh khác phân bố dù test 99 %. Demo sẽ sai khó hiểu. Repo cấm viết lại preprocess trong `app.py`/notebook.

**Nơi dùng:**

| Nơi | Cách dùng |
|---|---|
| Train eval (không aug) | `_array_to_tensor` trong dataset |
| Evaluate | `GTSRBTensorDataset` với `transform=None` |
| Demo | `predict_topk` gọi `preprocess` |
| Golden test | `Predictor.predict` |

### 8.5. Augmentation — `src/data.py:1190`

**Khái niệm:** tạo ảnh biến thể để model bớt học thuộc.

Các phép trong repo:

| Phép | Ý nghĩa | Tham số |
|---|---|---|
| `RandomAffine` | xoay nhẹ, dịch, zoom, shear | rotation 10°, translate 0.1, scale 0.9–1.1, shear 5° |
| `ColorJitter` | đổi sáng/tương phản/bão hòa/hue nhẹ | 0.2 / 0.2 / 0.2 / 0.02 |
| `RandomPerspective` | biến dạng phối cảnh nhẹ | 0.1, p=0.3 |
| `ToTensor` | PIL → tensor [0,1] | — |
| `Normalize` | trừ mean, chia std | từ config |

**Cấm tuyệt đối:**

- `HorizontalFlip`: lật ngang biển có thể đổi nghĩa (ví dụ mũi tên trái ↔ phải).
- `VerticalFlip`: biển lộn ngược, vô nghĩa.
- `RandomRotation` 90/180: đổi hướng biển.

Repo có hàm `_assert_no_flip` — `src/data.py:1233` chặn nếu transform chứa `Flip` hoặc `RandomRotation`.

**Chỉ áp dụng cho train.** Val/test/demo không augmentation.

### 8.6. Dataset và DataLoader — `src/data.py:1244`, `:1282`

`GTSRBTensorDataset`:

```python
__getitem__(vị trí):
    idx = indices[vị trí]
    ảnh = cache_x[idx]          # uint8 64×64×3
    nếu có transform: tensor = transform(PIL(ảnh))   # train, có aug
    nếu không:        tensor = _array_to_tensor(...)  # eval, chỉ normalize
    return tensor, label
```

`make_loaders`:

- Train: `shuffle=True`, có augmentation.
- Val: `shuffle=False`, không augmentation.
- `num_workers=4` trên Windows phải có `if __name__ == "__main__"` (repo đã đảm bảo entrypoint).
- `worker_init_fn=seed_worker` để mỗi worker có seed xác định.
- `pin_memory=True` khi có CUDA.

Trả về `meta` gồm split, seed, số ảnh, batch size, transform, mean/std để ghi log/artifact.

### 8.7. Checklist Giai đoạn 2 — `src/data.py:1431`

`--verify` kiểm tra 12 mục, ví dụ:

- Tổng train + val = 39.201?
- Giao train/val = 0?
- Đủ 43 lớp ở train và val?
- Lệch tỷ lệ lớp ≤ 0,01?
- Split group có track chồng = 0?
- Cache đúng shape `(39201, 64, 64, 3)` uint8?
- Preprocess ra `(3, 64, 64)`, giá trị hữu hạn, trong biên lý thuyết?
- Augmentation không chứa Flip/RandomRotation?
- Đọc lại split cho cùng chỉ số?
- Import `preprocess` từ ngoài chạy được?

Kết quả: `outputs/results/phase2_check.json`.

### 8.8. Tự kiểm tra

1. Vì sao `X_train.npy` có 39.201 ảnh nhưng train chỉ 31.371?
2. Vì sao không dùng `random` để chốt kết quả?
3. Vì sao không được lật ảnh biển?
4. Chạy `python -m src.data --verify`, xem bao nhiêu PASS.

---

## 9. Hệ thống cấu hình YAML

### 9.1. Khái niệm

Không hard-code hyperparameter trong code. Mọi thứ nằm trong YAML để:

- Tái lập dễ.
- So sánh công bằng.
- Ghi lại chính xác mỗi run đã dùng gì.

### 9.2. Hai lớp cấu hình

```text
configs/default.yaml          nền
        ↓ deep-merge
configs/e3_cnnb_aug.yaml      chỉ ghi phần khác
        ↓ validate
dict cấu hình hoàn chỉnh
```

`load_config` — `src/config.py:118`:

1. Đọc `default.yaml`.
2. Nếu có file thí nghiệm → `deep_merge`.
3. Nếu có `overrides` (ví dụ seed) → merge tiếp.
4. `_fill_stats` điền `mean`/`std` từ `data_stats.json` nếu YAML để `null`.
5. `validate_config` kiểm tra kiểu/giá trị.

### 9.3. Deep-merge

`deep_merge` — `src/config.py:107` merge đệ quy dict. Ví dụ:

```yaml
# default
model: {name: cnn_b, widths: [32,64,128], dropout: 0.3}
# e1
model: {name: cnn_a, use_bn: false, dropout: 0.0}
# kết quả
model: {name: cnn_a, widths: [32,64,128], dropout: 0.0, use_bn: false}
```

`widths` không bị mất vì e1 không khai báo.

### 9.4. Validate — `src/config.py:171`

Các kiểm tra tiêu biểu:

- Đủ 5 nhóm: `experiment`, `data`, `model`, `train`, `output`.
- `num_classes == 43`.
- `image_size > 0`.
- `split` thuộc `random|group|stratified_group`.
- `val_ratio` trong (0,1).
- `mean/std` là list 3 số, `std > 0`.
- `model.name` thuộc `cnn_a|cnn_b|cnn_c`.
- `monitor == "val_macro_f1"`.
- `lr`, `batch_size`, `max_epochs` > 0.
- `patience >= 1`.

Nếu sai, raise `ValueError` với nguồn file. Không “âm thầm sửa”.

### 9.5. Các file config quan trọng

| File | Vai trò |
|---|---|
| `default.yaml` | Nền: 64×64, 43 lớp, CNN-B, AdamW, 40 epoch, patience 7 |
| `smoke.yaml` | Chạy cực nhanh để test code |
| `e1_cnna.yaml` | CNN-A, không BN/dropout/WD/aug |
| `e2_cnnb.yaml` | Thêm BN + dropout + WD |
| `e3_cnnb_aug.yaml` | Thêm augmentation |
| `e4_cnnc_gap_aug.yaml` | CNN-C 4 khối + GAP |
| `e5a/e5b` | Thử learning rate thấp/cao |
| `e7_classweight.yaml` | Class weight `1/sqrt(freq)` |
| `final.yaml` | Đóng băng: E3 + `stratified_group` |

### 9.6. Tự kiểm tra

Mở `configs/e1_cnna.yaml` và `configs/e2_cnnb.yaml`. Liệt kê đúng **một nhóm yếu tố** thay đổi giữa hai file.  
Đáp án: regularization (BN + dropout + weight decay), không đổi dữ liệu/kiến trúc cơ bản.

---

## 10. Mô hình CNN

### 10.1. Khái niệm

`SmallCNN` — `src/models.py:26` — là một CNN nhỏ, tự xây, không pretrained. Ba biến thể dùng chung class, khác tham số.

### 10.2. Kiến trúc tổng quát

Với `widths = (32, 64, 128)`, `in_size = 64`:

```text
Input: (B, 3, 64, 64)
  Block 1: Conv3×3(3→32) → BN → ReLU → MaxPool2   → (B, 32, 32, 32)
  Block 2: Conv3×3(32→64) → BN → ReLU → MaxPool2  → (B, 64, 16, 16)
  Block 3: Conv3×3(64→128) → BN → ReLU → MaxPool2 → (B, 128, 8, 8)
  Flatten: 128 × 8 × 8 = 8192
  Dropout(0.3)
  Linear(8192 → 256)
  ReLU
  Dropout(0.3)
  Linear(256 → 43)
Output: (B, 43) logits
```

Nếu `gap=True` (CNN-C):

```text
Block 4: Conv(128→256) → ... → (B, 256, 4, 4)
GlobalAvgPool: trung bình theo không gian → (B, 256)
Dropout → Linear(256 → 43)
```

GAP giảm mạnh tham số vì bỏ lớp fully-connected lớn.

### 10.3. Vì sao `in_size` phải chia hết cho `2^len(widths)`?

Mỗi MaxPool chia 2. Với 3 khối, 64 phải chia hết 8. Code kiểm tra — `src/models.py:44`:

```python
if in_size % (2 ** len(widths)) != 0:
    raise ValueError(...)
```

Nếu không, phép tính `flattened = previous * (in_size // 2**len(widths))**2` sai shape.

### 10.4. Ba biến thể

| Tên | widths | BN | Dropout | GAP | Vai trò | Params |
|---|---|---|---:|---|---|---:|
| CNN-A | 32/64/128 | Không | 0.0 | Không | Mốc “chưa tối ưu” | 2.201.707 |
| CNN-B | 32/64/128 | Có | 0.3 | Không | Mô hình chính | **2.201.931** |
| CNN-C | 32/64/128/256 | Có | 0.4 | Có | Sâu hơn, ít tham số | 399.947 |

CNN-A và CNN-B gần bằng tham số vì BN chỉ thêm buffer nhỏ, không thêm nhiều trọng số học. Khác biệt chính là **cách học**, không phải kích thước.

### 10.5. Logits, không softmax

`forward` trả logits thô — `src/models.py:80`. Vì `CrossEntropyLoss` của PyTorch tự làm softmax + log bên trong. Nếu model cũng softmax sẽ **softmax hai lần**, sai.

Khi suy luận mới gọi `torch.softmax(logits, dim=1)` để lấy xác suất.

### 10.6. Đếm tham số — `src/models.py:90`

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Chỉ đếm tham số học được. Buffer BN (`running_mean`, `running_var`) không tính.

### 10.7. Smoke forward — `src/models.py:120`

Tạo batch giả `(2, 3, 64, 64)`, chạy forward, kiểm tra output `(2, 43)`. Không cần dữ liệu, không cần GPU. Dùng để test kiến trúc nhanh.

### 10.8. Tự kiểm tra

1. Với input 64×64 và 3 khối, sau flatten có bao nhiêu đặc trưng?  
   Đáp án: `128 × 8 × 8 = 8192`.
2. Vì sao CNN-C ít tham số hơn CNN-B dù sâu hơn?  
   Đáp án: GAP bỏ lớp FC 8192→256, thay bằng 256→43.
3. Vì sao không softmax trong model?  
   Đáp án: CrossEntropyLoss đã bao gồm softmax.

---

## 11. Baseline — mốc so sánh

### 11.1. Khái niệm

Baseline là mô hình đơn giản, không học sâu, để trả lời: “CNN có thực sự cần thiết không?” Nếu CNN chỉ hơn baseline 0,1 %, cần suy nghĩ lại.

### 11.2. E0a — Majority class — `src/baseline.py:98`

**Luật:** dự đoán mọi ảnh là lớp nhiều nhất trong train.

Code:

```python
counts = np.bincount(y_train, minlength=43)
majority = counts.argmax()
predictions = np.full(len(y_val), majority)
```

**Kết quả val:**

| Metric | Giá trị |
|---|---:|
| Accuracy | 0,057391 |
| Macro-F1 | 0,002524 |

Vì lớp 2 (50 km/h) chiếm khoảng 5,74 % val, đoán tất cả là lớp 2 → accuracy 5,74 %. Macro-F1 gần 0 vì 42 lớp còn lại F1 = 0.

**Ý nghĩa:** đây là “sàn”. Mọi mô hình thật phải vượt xa.

### 11.3. E0b — HOG + LinearSVC — `src/baseline.py:152`

**HOG** (Histogram of Oriented Gradients) là đặc trưng thủ công mô tả hướng cạnh trong ảnh. Ảnh xám 64×64, cấu hình mặc định cho **1.764 chiều**.

Pipeline:

```text
ảnh RGB uint8
  → xám [0,1]
  → HOG 1.764 chiều
  → StandardScaler (fit trên train)
  → LinearSVC
```

Code thử `C ∈ {0.1, 1.0, 10.0}`, chọn `C` có **val macro-F1** cao nhất. Không chạm test.

**Kết quả val:**

| Metric | Giá trị |
|---|---:|
| Accuracy | 0,965693 |
| Macro-F1 | 0,973994 |
| Params (coef + intercept) | 75.895 |
| Thời gian train | 1.926,43 s |
| Suy luận | 0,0069 ms/ảnh |

**Nhận xét:** HOG + SVC rất mạnh trên GTSRB, nhưng CNN-B vẫn cao hơn trên val sạch (0,9835 / 0,9706). Đáng chú ý: macro-F1 của HOG (0,9740) cao hơn accuracy (0,9657) — chuyện có thể xảy ra khi hiệu năng giữa các lớp tương đối đồng đều; không phải lỗi.

### 11.4. Vì sao baseline cũng chỉ đánh giá trên val?

Vì test chỉ được dùng một lần cho mô hình cuối. Nếu dùng test để chọn `C` cho baseline, test không còn độc lập.

### 11.5. Tự kiểm tra

1. Vì sao majority có macro-F1 gần 0 nhưng accuracy 5,7 %?
2. HOG chọn `C` bằng gì?  
   Đáp án: val macro-F1.
3. Baseline có augmentation không?  
   Đáp án: không.

---

## 12. Huấn luyện

### 12.1. Khái niệm

Train = lặp qua dữ liệu, tính loss, lan truyền ngược gradient, cập nhật trọng số.

### 12.2. Các thành phần — `src/train.py`

#### Loss — `build_criterion`, dòng 74

Mặc định: `CrossEntropyLoss`, có thể thêm:

- `class_weight="inv_sqrt_freq"`: trọng số lớp hiếm = `1/√(tần suất)`, chuẩn hóa trung bình = 1. Dùng cho E7.
- `label_smoothing`: làm mềm nhãn, chống tự tin quá mức.

#### Optimizer — `build_optimizer`, dòng 91

- `adamw`: mặc định.
- `adam`.
- `sgd` với momentum 0.9.

Tất cả đọc `lr`, `weight_decay` từ YAML.

#### Scheduler — `build_scheduler`, dòng 106

- `cosine`: `CosineAnnealingLR(T_max=max_epochs)`.
- `none`: không dùng.

#### AMP

```python
use_amp = cfg["train"]["amp"] and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
```

Chỉ bật trên CUDA. Trên CPU tự tắt.

### 12.3. Một epoch train — `train_one_epoch`, dòng 115

```text
model.train()
for inputs, targets in loader:
    chuyển lên device
    optimizer.zero_grad()
    với autocast (nếu AMP):
        logits = model(inputs)
        loss = criterion(logits, targets)
    nếu AMP:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    nếu không:
        loss.backward()
        optimizer.step()
    cộng dồn loss, đếm đúng
trả loss trung bình, accuracy
```

Bốn bước cốt lõi, lặp đi lặp lại:

1. Xóa gradient cũ.
2. Forward + loss.
3. Backward.
4. Cập nhật trọng số.

### 12.4. Đánh giá val trong train — `evaluate`, dòng 150

```text
@torch.no_grad()
model.eval()
kiểm tra model.training == False
kiểm tra gradient đã tắt
for batch:
    logits = model(...)
    loss
    pred = argmax
tính accuracy
tính macro-F1 trên 43 lớp
```

Hai kiểm tra `model.training` và `torch.is_grad_enabled()` là “chốt an toàn”: nếu ai quên `eval()` hoặc `no_grad()`, chương trình báo lỗi thay vì cho kết quả sai âm thầm.

### 12.5. Vòng train đầy đủ — `run_training`, dòng 259

Trình tự:

1. Xác định run id `<experiment.id>_s<seed>`.
2. Nếu thư mục run đã tồn tại → **raise**, không ghi đè.
3. Seed.
4. Tạo DataLoader.
5. Tạo model, đếm params.
6. Tạo loss, optimizer, scheduler, scaler.
7. Ghi `config.yaml`, `env.json`.
8. Mở `metrics.csv`, ghi header, flush.
9. Lặp epoch:
   - đo thời gian, reset bộ đếm VRAM;
   - train một epoch;
   - scheduler step;
   - eval val;
   - ghi một dòng CSV, **flush ngay**;
   - nếu `val_macro_f1` tốt hơn → lưu `best.pt`, reset `bad_epochs`;
   - nếu không tốt hơn → tăng `bad_epochs`; đủ `patience=7` → early stop.
10. Vẽ `curves.png` (loss, accuracy, macro-F1).
11. Ghi `meta.json`.
12. Trả meta.

### 12.6. Vì sao chọn checkpoint theo val macro-F1?

Accuracy có thể cao nhờ lớp nhiều ảnh. Macro-F1 buộc mô hình làm tốt cả lớp hiếm. Repo đặt `MONITOR = "val_macro_f1"` và validate config bắt buộc.

### 12.7. Early stopping

Nếu 7 epoch liên tiếp không cải thiện val macro-F1, dừng. Mục đích: tiết kiệm thời gian, tránh overfit.

Lưu ý: `final_s42` chạy đủ 40 epoch, best epoch 34, 6 epoch không cải thiện < 7 → không kích hoạt early stop.

### 12.8. Artifact mỗi run

```text
outputs/runs/<run_id>/
├── config.yaml      cấu hình đã dùng
├── env.json         Python, torch, CUDA, GPU, git commit
├── metrics.csv      1 dòng/epoch
├── best.pt          checkpoint tốt nhất theo val macro-F1
├── curves.png       3 biểu đồ
└── meta.json        tổng kết run
```

Schema `metrics.csv`:

```text
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

### 12.9. Checkpoint chứa gì?

`_save_checkpoint` — `src/train.py:230`:

```python
{
  "state_dict": ...,
  "config": ...,
  "class_names": ...,
  "epoch": ...,
  "val_macro_f1": ...,
  "git_commit": ...,
  "torch_version": ...
}
```

Không pickle cả object model. Lý do: tránh lỗi khi phiên bản PyTorch thay đổi. Khi load, tạo lại model từ `config` rồi `load_state_dict`.

### 12.10. Hai bài test hạ tầng

#### Overfit một batch — `src/train.py:418`

Lấy 32 ảnh, tắt augmentation, train 50 bước. Kỳ vọng loss giảm gần 0, accuracy = 100 %.

Kết quả thật — `phase3_overfit.json`:

```json
{"loss_first": 3.858838, "loss_last": 0.021139, "accuracy": 1.0, "passed_loss": true, "passed_acc": true}
```

Nếu không overfit nổi 32 ảnh, code/label/optimizer có lỗi.

#### Determinism — `src/train.py:465`

Train 2 lần cùng seed, so từng epoch. Kết quả thật: `max_abs_diff = 0.0`.

### 12.11. Tự kiểm tra

1. Vì sao `metrics.csv` phải flush từng epoch?  
   Đáp án: mất điện/gián đoạn vẫn còn dữ liệu.
2. Vì sao không ghi đè run cũ?  
   Đáp án: bảo toàn bằng chứng, tránh lẫn kết quả.
3. Vì sao checkpoint lưu dict thay vì cả model?  
   Đáp án: an toàn phiên bản, rõ ràng, dễ audit.

---

## 13. Đánh giá val/test

### 13.1. Khái niệm

Đánh giá = nạp checkpoint, chạy trên tập cố định, tính metric, lưu dự đoán chi tiết. Đây là **nơi duy nhất** tính metric cho val/test (train.py có bản riêng để chọn checkpoint, nhưng logic macro-F1 giống).

### 13.2. Nạp checkpoint — `src/evaluate.py:93`

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model, _ = build_model(checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

`weights_only=True` chỉ cho phép tensor/dict cơ bản, giảm rủi ro bảo mật.

### 13.3. Tạo loader đánh giá — `src/evaluate.py:131`

- `val`: lấy chỉ số val từ split, ảnh từ cache train.
- `test`: lấy toàn bộ 12.630 ảnh test.
- Không augmentation.
- Kèm `paths`, `labels`, `roi_area_ratio`, ảnh uint8 để tính chất lượng ảnh.
- Batch 256, `shuffle=False`.

### 13.4. Dự đoán — `src/evaluate.py:211`

```python
probability = torch.softmax(model(inputs).float(), dim=1)
pred = probability.argmax(axis=1)
```

Lưu cả xác suất để tính top-3, margin, ngưỡng.

### 13.5. Metric — `src/evaluate.py:235`

Tính:

- `accuracy`
- `macro_f1_43` (đủ 43 lớp)
- `macro_f1_present_classes` (chỉ lớp có mặt)
- `n_classes_present`, `missing_classes`
- `macro_f1_ceiling_43` = số lớp có mặt / 43
- `per_class`: precision, recall, F1, support cho từng lớp
- `top_confusions`: 10 cặp nhầm nhiều nhất
- `confusion_matrix`: 43×43

Sau đó thêm `top3_accuracy` bằng cách đếm nhãn đúng nằm trong 3 xác suất cao nhất.

### 13.6. Confusion matrix — `src/evaluate.py:276`

Chuẩn hóa **theo hàng**:

```text
ô [i, j] = số ảnh thật lớp i bị dự đoán lớp j / tổng số ảnh thật lớp i
```

Vì chuẩn hóa theo hàng, đường chéo = recall. Hình 43×43, màu càng sáng càng nhiều.

### 13.7. `predictions.csv` — `src/evaluate.py:296`

Mỗi dòng một ảnh, các cột:

```text
path,true_id,true_name,pred_id,pred_name,
prob_true,prob_pred,margin,top3,
blur_laplacian,brightness_mean,roi_area_ratio,is_correct
```

- `margin = prob_pred - xác suất cao thứ hai`.
- `top3` dạng `27:0.99|11:0.004|...`.
- `blur_laplacian`: phương sai Laplacian, cao = nét.
- `brightness_mean`: độ sáng trung bình.
- `roi_area_ratio`: diện tích biển / diện tích ảnh.

### 13.8. Val và test khác nhau thế nào?

#### Val — `evaluate_run`, dòng 342

- Chạy nhiều lần được.
- Lưu `eval_val/metrics.json`, `predictions.csv`, `confusion_matrix.png`.
- Không cần `--confirm-freeze`.

#### Test — `run_test_evaluation`, dòng 420

Ba chốt bảo vệ:

1. Nếu `test_report.json` đã tồn tại → **từ chối**, trừ khi `--allow-rerun`.
2. `--allow-rerun` bắt buộc có `--reason`.
3. `_check_freeze` — dòng 396:
   - `configs/final.yaml` phải tồn tại.
   - checkpoint phải có `experiment.id == "final"`.
   - `model`, `augmentation`, `train`, `data` trong checkpoint phải khớp `final.yaml` (bỏ qua mean/std vì được điền động).

Khi chạy:

- Ghi `test_report.json` và `test_final.json` (nội dung giống nhau).
- Ghi `predictions.csv`, `confusion_matrix.png`.
- Append một dòng vào `test_runs.log`.
- Nếu rerun: sinh bản `_v2`, **không ghi đè** bản gốc.

### 13.9. Kết quả test thật

`outputs/results/test_final.json`:

| Metric | Giá trị |
|---|---:|
| Số ảnh | 12.630 |
| Accuracy | **0,9855898654** |
| Macro-F1 (43) | **0,9798935305** |
| Top-3 accuracy | **0,9964370546** |
| Số lớp có mặt | 43 |
| Lớp thiếu | 0 |
| Checkpoint SHA-256 | `9c4dd0d6…` |

Xác minh 14/14 mục trong `test_verification.json`: đúng 12.630 dòng, 12.630 path duy nhất, nhãn khớp GT, accuracy/macro-F1 tính lại khớp, confusion tổng hàng = support, log test chỉ 1 dòng.

### 13.10. Vì sao test chỉ một lần?

Mỗi lần xem test và chỉnh gì đó (model, lr, ngưỡng) là một lần “học từ test”. Sau nhiều lần, test không còn là ước lượng khách quan. Repo biến quy tắc này thành code: chặn ghi đè + log + xác minh.

### 13.11. Tự kiểm tra

1. Vì sao `_check_freeze` bỏ qua `mean/std`?  
   Đáp án: chúng được điền từ `data_stats.json`, không phải siêu tham số đóng băng.
2. Confusion matrix chuẩn hóa theo hàng hay cột?  
   Đáp án: hàng.
3. Chạy lại test bình thường có được không?  
   Đáp án: không, trừ khi `--allow-rerun --reason`, và sinh `_v2`.

---

## 14. Phân tích lỗi, ngưỡng từ chối, OOD

### 14.1. Khái niệm

Sau khi biết “đúng bao nhiêu”, câu hỏi tiếp theo là “sai ở đâu, vì sao, có thể tin được không”.

### 14.2. Per-class — `src/error_analysis.py:92`

Đọc `per_class` từ test report, ghi `per_class_test.csv`, vẽ F1 tăng dần, tô đỏ lớp F1 < 0,90.

Kết quả: **1 lớp** dưới 0,90:

| Lớp | Tên | Precision | Recall | F1 | Support |
|---:|---|---:|---:|---:|---:|
| 27 | Người đi bộ | 1,000 | 0,733 | **0,846** | 60 |

Nghĩa: model không đoán nhầm **vào** lớp 27, nhưng bỏ sót 16/60 ảnh người đi bộ.

### 14.3. Top cặp nhầm — `src/error_analysis.py:154`

Lấy 10 cặp `(true, pred)` nhiều nhất, mỗi cặp tối đa 3 ảnh ví dụ, vẽ lưới.

Top 1: **27 → 11** (Người đi bộ → Đường ưu tiên tại giao lộ tới), **15 ảnh**.

Các cặp khác:

| True → Pred | Số ảnh |
|---|---:|
| 2 → 1 (50 → 30 km/h) | 12 |
| 8 → 5 (120 → 80 km/h) | 11 |
| 22 → 25 (Gồ ghề → Công trường) | 10 |
| 12 → 13 (Đường ưu tiên → Nhường đường) | 10 |

### 14.4. Tertile — `src/error_analysis.py:191`

Chia mỗi biến thành 3 nhóm theo phân vị 33/67:

- độ mờ (`blur_laplacian`, cao = nét),
- độ sáng (`brightness_mean`),
- diện tích biển / ảnh (`roi_area_ratio`).

Tính error rate từng nhóm, lưu `error_rates_tertiles.csv`, vẽ hình.

Kết quả effect size — `latency_effect.json`:

| Biến | Nhóm tệ nhất | Nhóm tốt nhất | Chênh điểm % | CI95 | r² |
|---|---|---|---:|---|---:|
| Độ mờ | 0 | 1 | 1,185 | [0,647; 1,722] | 0,00027 |
| Độ sáng | 0 | 2 | 0,879 | [0,392; 1,366] | 0,00121 |
| Diện tích biển | 0 | 2 | 1,490 | [0,965; 2,014] | 0,00429 |

**Cách đọc đúng:** p-value có thể rất nhỏ vì n = 12.630, nhưng r² rất nhỏ. Nghĩa là các biến này chỉ giải thích một phần rất nhỏ khả năng đúng/sai. Không được nói “ảnh mờ là nguyên nhân chính” chỉ vì p < 0,05.

### 14.5. Mẫu lỗi — `src/error_analysis.py:255`

Chọn 12 ảnh sai, ưu tiên confidence cao, chia 4 nhóm:

| Nhóm | Điều kiện |
|---|---|
| E-A mờ | blur tertile thấp nhất |
| E-B tối | brightness tertile thấp nhất |
| E-C biển nhỏ | ROI ratio tertile thấp nhất |
| E-E lớp giống nhau | còn lại |

Lấy round-robin để mẫu đa dạng. Vẽ `07_error_samples.png` + `error_samples_table.csv`.

### 14.6. Ngưỡng từ chối τ — `src/error_analysis.py:342`

**Khái niệm:** nếu model không chắc, trả “không xác định” thay vì đoán bừa.

Cách chọn trong repo:

```text
τ = phân vị 5 % của max-prob trên val
reject nếu max-prob < τ
```

Kết quả:

```json
{
  "tau": 0.9461585,
  "percentile": 5.0,
  "acceptance_in_distribution": 0.9499361430395913,
  "n_val": 7830,
  "chosen_on": "val stratified_group (final_s42), KHÔNG dùng test/demo"
}
```

Nghĩa: giữ khoảng 95 % ảnh giống phân bố train, từ chối 5 % kém tự tin nhất.

**Cảnh báo:** τ là ngưỡng confidence, **không phải bộ phát hiện OOD**. Ảnh không có biển vẫn có thể được dự đoán với confidence ~1,00.

### 14.7. OOD — `src/error_analysis.py:358`

OOD = out-of-distribution, ảnh ngoài phân bố GTSRB (biển nước khác, cảnh không biển).

Repo dùng `external_test/manifest.csv` gồm 11 ảnh:

- 1 ảnh GTSRB tham chiếu (không tính OOD),
- 10 ảnh OOD: 2 biển tốc độ nước ngoài, 4 biển dạng chữ, 4 cảnh không biển.

Kết quả — `ood_summary.json`:

| Nhóm | n | Bị từ chối | Tỷ lệ |
|---|---:|---:|---:|
| Biển tốc độ | 2 | 1 | 50 % |
| Biển chữ | 4 | 2 | 50 % |
| Không có biển | 4 | 3 | 75 % |

Vẫn có 1/4 ảnh không biển bị nhận nhầm với confidence **0,99595**. Vì vậy không được quảng cáo τ là “bộ lọc OOD”.

### 14.8. Tự kiểm tra

1. Lớp nào yếu nhất? F1 bao nhiêu?  
   Đáp án: lớp 27, 0,846154.
2. Cặp nhầm nhiều nhất?  
   Đáp án: 27 → 11, 15 ảnh.
3. τ chọn trên tập nào?  
   Đáp án: val, phân vị 5 %.
4. τ có phát hiện được mọi ảnh không biển?  
   Đáp án: không.

---

## 15. Suy luận và demo Streamlit

### 15.1. Nạp model — `src/predict.py:31`

```python
checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
model, _ = build_model(checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Trả `(model, config, class_names, device)`.

### 15.2. Dự đoán top-k — `src/predict.py:92`

```text
preprocess(ảnh, size, mean, std) → (3,64,64)
unsqueeze(0) → (1,3,64,64)
model → logits
softmax → 43 xác suất
argsort giảm dần → top-k
```

Trả:

```python
{
  "topk": [{"class_id", "name_vi", "name_en", "prob"}, ...],
  "probabilities": [...],
  "max_prob": ...,
  "margin": top1 - top2,
  "tau": ...,
  "unknown": max_prob < tau
}
```

### 15.3. Kiểm tra ảnh upload — `src/predict.py:68`

`validate_image_bytes`:

- Kích thước ≤ 20 MB.
- Đuôi `.jpg/.jpeg/.png`.
- `Image.verify()` + `load()` để phát hiện file hỏng.
- Convert RGB.

Nếu sai → `ValueError` thân thiện, app hiển thị lỗi thay vì crash.

### 15.4. App Streamlit — `app.py`

Luồng:

```text
main()
  → kiểm tra checkpoint tồn tại
  → kiểm tra threshold.json tồn tại
  → load τ
  → get_predictor (cache_resource, nạp 1 lần)
  → sidebar: test acc, macro-F1, top-3, τ, multi-seed
  → tab "Dự đoán":
       upload ảnh hoặc chọn ảnh mẫu
       validate
       predictor.predict(image, k=3, tau)
       render top-3 + cảnh báo unknown
  → tab "Về mô hình":
       kiến trúc, split, test, giải thích vì sao biển Việt Nam không phân loại được
```

Điểm cần nói khi bảo vệ:

- App **không viết lại preprocess**; gọi `src.data.preprocess`.
- Ngưỡng τ lấy từ file, không chỉnh theo ảnh demo.
- Có cảnh báo “không dùng cho quyết định giao thông thực tế”.
- Ảnh mẫu nằm trong `external_test/` và `data/gtsrb/...`, không commit ảnh dữ liệu.

### 15.5. Đo thời gian — `latency_effect.json`

| Phạm vi | Median | P95 | Max |
|---|---:|---:|---:|
| Model-only | 3,016 ms | 3,662 ms | 4,992 ms |
| End-to-end Python | 5,780 ms | 7,798 ms | 12,207 ms |

End-to-end Python gồm: đọc bytes → validate → preprocess → forward → top-3. **Chưa** gồm upload mạng, Streamlit rerun, render trình duyệt.

Khi báo cáo, phải nói rõ phạm vi. Không gọi 3 ms là “thời gian phản hồi app”.

### 15.6. Tự kiểm tra

1. Vì sao app không được tự resize ảnh?  
   Đáp án: phải dùng đúng preprocess của train.
2. `unknown` được quyết định bởi gì?  
   Đáp án: `max_prob < tau`.
3. 3,016 ms có phải thời gian người dùng chờ?  
   Đáp án: không, chỉ model-only trên GPU.

---

## 16. Tổng hợp kết quả và multi-seed

### 16.1. `summary.csv` — `src/summarize.py:153`

Quét `outputs/runs/*/`:

- Run CNN: có `meta.json` + `metrics.csv`.
- Baseline: có `metrics.json`.

Bỏ qua run phụ `diag_`, `smoke_`, `cand_` (vẫn giữ artifact, chỉ không đưa vào bảng chính).

Cột:

```text
run_id,model,split,role,params,aug,epochs_run,best_epoch,
val_acc,val_macro_f1,test_acc,test_macro_f1,
train_time_s,infer_ms_per_img,seed,notes,
train_val_acc_gap,best_train_acc
```

Đây là **nguồn số liệu duy nhất** cho bảng trong báo cáo/slide.

### 16.2. Multi-seed — `src/summarize.py:363`

Chạy final với seed 42, 123, 2024. Tính mean/std (`ddof=1`) trên val.

Kết quả:

| Seed | Val acc | Val macro-F1 | Best epoch | Params |
|---:|---:|---:|---:|---:|
| 42 | 0,983525 | 0,970613 | 34 | 2.201.931 |
| 123 | 0,977522 | 0,961541 | 29 | 2.201.931 |
| 2024 | 0,981481 | 0,964021 | 39 | 2.201.931 |
| **Mean ± std** | **0,980843 ± 0,003052** | **0,965392 ± 0,004689** | — | — |

Vì sao cần multi-seed? Một seed có thể may mắn. Báo cáo mean ± std cho thấy độ ổn định.

Lưu ý: multi-seed chỉ trên val; test chỉ chạy một lần cho `final_s42`.

### 16.3. Tự kiểm tra

1. Vì sao `summary.csv` bỏ run `cand_`?  
   Đáp án: run chọn mô hình, không phải kết quả cuối; tránh lẫn bảng.
2. Std tính theo `ddof` nào?  
   Đáp án: 1 (mẫu).
3. Multi-seed có test không?  
   Đáp án: không.

---

## 17. Kiểm thử và tái lập

### 17.1. `pytest` — `tests/test_pipeline.py`

11 test, không cần GPU, không train:

| Test | Kiểm tra |
|---|---|
| `test_preprocess_shape_range` | preprocess ra (3,64,64), giá trị hợp lệ |
| `test_split_random_disjoint` | train/val random không giao, tổng 39.201 |
| `test_split_stratified_group_tracks_disjoint_and_all_classes` | track rời nhau, val đủ 43 lớp |
| `test_macro_f1_counts_all_43_classes` | macro-F1 tính đủ 43 lớp |
| `test_checkpoint_loads_and_forward_43` | checkpoint đủ khóa, forward (2,43) |
| `test_validate_image_rejects_bad_inputs` | chặn file hỏng/sai đuôi/quá lớn |
| `test_predictions_and_report_consistent` | predictions khớp report |
| `test_test_evaluated_once` | log test đúng 1 dòng |
| `test_threshold_chosen_on_val` | τ chọn trên val |
| `test_manifest_matches_images` | manifest khớp 11 ảnh |
| `test_class_names_cover_43` | đủ 43 lớp, có vi/en |

Chạy:

```powershell
python -m pytest tests -q
```

Kỳ vọng: `11 passed`.

### 17.2. Golden test — `tests/golden_test.py`

Mục đích: chuyển sang máy khác, kiểm tra model cho cùng xác suất.

```powershell
python tests/golden_test.py dump --out outputs/results/golden_venv.json
python tests/golden_test.py dump --out outputs/results/golden_venv_test.json
python tests/golden_test.py compare outputs/results/golden_venv.json outputs/results/golden_venv_test.json
```

Điều kiện đạt: lệch xác suất lớn nhất < 1e-3, không lệch nhãn. Kết quả trong repo:

```json
{"max_probability_difference": 0.0, "prediction_mismatches": 0, "passed": true}
```

### 17.3. E2E UI — `tests/e2e_app.py`

Dùng Playwright + Chromium, chạy 9 kịch bản giao diện: mở app, upload ảnh đúng, ảnh sai, ảnh quá lớn, chọn mẫu, tab about, thiếu checkpoint…

```powershell
python -m playwright install chromium
python tests/e2e_app.py
```

### 17.4. Clean-room — `outputs/results/cleanroom.json`

Tạo `.venv_test` mới, cài torch cu128 + requirements từ đầu, chạy smoke train, eval val, app, pytest.

Kết quả:

```json
{
  "torch_version": "2.11.0+cu128",
  "cuda_available": true,
  "capability": "(12, 0)",
  "gpu": "NVIDIA GeForce RTX 5070",
  "val_acc": 0.9835249042145594,
  "val_macro_f1": 0.970612678826378,
  "expected_val_acc": 0.983525,
  "pytest_tail": "11 passed in 3.33s",
  "elapsed_min": 1.01
}
```

Điểm mạnh: chứng minh không phụ thuộc “máy cũ còn file rác”.

### 17.5. `env.json`

Mỗi run ghi:

- Python version, implementation.
- Torch, torchvision.
- CUDA build, cuDNN.
- GPU name, capability, VRAM.
- Driver.
- OS, CPU, RAM.
- Git commit.

Đây là bằng chứng tái lập.

### 17.6. Tự kiểm tra

1. Vì sao test không cần GPU?  
   Đáp án: test đọc artifact/CPU, trừ khi bạn chủ động chạy train.
2. Golden test so cái gì?  
   Đáp án: xác suất top-3 của 20 ảnh val cố định.
3. Clean-room khác gì máy hiện tại?  
   Đáp án: venv mới hoàn toàn, cài từ requirements.

---

## 18. Đọc kết quả thật của repo

### 18.1. Bảng ablation trên val random

Từ `summary.csv`:

| Run | Model | Aug | Params | Val acc | Val macro-F1 | Ghi chú |
|---|---|---|---:|---:|---:|---|
| E0a majority | majority | no | 0 | 0,057391 | 0,002524 | sàn |
| E0b HOG+SVC | HOG+SVC | no | 75.895 | 0,965693 | 0,973994 | baseline mạnh |
| E1 | CNN-A | no | 2.201.707 | 0,996301 | 0,995705 | chưa regularization |
| E2 | CNN-B | no | 2.201.931 | 0,998342 | 0,998392 | thêm BN/dropout/WD |
| E3 | CNN-B | yes | 2.201.931 | 0,999362 | 0,999471 | thêm aug |
| E4 | CNN-C | yes | 399.947 | 0,999745 | 0,999763 | GAP, ít tham số |
| E5a | CNN-B | yes | 2.201.931 | 0,989542 | 0,993387 | lr 3e-4, 10 epoch |
| E5b | CNN-B | yes | 2.201.931 | 0,771968 | 0,567247 | lr 3e-3, phân kỳ |
| E7 | CNN-B | yes | 2.201.931 | 0,998470 | 0,998914 | class weight |

> **Đọc bảng này đúng cách:** đây là val **random**, bị rò rỉ frame. Các số ~99,9 % **không dùng để chốt mô hình**. Chúng chỉ dùng để so xu hướng ablation trong cùng điều kiện.

### 18.2. Chọn mô hình trên val sạch

Từ `final.yaml`:

| Ứng viên | Val acc | Val macro-F1 | Params |
|---|---:|---:|---:|
| `cand_e3_sgroup_s42` | 0,983525 | 0,970613 | 2.201.931 |
| `cand_e4_sgroup_s42` | 0,976373 | 0,954453 | 399.947 |

E3 hơn E4 **1,62 điểm %** macro-F1 (> 0,3 điểm %). Quy tắc “ưu tiên ít tham số” chỉ áp dụng khi chênh < 0,3 điểm %. Vì vậy chọn **E3**.

### 18.3. Test cuối

| Metric | Giá trị |
|---|---:|
| Accuracy | **0,985590** |
| Macro-F1 | **0,979894** |
| Top-3 | **0,996437** |
| Số ảnh | 12.630 |

### 18.4. Vì sao val random 99,9 % mà val sạch 98,35 %?

Vì rò rỉ frame. Khi track bị chia đôi, val chứa frame gần giống train. Model nhận ra “track” thay vì học tổng quát. Khi track tách hoàn toàn, điểm giảm về mức thật.

Đây không phải “model kém đi”, mà là **đo đúng hơn**.

### 18.5. Cách trình bày số liệu

Nên nói:

> “Trên val random, các mô hình đạt ~99,9 % nhưng val này bị rò rỉ frame. Trên val stratified_group sạch, E3 đạt 98,35 % accuracy và 97,06 % macro-F1. Test một lần đạt 98,56 % accuracy, 97,99 % macro-F1, top-3 99,64 %.”

Không nên nói:

> “Model đạt 99,9 %.”

### 18.6. Tự kiểm tra

1. Vì sao E4 thấp hơn E3 dù val random cao hơn?  
   Đáp án: val random bị rò rỉ; trên val sạch E4 thấp hơn.
2. Số nào là kết quả cuối để báo cáo?  
   Đáp án: test_final.json (acc 0,985590, macro-F1 0,979894).
3. Multi-seed dùng để làm gì?  
   Đáp án: đo độ ổn định trên val, không thay test.

---

## 19. Runbook chạy từ A đến Z

### 19.1. Chuẩn bị

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

### 19.2. Dữ liệu

```powershell
python -m src.data --download --report
python -m src.data --prepare
python -m src.data --verify
```

Kỳ vọng:

```text
data_report.json: train 39.209, test 12.630, loại 8 trùng
cache: X_train (39201,64,64,3), X_test (12630,64,64,3)
split_random 31360/7841, split_stratified_group 31371/7830
phase2_check.json: 12/12 PASS
```

### 19.3. Baseline

```powershell
python -m src.baseline --config configs/default.yaml
python -m src.baseline --config configs/default.yaml --verify
```

Kỳ vọng:

```text
outputs/runs/e0a_majority/metrics.json
outputs/runs/e0b_hog_svc/metrics.json + model.joblib
phase4_check.json: PASS
```

### 19.4. Smoke test trước khi train thật

```powershell
python -m src.train --config configs/smoke.yaml
```

Mục đích: chắc pipeline chạy, không quan tâm điểm.

### 19.5. Train các thí nghiệm

```powershell
python -m src.train --config configs/e1_cnna.yaml --seed 42
python -m src.train --config configs/e2_cnnb.yaml --seed 42
python -m src.train --config configs/e3_cnnb_aug.yaml --seed 42
python -m src.train --config configs/e4_cnnc_gap_aug.yaml --seed 42
```

### 19.6. Train final 3 seed

```powershell
python -m src.train --config configs/final.yaml --seed 42
python -m src.train --config configs/final.yaml --seed 123
python -m src.train --config configs/final.yaml --seed 2024
python -m src.summarize --multiseed
```

### 19.7. Đánh giá val

```powershell
python -m src.evaluate --run outputs/runs/final_s42 --split val
```

Chạy nhiều lần được. Kỳ vọng: acc 0,983525, macro-F1 0,970613.

### 19.8. Đánh giá test — một lần

```powershell
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze
```

Kỳ vọng: acc 0,985590, macro-F1 0,979894, top-3 0,996437.

Nếu file test report đã tồn tại, lệnh sẽ từ chối. Chỉ dùng khi thực sự cần:

```powershell
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze `
  --allow-rerun --reason "sửa lỗi X, lý do Y"
```

### 19.9. Tổng hợp + phân tích lỗi

```powershell
python -m src.summarize --figure
python -m src.error_analysis --run outputs/runs/final_s42
```

### 19.10. Demo

```powershell
streamlit run app.py
```

Mở `http://localhost:8501`.

### 19.11. Kiểm thử

```powershell
python -m pytest tests -q
python tests/e2e_app.py
```

### 19.12. Thứ tự đọc code đề xuất

1. `src/config.py` — hằng số, load/validate.
2. `src/models.py` — kiến trúc.
3. `src/data.py` — từ `preprocess`, `make_loaders`, rồi mới đọc phần split/cache.
4. `src/train.py` — `train_one_epoch`, `evaluate`, `run_training`.
5. `src/evaluate.py` — `classification_metrics`, `run_test_evaluation`.
6. `src/predict.py` — suy luận.
7. `app.py` — UI.
8. `src/error_analysis.py` — phân tích.
9. `src/summarize.py` — tổng hợp.
10. `tests/test_pipeline.py` — kiểm chứng.

---

## 20. Câu hỏi thường gặp và bẫy

### 20.1. Vì sao ảnh 64×64?

- GTSRB gốc 15×15 đến 250×250; 64×64 đủ giữ biển, nhẹ tính toán.
- 64 chia hết cho 2³ = 8, vừa 3 lần pool.
- CNN-B 2,2 triệu tham số, train ~6 phút trên RTX 5070.

### 20.2. Vì sao CNN-B có 2,2 triệu tham số mà vẫn nhỏ?

Vì lớp FC 8192→256 chiếm phần lớn: `8192 × 256 ≈ 2,1 triệu`. CNN-C dùng GAP nên chỉ 400 nghìn.

### 20.3. Vì sao train nhanh nhưng vẫn chạy 40 epoch?

Mỗi epoch ~5–10 giây trên GPU; 40 epoch ~6 phút. Early stopping và cosine scheduler cần đủ epoch để hội tụ.

### 20.4. Vì sao không dùng pretrained?

Đề bài yêu cầu mô hình tự xây dựng. Pretrained (ResNet, EfficientNet) sẽ làm mất tính “tự xây” và dễ đạt điểm cao không công bằng với mục tiêu học.

### 20.5. Vì sao test chỉ 1 lần mà vẫn có `test_final.json`?

Vì repo đã chạy test đúng 1 lần và lưu artifact. Khi bạn đọc file, không phải chạy lại.

### 20.6. Vì sao `predictions.csv` có 12.630 dòng?

Test có 12.630 ảnh, mỗi ảnh một dòng, không trùng path.

### 20.7. Vì sao app trả “không xác định” cho biển Việt Nam?

Vì GTSRB chỉ có 43 lớp biển Đức. Biển Việt Nam không có nhãn tương ứng. Đây là **domain shift**, không phải lỗi app.

### 20.8. Bẫy thường gặp

| Bẫy | Hậu quả | Cách tránh |
|---|---|---|
| Quên `model.eval()` | Val sai, dropout/BN hoạt động sai | Luôn eval + no_grad |
| Preprocess demo khác train | Demo sai dù test cao | Chỉ dùng `src.data.preprocess` |
| Chia val random khi có frame | Val cao giả | Dùng track/group hoặc stratified_group |
| Train xong mới xem test rồi sửa | Test không còn độc lập | Đóng băng config trước test |
| Chọn checkpoint theo accuracy | Lớp hiếm bị bỏ | Theo val macro-F1 |
| `num_workers > 0` trên Windows thiếu `__main__` | Treo/crash | Dùng entrypoint chuẩn |
| Ghi log cuối run | Mất dữ liệu khi gián đoạn | Flush từng epoch |
| Cài torch từ PyPI mặc định | Ra bản CPU | Index cu128 |
| Copy `.venv` sang máy khác | Lỗi đường dẫn/Python | Cài lại từ requirements |
| Dùng test để chọn τ | Rò rỉ test | Chọn trên val |

### 20.9. Lỗi thường gặp và cách xử lý

| Thông báo | Nguyên nhân | Xử lý |
|---|---|---|
| `Chưa có ...data_stats.json` | Chưa `--prepare` | `python -m src.data --prepare` |
| `Thiếu data/cache/X_train.npy` | Chưa cache | `python -m src.data --prepare` |
| `Run đã tồn tại` | Trùng run id/seed | Đổi `experiment.id` hoặc seed |
| `Checkpoint khác configs/final.yaml` | Sửa config sau khi train | Train lại hoặc khôi phục config |
| `test_report.json đã tồn tại` | Đã test rồi | Không chạy lại; nếu bắt buộc dùng `--allow-rerun --reason` |
| `torch.cuda.is_available() = False` | Cài torch CPU | Cài lại cu128 |
| `CUDA capability` sai | Torch cũ | Nâng torch cu128 |

---

## 21. Kịch bản thuyết trình 10 phút

### 21.1. Mở đầu (1 phút)

> “Em làm bài toán phân loại 43 lớp biển báo Đức GTSRB bằng CNN tự xây. Điểm em muốn nhấn mạnh không chỉ là độ chính xác, mà là quy trình: dữ liệu sạch, chia tập không rò rỉ, chọn trên val, test một lần, và phân tích lỗi.”

### 21.2. Dữ liệu (1,5 phút)

- 39.209 train / 12.630 test / 43 lớp.
- Phát hiện 8 ảnh train trùng MD5 với test → loại, còn 39.201 để chia.
- Cache 64×64.
- Ba split: random (rò rỉ), group (thiếu lớp), stratified_group (sạch + đủ lớp).

### 21.3. Mô hình (1,5 phút)

- CNN-B: 3 khối conv 32/64/128 + BN + ReLU + MaxPool, FC 8192→256→43.
- 2.201.931 tham số.
- So sánh A/B/C và baseline.
- Chọn checkpoint theo val macro-F1.

### 21.4. Kết quả (2 phút)

- Baseline majority 5,7 %; HOG+SVC 96,6 %.
- Val random ~99,9 % nhưng rò rỉ.
- Val sạch 98,35 % accuracy, 97,06 % macro-F1.
- Test một lần: 98,56 % accuracy, 97,99 % macro-F1, top-3 99,64 %.
- Multi-seed 98,08 % ± 0,31 điểm %.

### 21.5. Phân tích lỗi (2 phút)

- 1 lớp F1 < 0,90: lớp 27 Người đi bộ (0,846).
- Cặp nhầm 27→11 (15 ảnh): biển tam giác đỏ, hình người nhỏ dễ giống biển ưu tiên.
- Ảnh mờ/tối/biển nhỏ có tương quan yếu (r² rất nhỏ), không phải nguyên nhân chính.
- τ = 0,9462 chọn trên val; vẫn có 1/4 ảnh không biển bị nhận nhầm confidence cao → τ không phải OOD detector.

### 21.6. Kết luận (1 phút)

- CNN tự xây vượt baseline, đạt ~98,6 % test.
- Điểm mạnh: quy trình kiểm chứng, không tune test.
- Hạn chế: chỉ 43 lớp Đức, domain shift, τ chỉ là confidence.
- Không dùng cho quyết định giao thông thực tế.

### 21.7. Ba câu hỏi phòng thủ thường gặp

**Hỏi: Vì sao val random 99,9 % mà test 98,56 %?**  
Đáp: Val random bị rò rỉ frame; test sạch và khó hơn. Con số đúng để báo cáo là val stratified_group và test.

**Hỏi: Vì sao chọn E3 mà không phải E4 ít tham số hơn?**  
Đáp: Trên val sạch, E3 hơn E4 1,62 điểm % macro-F1, vượt ngưỡng 0,3 điểm %. Quy tắc ưu tiên ít tham số chỉ áp dụng khi chênh nhỏ.

**Hỏi: Ngưỡng τ có lọc được ảnh không phải biển báo không?**  
Đáp: Không hoàn toàn. Đó là ngưỡng confidence chọn trên val; OOD test cho thấy 3/4 ảnh không biển bị từ chối, nhưng 1 ảnh vẫn lọt với confidence 0,99595.

---

## 22. Phụ lục: schema, đường dẫn, bản đồ code

### 22.1. Schema `metrics.csv`

```text
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

- 1 dòng/epoch.
- Ghi và flush ngay.
- `gpu_mem_mb` = 0 nếu chạy CPU.

### 22.2. Schema checkpoint `best.pt`

```python
{
    "state_dict": {...},
    "config": {...},
    "class_names": {...},
    "epoch": int,
    "val_macro_f1": float,
    "git_commit": str,
    "torch_version": str,
}
```

Load:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

### 22.3. Schema `predictions.csv`

```text
path,true_id,true_name,pred_id,pred_name,
prob_true,prob_pred,margin,top3,
blur_laplacian,brightness_mean,roi_area_ratio,is_correct
```

### 22.4. Schema `summary.csv`

```text
run_id,model,split,role,params,aug,epochs_run,best_epoch,
val_acc,val_macro_f1,test_acc,test_macro_f1,
train_time_s,infer_ms_per_img,seed,notes,
train_val_acc_gap,best_train_acc
```

### 22.5. Cây artifact

```text
outputs/
├── runs/<run_id>/
│   ├── config.yaml
│   ├── env.json
│   ├── metrics.csv
│   ├── best.pt
│   ├── curves.png
│   ├── meta.json
│   ├── eval_val/
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── confusion_matrix.png
│   └── eval_test/
│       ├── metrics.json
│       ├── predictions.csv
│       └── confusion_matrix.png
├── figures/
│   ├── 01_class_distribution.png
│   ├── 02_image_sizes.png
│   ├── 03_sample_grid.png
│   ├── 04_augmentation.png
│   ├── 05_experiment_comparison.png
│   ├── 07_*.png
│   ├── 10_*.png
│   └── 11_ui_*.png
└── results/
    ├── data_report.json
    ├── data_stats.json
    ├── train_meta.csv
    ├── excluded_train_duplicates.csv
    ├── split_random.json
    ├── split_group.json
    ├── split_stratified_group.json
    ├── summary.csv
    ├── multiseed.json
    ├── test_report.json
    ├── test_final.json
    ├── test_runs.log
    ├── test_verification.json
    ├── predictions_test_final.csv
    ├── threshold.json
    ├── error_report.json
    ├── per_class_test.csv
    ├── top_confusions_test.csv
    ├── error_rates_tertiles.csv
    ├── error_samples_table.csv
    ├── ood_results.csv
    ├── ood_summary.json
    ├── latency_effect.json
    ├── leak_delta.json
    ├── cleanroom.json
    └── phase*_check.json
```

### 22.6. Bản đồ code theo chức năng

| Chức năng | Hàm | Dòng |
|---|---|---|
| Hằng số, đường dẫn | `config.py` | 34–87 |
| Đọc YAML | `load_yaml` | 95 |
| Merge cấu hình | `deep_merge` | 107 |
| Nạp + validate | `load_config` | 118 |
| Validate | `validate_config` | 171 |
| Seed | `set_seed` | 247 |
| Device | `get_device` | 272 |
| Môi trường | `collect_env` | 302 |
| Tải dữ liệu | `download_gtsrb` | 119 |
| Train archive chính thức | `_ensure_official_train` | 159 |
| Quét ảnh | `scan_split` | 303 |
| Trùng MD5 | `count_duplicates` | 347 |
| Báo cáo EDA | `run_report` | 1561 |
| Cache | `build_cache` | 870 |
| Chia tập | `make_splits` | 954 |
| Lưu split | `build_splits` | 1056 |
| Mean/std | `compute_stats` | 1104 |
| Preprocess | `preprocess` | 1167 |
| Augmentation | `build_train_transform` | 1190 |
| Dataset | `GTSRBTensorDataset` | 1244 |
| DataLoader | `make_loaders` | 1282 |
| Checklist GĐ2 | `verify_phase2` | 1431 |
| Model | `SmallCNN` | 26 |
| Forward | `SmallCNN.forward` | 80 |
| Build model | `build_model` | 95 |
| Majority | `run_majority` | 98 |
| HOG+SVC | `run_hog_svc` | 152 |
| Loss | `build_criterion` | 74 |
| Optimizer | `build_optimizer` | 91 |
| Scheduler | `build_scheduler` | 106 |
| Train epoch | `train_one_epoch` | 115 |
| Eval trong train | `evaluate` | 150 |
| Vòng train | `run_training` | 259 |
| Overfit test | `run_overfit_test` | 418 |
| Determinism | `run_determinism_check` | 465 |
| Load checkpoint | `load_checkpoint` (evaluate) | 93 |
| Loader eval | `build_eval_loader` | 131 |
| Predict | `predict_loader` | 211 |
| Metric | `classification_metrics` | 235 |
| Confusion | `plot_confusion` | 276 |
| Predictions CSV | `write_predictions` | 296 |
| Eval val | `evaluate_run` | 342 |
| Freeze check | `_check_freeze` | 396 |
| Eval test | `run_test_evaluation` | 420 |
| Load model demo | `load_model` | 31 |
| Ngưỡng | `load_tau` / `should_reject` | 54 / 63 |
| Validate upload | `validate_image_bytes` | 68 |
| Top-k | `predict_topk` | 92 |
| Predictor | `Predictor` | 132 |
| Tổng hợp | `collect_runs` | 153 |
| Multi-seed | `build_multiseed` | 363 |
| Threshold analysis | `choose_threshold` | 342 |
| OOD | `ood_evaluation` | 358 |
| Error run | `run` | 426 |
| App main | `main` | 193 |

### 22.7. Từ điển nhanh

| Thuật ngữ | Nghĩa ngắn |
|---|---|
| GTSRB | Bộ dữ liệu biển báo Đức, 43 lớp |
| CNN | Mạng nơ-ron tích chập |
| Logits | Điểm thô trước softmax |
| Batch | Nhóm ảnh trong một bước |
| Epoch | Một lượt qua tập train |
| LR | Learning rate |
| BN | Batch Normalization |
| GAP | Global Average Pooling |
| AMP | Mixed precision |
| Macro-F1 | Trung bình F1 43 lớp |
| Confusion matrix | Bảng thật × dự đoán |
| Leakage | Rò rỉ dữ liệu |
| Track | Nhóm frame cùng nguồn |
| ROI | Vùng quanh biển trong ảnh |
| OOD | Ảnh ngoài phân bố |
| τ | Ngưỡng từ chối |
| Ablation | Thí nghiệm bỏ/thêm từng yếu tố |
| Baseline | Mốc so sánh đơn giản |
| Seed | Số khởi tạo ngẫu nhiên |
| Freeze | Đóng băng cấu hình trước test |

### 22.8. Nơi cần xem khi báo cáo

| Bạn cần chứng minh | Mở file |
|---|---|
| Dữ liệu đúng | `outputs/results/data_report.json` |
| Split không rò rỉ | `split_stratified_group.json` |
| Mean/std | `data_stats.json` |
| Baseline | `summary.csv` |
| Kết quả từng run | `outputs/runs/<run_id>/meta.json`, `metrics.csv` |
| Chọn mô hình | `configs/final.yaml` |
| Test một lần | `test_final.json`, `test_runs.log` |
| Xác minh test | `test_verification.json` |
| Lỗi | `error_report.json`, `per_class_test.csv` |
| Ngưỡng | `threshold.json` |
| OOD | `ood_summary.json` |
| Thời gian | `latency_effect.json` |
| Tái lập | `cleanroom.json`, `golden_compare.json` |

---

## Kết luận

Nếu bạn chỉ cần nhớ 5 điều để giải thích repo này:

1. **Dữ liệu được kiểm chứng trước khi train**, kể cả loại 8 ảnh trùng train-test.
2. **Chia tập theo track** để tránh val cao giả; split cuối là `stratified_group` (31.371/7.830, đủ 43 lớp).
3. **Chọn mô hình trên val macro-F1**, không dùng test; test chỉ chạy một lần và bị khóa bằng code.
4. **Kết quả cuối:** CNN-B 2,2 triệu tham số, test accuracy 98,56 %, macro-F1 97,99 %, top-3 99,64 %.
5. **Biết giới hạn:** chỉ 43 lớp biển Đức, τ chỉ là confidence, không dùng cho giao thông thực tế.

Đó là toàn bộ câu chuyện từ A đến Z của mã nguồn này.
