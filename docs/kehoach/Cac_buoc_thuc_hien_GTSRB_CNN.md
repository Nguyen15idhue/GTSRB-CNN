# CÁC BƯỚC THỰC HIỆN — BÀI TẬP LỚN GTSRB + CNN
## Tài liệu thi hành chi tiết (kèm mẫu ghi kết quả và checklist test)

| Thông tin | Giá trị |
|---|---|
| Môn học | Học sâu — 12INF028 |
| Đề tài | Phân loại biển báo giao thông Đức (GTSRB) bằng CNN tự xây dựng |
| Tài liệu gốc | `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md` (bản chủ) |
| Phiên bản | 1.0 — 22/10/2026 |
| Hình thức | 1 người thực hiện toàn bộ, 15 ngày làm việc |
| Mục đích tài liệu | Biến kế hoạch thành **các bước thi hành có thể kiểm tra được**, mỗi bước có test và chỗ ghi kết quả thực tế |

---

## Cách dùng tài liệu này

1. Làm **tuần tự** từ Giai đoạn 0 → 12. Không nhảy gate.
2. Mỗi giai đoạn gồm 7 phần cố định:
   - **Mục tiêu** — làm gì.
   - **Giải thích** — vì sao phải làm, khái niệm liên quan.
   - **Việc cần làm** — các bước nhỏ, kèm lệnh/file.
   - **Yêu cầu cần đạt** — điều kiện tối thiểu để coi là xong.
   - **Checklist test** — tick từng ô, test phải chạy được, không chỉ đọc code.
   - **Kết quả thực hiện** — bảng để trống, điền số liệu thật khi làm.
   - **Ghi chú / lưu ý** — bẫy đã biết, cách tránh.
3. **Qua gate** chỉ khi: toàn bộ checklist test được tick **và** bảng "Kết quả thực hiện" đã điền đủ.
4. Nếu một test fail: ghi vào "Ghi chú", sửa, chạy lại test. Không đánh dấu đạt bằng cách bỏ qua test.

### Quy ước trạng thái

| Ký hiệu | Nghĩa |
|---|---|
| ☐ | Chưa làm |
| ◐ | Đang làm |
| ☑ | Đạt |
| ⚠ | Đạt có điều kiện (ghi rõ điều kiện) |
| ✗ | Chưa đạt — phải làm lại |

### Bảng theo dõi tiến độ tổng

| Giai đoạn | Tên | Ngày dự kiến | Gate | Trạng thái | Bằng chứng (file/hình) |
|---|---|---|---|---|---|
| 0 | Môi trường & công cụ | D1 sáng | G0 | ☑ | `outputs/results/env.json`, commit `62c7dbd` |
| 1 | Dữ liệu: tải, xác minh, EDA | D1 chiều | G1 | ☑ | `outputs/results/data_report.json`, `outputs/results/train_meta.csv`, `outputs/figures/01..03`, `src/data.py` |
| 2 | Tiền xử lý & chia tập | D2 sáng | G1 | ☑ | `outputs/results/split_random.json`, `split_group.json`, `data_stats.json`, `phase2_check.json`, `data/cache/cache_meta.json`, `outputs/figures/04_augmentation.png`, `src/data.py` |
| 3 | Hạ tầng huấn luyện & logging | D2 chiều | G1 | ☑ | `outputs/runs/smoke_cnnb_s42/{metrics.csv,best.pt,curves.png,config.yaml,env.json,meta.json}`, `outputs/results/phase3_check.json`, `phase3_overfit.json`, `phase3_determinism.json`, `summary.csv` |
| 4 | Baseline | D3 | G2 | ☑ | `outputs/runs/e0a_majority/metrics.json`, `outputs/runs/e0b_hog_svc/{metrics.json,model.joblib}`, `outputs/results/phase4_check.json` |
| 5 | CNN & thí nghiệm chính | D4–D6 | G2 | ☑ | `outputs/runs/e1_cnna_s42` … `e7_classweight_s42`, `outputs/results/summary.csv`, `phase5_check.json`, `outputs/figures/05_experiment_comparison.png`, 7 config `configs/e*.yaml` |
| 6 | Chốt mô hình, multi-seed, test 1 lần | D7 | G3 | ☐ | |
| 7 | Đánh giá sâu & phân tích lỗi | D8–D9 | G4 | ☐ | |
| 8 | Demo dữ liệu mới | D10 | G4 | ☐ | |
| 9 | Tái lập & đóng gói | D11 | G5 | ☐ | |
| 10 | Báo cáo | D12–D13 | G5 | ☐ | |
| 11 | Slide & phản biện | D14 | G5 | ☐ | |
| 12 | Nghiệm thu & nộp | D15 | G5 | ☐ | |

### Sơ đồ gate

```
G0: máy chạy được GPU
 └─> G1: dữ liệu + pipeline + log đáng tin
      └─> G2: có bảng so sánh, chọn được mô hình
           └─> G3: test đã đánh giá đúng 1 lần, có số cuối
                └─> G4: phân tích lỗi + demo xong
                     └─> G5: tái lập + báo cáo + slide xong
```

---

# GIAI ĐOẠN 0 — Môi trường & công cụ

**Ngày dự kiến:** D1 sáng · **Thời lượng:** 2–3 giờ · **Gate:** G0

### 0.1. Mục tiêu
Có một môi trường Python chạy được PyTorch trên GPU RTX 5070, có repo Git và cấu trúc thư mục chuẩn, sẵn sàng cho mọi bước sau.

### 0.2. Giải thích
RTX 5070 là kiến trúc **Blackwell (compute capability 12.0)**. Các bản PyTorch cũ (CUDA 11.x/12.1) **không có kernel** cho kiến trúc này, sẽ báo lỗi kiểu `no kernel image is available for execution on the device`. Vì vậy phải cài PyTorch từ index **cu128** trở lên. Đây là bước chặn toàn bộ dự án, phải làm và kiểm tra ngay, không để đến lúc train mới phát hiện.

Cài trong **virtual environment** để:
- Không làm hỏng Python hệ thống.
- Ghi lại đúng phiên bản trong `requirements.txt` → tái lập được.
- Clean-room test ở Giai đoạn 9 có ý nghĩa.

### 0.3. Việc cần làm

**Bước 0.3.1 — Cài Python 3.12**
- Kiểm tra: `python --version` (nếu chưa có, cài bằng `winget install Python.Python.3.12` hoặc tải từ python.org).
- Trong PowerShell tại thư mục dự án:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Bước 0.3.2 — Cài PyTorch cho Blackwell**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn streamlit pyyaml tqdm pillow imagehash
```

**Bước 0.3.3 — Smoke test GPU**

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul OK', float(y.mean()))"
```

**Bước 0.3.4 — Khởi tạo Git và cấu trúc thư mục**

```powershell
git init
```

Tạo tối thiểu:

```
src/{__init__,config,data,models,baseline,train,evaluate,error_analysis,predict,summarize}.py
configs/
notebooks/
outputs/{runs,figures,results}/
external_test/
data/
app.py
requirements.txt
README.md
.gitignore
```

`.gitignore` phải có:

```
.venv/
__pycache__/
data/
outputs/runs/
*.pt
.ipynb_checkpoints/
```

**Bước 0.3.5 — Ghi môi trường**
- Sinh `outputs/results/env.json`: phiên bản Python, torch, torchvision, CUDA, tên GPU, driver, OS, CPU, RAM.

**Bước 0.3.6 — Commit đầu tiên**
- `git add . && git commit -m "chore: init project structure and environment"`

### 0.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Python chạy trong venv | `sys.prefix` trỏ vào `.venv` |
| 2 | PyTorch nhận GPU | `torch.cuda.is_available() == True` |
| 3 | Kiến trúc GPU đúng | `get_device_capability() == (12, 0)` |
| 4 | Matmul trên GPU chạy | In ra `matmul OK` |
| 5 | Repo Git có commit đầu | `git log --oneline` ≥ 1 dòng |
| 6 | Cấu trúc thư mục đủ | Đủ 10 file trong `src/` (có thể rỗng) |
| 7 | `env.json` tồn tại | Có ít nhất 8 khóa thông tin |

### 0.5. Checklist test

- [x] `python -c "import sys; print(sys.prefix)"` trỏ đúng `.venv`. → `C:\0\1.Laptrinh\BTLHocSau\.venv`
- [x] `torch.__version__` có hậu tố `+cu128` (hoặc bản CUDA ≥ 12.8). → `2.11.0+cu128`
- [x] `torch.cuda.is_available()` = `True`.
- [x] `torch.cuda.get_device_capability()` = `(12, 0)`.
- [x] Matmul 4096×4096 trên GPU chạy < 2 giây, không lỗi. → `0,0619 s`
- [x] `nvidia-smi` thấy tiến trình Python dùng GPU khi chạy matmul (hoặc `torch.cuda.memory_allocated() > 0`). → `16,12 MB` sau matmul 1024×1024
- [x] `git status` sạch (không có file rác chưa ignore).
- [x] `data/`, `.venv/`, `*.pt` không bị Git theo dõi (`git check-ignore -v data`). → khớp `.gitignore` dòng 10, 2, 12
- [x] `outputs/results/env.json` đọc được bằng `json.load`. → 18 khóa

### 0.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 0**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | 22/10/2026 — AI agent (theo yêu cầu) |
> | Python version | 3.12.10 (CPython 64-bit), venv tại `.venv` |
> | torch / torchvision | `2.11.0+cu128` / `0.26.0+cu128` |
> | CUDA / driver | CUDA build 12.8, cuDNN 9.19.0 / driver 610.88 |
> | GPU + capability | NVIDIA GeForce RTX 5070, 11,94 GB, `(12, 0)`, 48 SM |
> | Thời gian matmul smoke test | **0,0619 s** (ma trận 4096×4096, fp32) |
> | Commit hash đầu tiên | `62c7dbd75b649da2fa34d63614b4bf7393f2ab3e` |
> | Trạng thái | ☑ Đạt |
> | Ghi chú | `env.json` có 18 khóa; `src/` đủ 10 file; `git check-ignore` xác nhận `data/`, `.venv/`, `outputs/runs/`, `*.pt`, `*.npy` bị chặn; không push lên GitHub |

**Bằng chứng kèm theo**

| Bằng chứng | Vị trí |
|---|---|
| Thông số môi trường (Python, torch, CUDA, GPU, driver, OS, CPU, RAM, matmul) | `outputs/results/env.json` |
| Danh sách thư viện và phiên bản | `requirements.txt` |
| Cấu trúc mã nguồn | `src/` (10 file), `configs/`, `notebooks/`, `outputs/`, `external_test/`, `app.py` |
| Quy tắc chặn file lớn | `.gitignore` |

### 0.6.1. Nhật ký thực hiện (todos)

> **Ghi chú:** Nhật ký này được bổ sung ngày 22/10/2026 sau khi Giai đoạn 0 đã hoàn tất, do lần thực hiện đầu chưa ghi danh sách todos vào tài liệu. Nội dung dưới đây đối chiếu lại đúng các bước đã chạy, không phải kế hoạch dự kiến.

| # | Todo | Trạng thái | Bằng chứng |
|---|---|---|---|
| 0.1 | Kiểm tra Python 3.12 64-bit | ☑ | `python --version` → 3.12.10; `env.json` |
| 0.2 | Tạo `.venv` và kích hoạt | ☑ | `.venv/Scripts/python.exe` tồn tại |
| 0.3 | Cài PyTorch cu128 + torchvision | ☑ | `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` |
| 0.4 | Cài thư viện còn lại từ `requirements.txt` | ☑ | `requirements.txt`; import smoke test |
| 0.5 | Smoke test GPU RTX 5070 | ☑ | `torch.cuda.is_available()` = True; capability `(12, 0)`; matmul 4096² = 0,0619 s |
| 0.6 | Tạo cấu trúc repo (`src/`, `configs/`, `docs/`, `outputs/`, ...) | ☑ | `src/` đủ 10 file; cây thư mục trong báo cáo |
| 0.7 | Sinh `outputs/results/env.json` | ☑ | 18 khóa: Python, torch, CUDA, GPU, driver, OS, CPU, RAM, matmul |
| 0.8 | Cấu hình `.gitignore` chặn dữ liệu/checkpoint | ☑ | `git check-ignore` xác nhận `data/`, `.venv/`, `outputs/runs/`, `*.pt`, `*.npy` |
| 0.9 | Commit khung dự án | ☑ | `62c7dbd` — `chore: init project structure and environment` |
| 0.10 | Chạy checklist 0.5 và ghi kết quả vào tài liệu | ☑ | Mục 0.5, 0.6; commit `f0d9b56` |
| 0.11 | Bổ sung nhật ký todos còn thiếu | ☑ | Mục 0.6.1 này (commit bổ sung sau) |

**Tổng kết Giai đoạn 0:** 11/11 todo đạt; không có todo nào bị bỏ dở. Bài học quy trình: từ Giai đoạn 1 trở đi, danh sách todo và kết quả được ghi vào tài liệu **ngay trong lúc thực hiện**, không để bù sau.

### 0.7. Ghi chú / lưu ý

- **Không cài** `pip install torch` trần rồi mong có CUDA — bản PyPI mặc định thường là CPU.
- Nếu `cuda.is_available()` = False: kiểm tra driver, kiểm tra đã cài đúng index cu128, khởi động lại terminal, kiểm tra `nvidia-smi` thấy GPU.
- Không dùng Python 3.13 nếu gặp lỗi wheel; 3.11/3.12 an toàn nhất.
- Đừng commit `data/` và checkpoint — repo sẽ phình rất nhanh.
- Ghi lại **đúng** phiên bản vào `env.json`; đây là bằng chứng tái lập cho Chương 3 và phụ lục.

---

# GIAI ĐOẠN 1 — Dữ liệu: tải, xác minh, EDA

**Ngày dự kiến:** D1 chiều · **Thời lượng:** 3–4 giờ · **Gate:** G1

### 1.1. Mục tiêu
Có bộ GTSRB đầy đủ trên đĩa, đã **xác minh bằng số** (số ảnh, số lớp, phân bố, ảnh lỗi, trùng lặp), có hình EDA và `data_report.json` làm bằng chứng cho Chương 2.

### 1.2. Giải thích
Báo cáo không được dùng "số công bố" thay cho số thật. GTSRB công bố 39.209 train / 12.630 test, nhưng phải đếm lại vì:
- Cách thư viện tổ chức split có thể khác.
- Có thể thiếu file do tải lỗi.
- Mất cân bằng lớp là **luận cứ bắt buộc** cho việc dùng macro-F1, nên phải có biểu đồ và số liệu thật.

Ngoài ra phải kiểm tra **trùng lặp** vì GTSRB trích từ video: ảnh giống nhau giữa train và test (nếu có) sẽ làm kết quả test sai lệch.

### 1.3. Việc cần làm

**Bước 1.3.1 — Tải dữ liệu**

```powershell
python -m src.data --download
```

Nếu trang chính thức không truy cập được: dùng `download=True` của torchvision (mirror riêng), hoặc tải thủ công và đặt đúng cấu trúc thư mục mà torchvision mong đợi. Ghi lại cách đã dùng.

**Bước 1.3.2 — Đếm và xác minh**
- `len(train_ds)`, `len(test_ds)`.
- Đếm số ảnh từng lớp ở train và test.
- Kiểm tra đủ 43 lớp, không lớp nào 0 ảnh.
- Tổng số ảnh theo lớp == tổng dataset.

**Bước 1.3.3 — Kiểm tra ảnh**
- Mở toàn bộ ảnh (hoặc mẫu lớn) bằng PIL, đếm ảnh lỗi.
- Kiểm tra mode ảnh (RGB/L/P), chuyển RGB.
- Histogram kích thước gốc (width, height, aspect ratio).

**Bước 1.3.4 — Kiểm tra trùng lặp**
- Tính hash (MD5/SHA1) toàn bộ ảnh train → phát hiện ảnh trùng trong train.
- So hash train ↔ test → phát hiện trùng giữa hai tập.
- Ghi số cặp trùng; nếu có, quyết định loại khỏi train và ghi lý do.

**Bước 1.3.5 — Đọc ROI và track ID**
- Đọc CSV GT trong thư mục train và test: xác nhận cột `Roi.X1..Y2`, `ClassId` khớp nhãn torchvision.
- Trích **track ID** từ tên file (phần số đầu tiên). Kiểm tra bằng cách nhóm thử: số track mỗi lớp, số frame mỗi track.
- Lưu `track_id` cho từng ảnh train vào file trung gian `outputs/results/train_meta.csv` (path, label, track_id, roi).

**Bước 1.3.6 — Vẽ hình EDA**
1. Biểu đồ cột số ảnh/lớp (sắp tăng dần, tô đỏ lớp < 300 ảnh).
2. Histogram kích thước ảnh gốc.
3. Lưới ảnh mẫu 5×9 (mỗi lớp 1 ảnh, có mã + tên).

**Bước 1.3.7 — Ghi `data_report.json`**
Bao gồm: nguồn, ngày tải, cách tải, số ảnh train/test, số lớp, min/max/median ảnh/lớp, tỷ lệ mất cân bằng, số ảnh lỗi, số cặp trùng, số track, ghi chú license.

### 1.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Đủ 43 lớp | 43/43, không lớp rỗng |
| 2 | Số ảnh khớp công bố (hoặc giải thích được lệch) | 39.209 / 12.630 ± 0 |
| 3 | Tổng theo lớp == tổng dataset | Sai số 0 |
| 4 | Ảnh lỗi | 0 ảnh không đọc được (hoặc liệt kê + loại) |
| 5 | Trùng lặp | Đã đo; nếu có, đã xử lý và ghi lại |
| 6 | ROI đọc được | ≥ 99% ảnh có ROI hợp lệ |
| 7 | Track ID trích được | ≥ 95% ảnh train (nếu không, ghi rõ) |
| 8 | Hình EDA | 3 hình PNG trong `outputs/figures/` |
| 9 | `data_report.json` | Đủ trường, đọc được bằng `json.load` |

### 1.5. Checklist test

- [x] `len(train_ds) == 39209` (hoặc số thật đã giải thích). → **39.209**
- [x] `len(test_ds) == 12630`. → **12.630**
- [x] `sum(counts_per_class) == len(train_ds)`. → 39.209 = 39.209 (sai số 0)
- [x] `len(counts_per_class) == 43`. → **43**
- [x] `min(counts_per_class) > 0`. → **210** (lớp 0)
- [x] Tỷ lệ max/min ≥ 8 (xác nhận mất cân bằng thật sự). → **10,71** (2.250 / 210)
- [x] Ảnh lỗi = 0, hoặc danh sách lỗi được log. → **0/51.839**
- [x] Đã chạy hash và có số cặp trùng (kể cả 0). → train **0**, test **0**, train↔test **8 cặp** → đã **loại 8 ảnh train**, còn 39.201 ảnh dùng chia split
- [x] `train_meta.csv` có đúng `len(train_ds)` dòng, có cột `track_id`. → **39.209 dòng**, cột `track_id` + `track_key`
- [x] Mở 3 hình EDA bằng mắt: đọc được nhãn trục, thấy rõ lớp ít mẫu. → đã xem cả 3 hình; hình 03 đã sửa (ẩn ô trống, tiêu đề không chồng)
- [x] `data_report.json` có `license_note` (câu điều kiện sử dụng chép từ trang chính thức kèm ngày). → có, kèm ngày truy cập 2026-10-22 và nguồn Wayback

**Kết quả:** 11/11 mục PASS (script kiểm tra `phase1_check.py`: 18/18 PASS, gồm cả ROI ≥ 99%, track ≥ 95%, 3 hình > 10 KB, JSON đọc được).

### 1.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 1**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | 22/10/2026 — Nguyễn (repo `Nguyen15idhue/GTSRB-CNN`), chạy trên RTX 5070 / i9-14900KF / Windows 11 |
> | Cách tải (torchvision / thủ công / mirror) | Test: `torchvision.datasets.GTSRB(root="data", download=True)` (2 archive gốc). Train: tải `GTSRB_Final_Training_Images.zip` (MD5 `f33fd80ac59bff73c82d25ab499e03a3`) từ mirror `sid.erda.dk`, giải nén vào `data/gtsrb/GTSRB/Training` |
> | Ngày truy cập | 2026-10-22 (trang chính thức `benchmark.ini.rub.de` không phân giải DNS từ máy này) |
> | Số ảnh train / test | **39.209 / 12.630** trên đĩa (khớp công bố, lệch 0); sau khi loại 8 ảnh trùng test, **39.201** ảnh dùng để chia split |
> | Số lớp | **43/43** ở cả hai tập, không lớp rỗng |
> | Lớp ít nhất (ID, số ảnh) | Lớp **0** (Giới hạn tốc độ 20 km/h) — **210** ảnh train |
> | Lớp nhiều nhất (ID, số ảnh) | Lớp **2** (Giới hạn tốc độ 50 km/h) — **2.250** ảnh train |
> | Tỷ lệ mất cân bằng | **10,71** lần (max/min); trung vị 600, trung bình 911,8 ảnh/lớp; **9 lớp < 300 ảnh** |
> | Số ảnh lỗi | **0** — mở thành công 51.839/51.839 ảnh, 100% mode RGB |
> | Số cặp trùng trong train / train-test | Trong train **0** cặp; trong test **0** cặp; train↔test **8 cặp** (cùng nhãn 14, cùng track `00023`) → **đã loại 8 ảnh train** khỏi tập huấn luyện, danh sách ở `outputs/results/excluded_train_duplicates.csv` |
> | Số track (nếu trích được) | **1.307** track, trích được **100%** ảnh train (1.306 track đủ 30 frame, 1 track 29 frame); test không có track trong tên file |
> | License note (tóm tắt) | Nguyên văn: *"The data is free to use. However, we cordially ask you to cite the following publication if you do: ..."* (Stallkamp et al., IJCNN 2011) — không suy diễn thành giấy phép cụ thể |
> | Trạng thái | ☑ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | `torchvision` chỉ tải bản train rút gọn 26.640 ảnh → đã chuyển sang archive chính thức 39.209 ảnh. 8 ảnh train trùng MD5 với test **đã bị loại khỏi tập huấn luyện** (file vẫn giữ trên đĩa để số đếm khớp công bố; đánh dấu `usable_for_train=0`) → tập chia split còn **39.201** ảnh, dự kiến 31.360 / 7.841 (sẽ ghi số thật ở Giai đoạn 2). Track chỉ duy nhất trong một lớp → Giai đoạn 2 dùng `track_key = "<label>_<track>"` làm khóa nhóm. |

### 1.6.1. Nhật ký thực hiện (todos)

> **Cập nhật 22/10/2026 — hoàn tất.** Bảng ghi trong lúc chạy; số liệu cuối đã đối chiếu với `outputs/results/data_report.json`.

| # | Todo | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|---|
| 1.1 | Viết `src/data.py`: tải, quét ảnh, hash, ROI, track, EDA, report + CLI `--download/--report` | ☑ | `src/data.py` (~790 dòng); smoke test `--limit 400` chạy hết |
| 1.2 | Tải dữ liệu: test qua `torchvision` (12.630), train từ archive chính thức (39.209) | ☑ | `data/gtsrb/GTSRB/Training` (39.209 ảnh), `Final_Test/Images` (12.630), `GT-final_test.csv` |
| 1.3 | Đếm & xác minh số ảnh, 43 lớp, tổng theo lớp, tỷ lệ mất cân bằng | ☑ | 39.209 / 12.630 ảnh; 43/43 lớp; min 210 (lớp 0), max 2.250 (lớp 2); max/min = **10,71**; 9 lớp train < 300 ảnh |
| 1.4 | Quét ảnh: PIL, ảnh lỗi, mode, histogram kích thước gốc | ☑ | 51.839 ảnh mở được, **0 lỗi**, 100% mode RGB; cạnh ngắn 25–266 px, median 43 px; aspect ratio 0,368–1,438 |
| 1.5 | Hash MD5: trùng trong train, trong test, train↔test | ☑ | Trong train: 0; trong test: 0; **8 cặp train↔test** (cùng nhãn 14, cùng track `00023`) → **đã loại 8 ảnh train** khỏi tập huấn luyện, lưu `excluded_train_duplicates.csv` |
| 1.6 | ROI từ `GT-*.csv` + `GT-final_test.csv`, đối chiếu ClassId, trích `track_id` → `train_meta.csv` | ☑ | ROI hợp lệ 39.209/39.209 và 12.630/12.630 (**100%**); ClassId lệch **0**; track trích được **100%**, 1.307 track (1.306 track đủ 30 frame, 1 track 29 frame); `train_meta.csv` 39.209 dòng + cột `usable_for_train` (39.201 = 1, 8 = 0) |
| 1.7 | Vẽ 3 hình EDA | ☑ | `outputs/figures/01_class_distribution.png` (50 KB), `02_image_sizes.png` (45 KB), `03_sample_grid.png` (308 KB); đã mở cả 3 bằng mắt, đọc được nhãn trục, lớp < 300 tô đỏ, lưới 43 ảnh crop theo ROI |
| 1.8 | Sinh `data_report.json` đầy đủ trường + `license_note` | ☑ | `outputs/results/data_report.json` (6,7 KB): nguồn, cách tải, MD5 archive, per-class train/test, thống kê kích thước, trùng lặp + danh sách 8 cặp, ROI, track, `license_note` |
| 1.9 | Chạy checklist 1.5, điền 1.6, cập nhật bảng tiến độ + commit | ☑ | Checklist 11/11 mục (script phụ: 18/18 PASS); đã điền 1.5 + 1.6 + bảng tiến độ; commit local |

**Tổng kết: 9/9 todo hoàn thành.** Bằng chứng: `src/data.py`, `outputs/results/data_report.json`, `outputs/results/train_meta.csv`, `outputs/figures/01..03`. Lệnh tái tạo: `python -m src.data --download --report` (khoảng 25 giây khi cache đĩa nóng, ~4,5 phút khi nguội).

**Phát hiện quan trọng (đã xử lý):** `torchvision.datasets.GTSRB` tải train từ `GTSRB-Training_fixed.zip`, bản này chỉ có **26.640 ảnh** (888 track × 30 frame), không khớp con số công bố 39.209. Vì vậy train được lấy từ archive chính thức `GTSRB_Final_Training_Images.zip` (MD5 `f33fd80ac59bff73c82d25ab499e03a3`, 39.209 ảnh, có `GT-*.csv`), giải nén vào `data/gtsrb/GTSRB/Training` để `torchvision.datasets.GTSRB(root="data")` vẫn đọc được; test vẫn dùng torchvision (12.630 ảnh). `src/config.py` đã sửa `TRAIN_IMAGES_DIR` từ `Final_Training/Images` sang `Training` cho khớp thực tế trên đĩa.

### 1.7. Ghi chú / lưu ý

- **Không** dùng `random_split` mù quáng: phải `stratify` theo nhãn.
- Nếu tải lại dữ liệu, phải chạy lại toàn bộ hash và cập nhật `data_report.json`.
- Ảnh PPM có thể có mode `P`/`L`; luôn `convert("RGB")` trước khi resize.
- Nếu track ID không trích được theo giả định, **đừng bịa**; ghi rõ "không trích được" và dùng split random + phân tích val–test.
- Khi hiển thị ảnh trong báo cáo: crop quanh ROI để tránh lộ người/xe ở nền.
- License: chép nguyên văn câu điều kiện sử dụng tại ngày truy cập; không suy diễn thành "CC-BY" hay bất kỳ giấy phép cụ thể nào.
- **8 cặp trùng train↔test đã xử lý** (đều nhãn 14, track `00023`): **loại 8 ảnh train** khỏi tập huấn luyện, lưu danh sách `outputs/results/excluded_train_duplicates.csv`; file vẫn giữ trên đĩa nên `len(train_ds) = 39.209`, nhưng Giai đoạn 2 chỉ chia split trên 39.201 ảnh `usable_for_train=1`.

---

# GIAI ĐOẠN 2 — Tiền xử lý & chia tập

**Ngày dự kiến:** D2 sáng · **Thời lượng:** 3 giờ · **Gate:** G1

### 2.1. Mục tiêu
Có pipeline dữ liệu **dùng chung** cho train/val/test/demo, hai bộ split (random và group), cache ảnh đã resize, và bằng chứng không rò rỉ.

### 2.2. Giải thích
Ba lý do phải làm cẩn thận:
1. **Nhất quán**: nếu demo resize/normalize khác lúc train, model sẽ dự đoán sai dù test 99%. Vì vậy chỉ có **một** hàm `preprocess()`.
2. **Tốc độ**: decode PPM mỗi epoch rất chậm; resize một lần rồi giữ trong RAM giúp epoch nhanh gấp nhiều lần.
3. **Rò rỉ**: GTSRB trích từ video, frame liên tiếp gần trùng. Chia theo track cho val sạch hơn; so sánh hai cách để **định lượng** mức lạc quan.

### 2.3. Việc cần làm

**Bước 2.3.1 — Cache dữ liệu**
- Resize toàn bộ ảnh về 64×64, lưu `uint8` dạng `.npy` (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`).
- Ghi kích thước file và thời gian tạo vào log.

**Bước 2.3.2 — Tính mean/std từ train**
- Tính trên train **sau khi** chia (không tính trên val/test).
- Lưu vào `configs/final.yaml` (và config từng run).

**Bước 2.3.3 — Tạo split**
- `split_random`: `train_test_split(stratify=y, test_size=0.2, random_state=42)`.
- `split_group`: `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` với group = track ID.
- Lưu index vào `outputs/results/split_random.json` và `split_group.json` để tái sử dụng (không sinh lại mỗi lần chạy).

**Bước 2.3.4 — Viết `src/data.py`**
Các hàm tối thiểu:
- `load_gtsrb(root, size)` → ảnh, nhãn, path, track_id.
- `make_splits(labels, tracks, val_ratio, seed, mode)` → index.
- `compute_stats(images)` → mean, std.
- `preprocess(image)` → tensor đã resize + normalize (dùng cho cả demo).
- `make_loaders(...)` với augmentation chỉ ở train.

**Bước 2.3.5 — Kiểm tra trực quan augmentation**
- Vẽ lưới 8 ảnh gốc và 8 ảnh sau augmentation.
- Xác nhận: không lật ngang, không xoay 90/180, màu đỏ/xanh vẫn giữ nghĩa.

### 2.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Train/val | 31.360 / 7.841 (±1 do làm tròn) — trên 39.201 ảnh `usable_for_train=1` |
| 2 | Đủ lớp ở val | 43/43 |
| 3 | Tỷ lệ lớp val vs train | Lệch ≤ 1% mỗi lớp |
| 4 | Không giao index train/val | Giao = 0 |
| 5 | Split group: track rời nhau | Giao track train/val = 0 |
| 6 | mean/std | Nằm trong khoảng hợp lý (0,1–0,8) |
| 7 | Cache | Đúng shape `(N, 64, 64, 3)`, dtype `uint8` |
| 8 | `preprocess` dùng chung | Chỉ định nghĩa 1 lần trong `src/data.py` |

### 2.5. Checklist test

- [x] `len(train_idx) + len(val_idx) == 39201` (39.209 trừ 8 ảnh trùng test đã loại). → **31.360 + 7.841 = 39.201**
- [x] `len(set(train_idx) & set(val_idx)) == 0`. → **0**
- [x] `len(np.unique(y[train_idx])) == 43` và `len(np.unique(y[val_idx])) == 43`. → **43/43** cả hai
- [x] Với split group: `set(track[train_idx]) & set(track[val_idx]) == ∅`. → **∅** (1.045 track train / 262 track val); **4 lớp vắng ở val: 0, 19, 34, 39** (đã ghi rõ, đúng cảnh báo 2.7)
- [x] `X_train.npy` có shape `(39201, 64, 64, 3)`, `dtype == uint8`. → đúng shape, `uint8`; `X_test.npy` = `(12630, 64, 64, 3)`
- [x] `preprocess(PIL.open(any))` trả tensor shape `(3, 64, 64)`, giá trị xấp xỉ [-2, 2]. → `(3, 64, 64)`, min −0,667 / max **2,558**; biên lý thuyết ±2,60 do std ≈ 0,27 (tài liệu ghi "xấp xỉ" — đã kiểm tra theo biên lý thuyết, không phải bug)
- [x] Lưới augmentation: không có ảnh bị lật; biển "rẽ trái" không thành "rẽ phải". → đã mở `04_augmentation.png`: chỉ dịch/scale/shear + jitter màu nhẹ, số trên biển 20/30/50/60/70/100 giữ nguyên chiều
- [x] Import `preprocess` từ `src.data` trong notebook/app hoạt động. → subprocess `from src.data import preprocess` → `(3, 64, 64)`
- [x] `split_random.json` và `split_group.json` tồn tại, đọc lại cho cùng index. → cả hai tồn tại; đọc lại cho index giống hệt

**Kết quả:** 12/12 mục PASS (script `python -m src.data --verify` → `outputs/results/phase2_check.json`).

### 2.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 2**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | 22/10/2026 — AI agent (theo yêu cầu), RTX 5070 / i9-14900KF / Windows 11 |
> | Số train / val (random) | **31.360 / 7.841** (stratified, seed 42) |
> | Số train / val (group) | **31.341 / 7.860** (GroupShuffleSplit theo `track_key`, seed 42) |
> | Số track train / val (group) | **1.045 / 262** (tổng 1.307; giao = 0) |
> | Lớp ít mẫu nhất ở val (ID, số ảnh) | Random: lớp **0** — **42** ảnh. Group: 4 lớp **vắng mặt** ở val (0, 19, 34, 39) → chỉ dùng group để đo Δ rò rỉ, không dùng để chọn mô hình |
> | mean / std | mean **[0,340408, 0,312365, 0,321632]** / std **[0,275148, 0,264242, 0,270747]** (tính trên train split random, đơn vị [0,1]) |
> | Dung lượng cache | **610,7 MB** (`X_train.npy` 481,7 MB · `X_test.npy` 155,2 MB · nhãn/ROI/paths 8,3 MB), tại `data/cache/` |
> | Thời gian tạo cache | **13,6 s** (39.201 + 12.630 ảnh, resize 64×64 bilinear, ổ đĩa nóng) |
> | Trạng thái | ☑ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | `preprocess()` là **hàm duy nhất** trong `src/data.py`, không viết lại ở nơi khác; augmentation chỉ gắn vào `train_loader`; có kiểm tra tự động chặn `Flip`/`RandomRotation`; ảnh đã chuẩn hóa có max 2,558 (biên lý thuyết 2,60) nên "xấp xỉ [-2,2]" trong tài liệu chỉ là ước lượng. Group split không stratify nên 4 lớp hiếm vắng ở val — đã ghi rõ theo 2.7. |

### 2.6.1. Nhật ký thực hiện (todos)

| # | Todo | Trạng thái | Bằng chứng |
|---|---|---|---|
| 2.1 | Dựng cache `.npy` uint8 64×64 (train pool 39.201 + test 12.630), kèm nhãn, ROI, paths, `cache_meta.json` | ☑ | `data/cache/` 610,7 MB; 13,6 s |
| 2.2 | `make_splits` random (stratified) + group (theo track), lưu JSON, không sinh lại mỗi run | ☑ | `outputs/results/split_random.json`, `split_group.json` |
| 2.3 | `compute_stats` trên **train split** → `data_stats.json`, nạp tự động vào config | ☑ | `outputs/results/data_stats.json`; `src/config.py::_fill_stats` |
| 2.4 | `preprocess()` dùng chung + `make_loaders` (aug chỉ train, seed cả generator/worker) | ☑ | `src/data.py`; smoke batch `(128,3,64,64)` |
| 2.5 | Hình lưới augmentation, kiểm tra không lật/xoay 90–180 | ☑ | `outputs/figures/04_augmentation.png` (đã mở bằng mắt) |
| 2.6 | CLI `--prepare`, `--verify`; chạy checklist 2.5 | ☑ | `python -m src.data --prepare|--verify` → 12/12 PASS |
| 2.7 | Điền 2.5/2.6 + bảng tiến độ, commit local | ☑ | commit local (không push) |

**Tổng kết: 7/7 todo đạt.** Lệnh tái tạo: `python -m src.data --prepare --verify` (khoảng 20 s khi cache nóng, ~1 phút khi nguội).

### 2.7. Ghi chú / lưu ý

- `train_test_split` với `stratify` cần nhãn dạng mảng 1D; kiểm tra dtype.
- Với `GroupShuffleSplit`, **không** dùng `stratify` — phải kiểm tra lại phân bố lớp sau khi chia; nếu lớp hiếm vắng mặt ở val, ghi rõ.
- Không tính mean/std trên toàn bộ dữ liệu (rò rỉ nhẹ từ val/test).
- Nếu đổi kích thước ảnh (E6), phải tạo cache riêng; không ghi đè cache 64×64.
- Lưu index split ra file: nếu mỗi lần chạy sinh lại, các run sẽ không so sánh được với nhau.

---

# GIAI ĐOẠN 3 — Hạ tầng huấn luyện & logging

**Ngày dự kiến:** D2 chiều · **Thời lượng:** 4 giờ · **Gate:** G1

### 3.1. Mục tiêu
Có một vòng train hoàn chỉnh, chạy được 1 epoch, ghi log đúng schema, vẽ curve, lưu checkpoint, và tái lập được với seed. Chưa cần kết quả tốt — chỉ cần **đúng và đáng tin**.

### 3.2. Giải thích
Toàn bộ giá trị của báo cáo nằm ở chỗ số liệu đáng tin. Nếu log sai (ví dụ tính accuracy trên tập augmented, hoặc quên `model.eval()`), mọi kết luận sau đó vô nghĩa. Vì vậy phải **test hạ tầng trước khi chạy thí nghiệm**: overfit 1 batch, kiểm tra loss giảm, kiểm tra determinism.

### 3.3. Việc cần làm

**Bước 3.3.1 — Viết `src/models.py`**
- `SmallCNN(widths, use_bn, dropout, gap)`.
- `build_model(cfg)` trả model + số params.
- Test: forward với batch giả `(2, 3, 64, 64)` → output `(2, 43)`.

**Bước 3.3.2 — Viết `src/train.py`**
- Đọc config YAML, validate key.
- Set seed (Python, NumPy, torch, CUDA, DataLoader generator).
- Train loop: AMP, gradient step, eval val mỗi epoch.
- Ghi `metrics.csv` sau mỗi epoch (flush ngay).
- Lưu `best.pt` theo `val_macro_f1`.
- Early stopping patience 7.
- Sinh `curves.png` cuối run.
- Copy `config.yaml` + `env.json` vào thư mục run.

**Bước 3.3.3 — Test "overfit 1 batch"**
- Lấy 1 batch 32 ảnh, train 50 vòng chỉ trên batch đó.
- Kỳ vọng: loss giảm gần 0, accuracy = 100%.
- Nếu không → bug ở model/loss/optimizer/label.

**Bước 3.3.4 — Test determinism**
- Chạy 2 lần cùng seed, 3 epoch, so sánh `train_loss` từng epoch.
- Kỳ vọng: giống nhau (sai khác < 1e-4) hoặc chênh rất nhỏ do AMP/CUDA (ghi lại).

**Bước 3.3.5 — Test "rò rỉ trong code"**
- Kiểm tra augmentation **chỉ** áp dụng cho train loader.
- Kiểm tra `model.eval()` + `torch.no_grad()` khi eval val.
- Kiểm tra không có thao tác nào đọc nhãn val để cập nhật trọng số.

**Bước 3.3.6 — Viết `src/summarize.py`**
- Quét `outputs/runs/*/metrics.csv` + `config.yaml` → sinh `outputs/results/summary.csv`.

### 3.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | 1 epoch chạy xong | < 60 giây (mục tiêu 10–25 s) |
| 2 | `metrics.csv` | Đúng 9 cột, 1 dòng/epoch, không NaN |
| 3 | Overfit 1 batch | Loss < 0,05 sau 50 vòng |
| 4 | Determinism | 2 lần chạy cùng seed lệch < 1e-4 (hoặc ghi rõ lý do) |
| 5 | Checkpoint | `best.pt` load lại được, forward ra `(N, 43)` |
| 6 | Curves | `curves.png` có 3 subplot (loss, acc, macro-F1) |
| 7 | `env.json` | Có trong thư mục run |
| 8 | `summary.csv` | Sinh được từ 1 run |

### 3.5. Checklist test

- [x] `python -m src.train --config configs/smoke.yaml` chạy hết 1 epoch không lỗi. → 5 epoch xong trong **29,3 s** (epoch 1: 18,9 s do warm-up cache; các epoch sau 2,5–2,6 s)
- [x] `metrics.csv` dòng đầu có `epoch=1`, không có ô trống. → epoch 1, 5 dòng, 0 ô trống, đúng 9 cột
- [x] `val_macro_f1` nằm trong [0, 1]. → min 0,9495 / max 0,9946
- [x] `train_loss` giảm qua 3 epoch đầu. → 1,1543 → 0,1729 → 0,0863
- [x] Overfit 1 batch đạt accuracy 100% (test riêng). → loss cuối **0,021139** (< 0,05), accuracy **100%** sau 50 vòng (batch 32)
- [x] Chạy lại cùng seed 3 epoch: loss 2 lần khớp (hoặc chênh < 1e-4). → **lệch 0,000e+00** (1,15432531 / 0,17986633 / 0,09668001 ở cả hai lần)
- [x] `torch.load("best.pt")` → `model.load_state_dict` không lỗi. → load `weights_only=True` OK, forward `(2, 43)`, khóa đủ 7/7
- [x] `model.eval()` được gọi trước khi eval val (kiểm tra bằng biến `model.training == False` trong log). → `_ModeSpy` ghi lại: forward khi eval có `training=False`, `grad_enabled=False`
- [x] `curves.png` mở được, trục có nhãn, có legend. → 3 subplot (loss / accuracy / macro-F1), 77 KB, đã mở bằng mắt
- [x] `summary.csv` có đúng 1 dòng cho run smoke. → 1 dòng `smoke_cnnb_s42`

**Kết quả:** 12/12 mục PASS (`python -m src.train --config configs/smoke.yaml --verify` → `outputs/results/phase3_check.json`).

### 3.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 3**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | 22/10/2026 — AI agent, RTX 5070 / i9-14900KF / Windows 11 |
> | Thời gian 1 epoch | **2,5–2,6 s** (epoch 1: 18,9 s do warm-up cache/worker); trung bình 5,86 s/epoch cho run 5 epoch |
> | Overfit 1 batch: loss cuối / acc | **0,021139 / 100 %** (batch 32, 50 vòng, AdamW lr 1e-3, không AMP) |
> | Determinism: loss epoch 1/2/3 lần 1 | 1,15432531 / 0,17986633 / 0,09668001 |
> | Determinism: loss epoch 1/2/3 lần 2 | 1,15432531 / 0,17986633 / 0,09668001 → **lệch lớn nhất 0,000e+00** |
> | Số params model smoke | **2.201.931** (CNN-B: 3 khối conv 32/64/128 + BN + dropout 0,3 + FC 8192→256→43) |
> | Trạng thái | ☑ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | Val acc 0,9949 sau 5 epoch là **cao bất thường** vì `split_random` chứa frame gần trùng của train (rò rỉ frame) — số này **không dùng làm kết quả báo cáo**, Giai đoạn 6 sẽ đo Δ giữa random và group. Đã sửa 2 bug trong lúc test: (1) `torch.__version__` là `TorchVersion` nên `weights_only=True` không load được checkpoint → lưu `str(...)`; (2) test `eval_guard` ban đầu PASS giả do lỗi lệch device, đã thay bằng `_ModeSpy` ghi lại `training`/`grad_enabled` thật sự. `metrics.csv` ghi + flush từng epoch. Run smoke vẫn nằm trong `summary.csv` để làm bằng chứng, sẽ loại bỏ ở Giai đoạn 9. |

### 3.6.1. Nhật ký thực hiện (todos)

| # | Todo | Trạng thái | Bằng chứng |
|---|---|---|---|
| 3.1 | `src/models.py`: `SmallCNN` (widths/BN/dropout/GAP) + `build_model` + đếm params | ☑ | forward `(2,3,64,64)` → `(2,43)`; 2.201.931 params |
| 3.2 | `src/train.py`: đọc config, seed, AMP, train/val loop, ghi `metrics.csv` flush, early stopping, `best.pt`, `curves.png` | ☑ | `outputs/runs/smoke_cnnb_s42/` |
| 3.3 | Test overfit 1 batch (50 vòng) | ☑ | `outputs/results/phase3_overfit.json` |
| 3.4 | Test determinism 2 lần × 3 epoch | ☑ | `outputs/results/phase3_determinism.json` (lệch 0) |
| 3.5 | Test rò rỉ trong code: aug chỉ train, eval luôn `eval()` + `no_grad` | ☑ | check `aug_chi_train`, `eval_guard` trong `phase3_check.json` |
| 3.6 | `src/summarize.py`: quét run → `summary.csv` | ☑ | `outputs/results/summary.csv` 1 dòng smoke |
| 3.7 | Điền 3.5/3.6 + bảng tiến độ, commit local | ☑ | commit local (không push) |

**Tổng kết: 7/7 todo đạt.** Lệnh tái tạo: `python -m src.train --config configs/smoke.yaml [--overfit-batch|--determinism-check|--verify]`.

### 3.7. Ghi chú / lưu ý

- **Quên `model.eval()`** là lỗi phổ biến nhất: BN/Dropout vẫn hoạt động ở chế độ train → val không ổn định.
- Không tính metric trên ảnh đã augmentation. Chỉ augment train.
- `metrics.csv` phải flush từng dòng; nếu chỉ ghi cuối run, mất điện là mất hết.
- Trên Windows, `num_workers > 0` cần `if __name__ == "__main__":`; nếu treo, giảm về 0 để test.
- Nếu 1 epoch > 60 s: kiểm tra cache có được dùng không, `pin_memory`, `persistent_workers`, và AMP đã bật chưa.
- Đừng căn chỉnh hyperparameter ở giai đoạn này; chỉ cần hạ tầng đúng.

---

# GIAI ĐOẠN 4 — Baseline

**Ngày dự kiến:** D3 · **Thời lượng:** 3 giờ · **Gate:** G2 (cùng Giai đoạn 5)

### 4.1. Mục tiêu
Có 2 baseline chạy trên **cùng split**, cùng metric: majority-class và HOG + LinearSVC.

### 4.2. Giải thích
Baseline trả lời câu hỏi "CNN có thực sự đáng không?". Majority-class là **sàn** (cho thấy accuracy 6% là vô nghĩa khi dữ liệu mất cân bằng). HOG+SVM là baseline **chuẩn của GTSRB**, đủ mạnh để so sánh có ý nghĩa nhưng không phải deep learning.

Nếu baseline HOG đạt 95% mà CNN chỉ 90%, vấn đề nằm ở pipeline CNN, không phải ở dữ liệu.

### 4.3. Việc cần làm

**Bước 4.3.1 — Majority class**
- Tìm lớp nhiều nhất trong train, dự đoán cho mọi ảnh val/test.
- Tính accuracy, macro-F1.

**Bước 4.3.2 — HOG + LinearSVC**
- Grayscale 64×64 → HOG (`orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`, `block_norm='L2-Hys'`).
- `StandardScaler` fit trên train.
- Thử `C ∈ {0.1, 1, 10}` trên **val**, chọn theo val macro-F1.
- Train lại với C tốt nhất, dự đoán **val**; chỉ đụng test ở Giai đoạn 6.

**Bước 4.3.3 — Đo thời gian**
- Thời gian trích đặc trưng, thời gian fit, thời gian inference/ảnh.

**Bước 4.3.4 — Lưu kết quả**
- `outputs/runs/e0a_majority/metrics.json`.
- `outputs/runs/e0b_hog_svc/metrics.json` + `model.joblib`.

### 4.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Majority accuracy | 5–7% (bằng tần suất lớp nhiều nhất) |
| 2 | Majority macro-F1 | < 0,01 |
| 3 | HOG+SVC val acc | 90–97% (dưới 90% → kiểm tra pipeline HOG) |
| 4 | HOG+SVC val macro-F1 | Chênh accuracy ≤ 5 điểm % |
| 5 | Cùng split | Dùng đúng `split_random.json` |
| 6 | Thời gian được ghi | Có số đo thực tế |

### 4.5. Checklist test

- [x] Majority dự đoán đúng lớp có tần suất cao nhất (kiểm tra bằng `np.bincount`). → lớp **2**, 1.800 ảnh train
- [x] `accuracy_majority ≈ count_max / N_train`. → val acc **0,057391** ≈ tỷ lệ lớp 2 ở val (0,057391)
- [x] HOG feature shape đúng (khoảng 1.764 chiều). → **(3, 1764)**, `feature_dim = 1764`
- [x] Không có NaN trong feature (kiểm tra `np.isnan(X).any()`). → **False**
- [x] Bảng thử C có 3 dòng, chọn C theo val macro-F1. → C=0,1 → 0,9740 · C=1 → 0,9673 · C=10 → 0,9656 → **chọn C=0,1**
- [x] Kết quả lưu JSON, đọc lại được. → `metrics.json` của cả hai baseline đọc lại OK
- [x] Đã ghi thời gian trích đặc trưng và thời gian fit. → trích **51,33 s**; fit C=0,1 **540,18 s**; suy luận **0,0069 ms/ảnh**

**Kết quả:** 9/9 mục PASS (`python -m src.baseline --verify` → `outputs/results/phase4_check.json`).

### 4.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 4**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | 22–23/10/2026 — AI agent, RTX 5070 / i9-14900KF / Windows 11 (baseline chạy CPU) |
> | Lớp majority (ID, số ảnh train) | Lớp **2** (Giới hạn tốc độ 50 km/h) — **1.800** ảnh train |
> | Majority: val acc / macro-F1 | **0,057391** / **0,002524** |
> | HOG: số chiều | **1.764** (ảnh xám 64×64, 9 hướng, cell 8×8, block 2×2, L2-Hys) |
> | HOG: thời gian trích đặc trưng | **51,33 s** cho 39.201 ảnh (train + val) |
> | HOG: C tốt nhất | **C = 0,1** (val macro-F1 0,9740; C=1 → 0,9673; C=10 → 0,9656) |
> | HOG: val acc / macro-F1 | **0,965693** / **0,973994** |
> | HOG: thời gian fit / inference | fit **540,18 s** (C=0,1) · tổng 3 lần thử C + trích đặc trưng **1.926,43 s** · suy luận **0,0069 ms/ảnh** |
> | Trạng thái | ☑ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | Bảng thử C: 0,1 → **0,9657**/**0,9740** (540,2 s) · 1 → 0,9594/0,9673 (660,8 s) · 10 → 0,9573/0,9656 (674,1 s). Macro-F1 **cao hơn** accuracy là hợp lý vì 43 lớp được tính ngang nhau còn accuracy bị chi phối bởi lớp nhiều mẫu. Lần chạy đầu dùng `dual="auto"` (primal) mất **993,9 s/lần fit** và bị timeout → đổi sang `dual=true` (liblinear dual coordinate descent, vẫn là `LinearSVC` chuẩn, ghi trong `configs/default.yaml`) nhanh hơn ~1,8×; đã chạy lại **toàn bộ** baseline, không dùng số của lần chạy dở. Baseline dùng ảnh xám, **không augmentation**, cùng `split_random` với CNN. **Chưa chạm test.** |

### 4.6.1. Nhật ký thực hiện (todos)

| # | Todo | Trạng thái | Bằng chứng |
|---|---|---|---|
| 4.1 | `src/baseline.py`: majority class + đo thời gian | ☑ | `outputs/runs/e0a_majority/metrics.json` |
| 4.2 | HOG (1764 chiều) + StandardScaler + LinearSVC, grid C trên val | ☑ | `outputs/runs/e0b_hog_svc/metrics.json` + `model.joblib` |
| 4.3 | Đo thời gian trích đặc trưng / fit / inference | ☑ | 51,33 s / 540,18 s / 0,0069 ms/ảnh |
| 4.4 | Lưu metrics.json, đưa baseline vào `summary.csv` | ☑ | `outputs/results/summary.csv` 3 dòng (2 baseline + smoke) |
| 4.5 | Checklist 4.5 + điền 4.6 + bảng tiến độ, commit local | ☑ | `phase4_check.json` 9/9 PASS; commit local (không push) |

**Tổng kết: 5/5 todo đạt.** Lệnh tái tạo: `python -m src.baseline --config configs/default.yaml [--force|--verify]` (~33 phút trên CPU).

### 4.7. Ghi chú / lưu ý

- `LinearSVC` không có `predict_proba`; nếu cần confidence, dùng `decision_function` (margin) chứ không giả lập xác suất.
- HOG+SVC với 43 lớp là one-vs-rest → có thể mất vài phút; đừng tưởng bị treo.
- Không tối ưu baseline quá mức: mục tiêu là **mốc so sánh**, không phải mô hình chính.
- Ghi rõ baseline dùng grayscale và không augmentation — nếu không, phần so sánh sẽ bị phản biện.

---

# GIAI ĐOẠN 5 — CNN & thí nghiệm chính

**Ngày dự kiến:** D4–D6 · **Thời lượng:** 3 ngày · **Gate:** G2

### 5.1. Mục tiêu
Chạy xong E1, E2, E3, E4 (bắt buộc) và E5, E7 (khuyến nghị), có bảng so sánh đầy đủ và chọn được mô hình theo **val macro-F1**.

### 5.2. Giải thích
Đây là phần "học sâu" của bài. Nguyên tắc: **mỗi run chỉ thay một nhóm yếu tố** để quy được tác động:
- E1 → E2: thêm BN + dropout + WD (gói regularization).
- E2 → E3: thêm augmentation.
- E3 → E4: đổi kiến trúc (sâu hơn + GAP).
- E5: đổi learning rate.
- E7: đổi cách xử lý mất cân bằng.

Không chạy song song nhiều run trên cùng GPU nếu VRAM không đủ; chạy tuần tự để thời gian/epoch đo được chính xác.

### 5.3. Việc cần làm

**Bước 5.3.1 — Tạo config cho từng run**
- `configs/e1_cnna.yaml`, `e2_cnnb.yaml`, `e3_cnnb_aug.yaml`, `e4_cnnc_gap_aug.yaml`, `e5a_lr3e4.yaml`, `e5b_lr3e3.yaml`, `e7_classweight.yaml`.
- Mỗi config ghi rõ `experiment.id`, `seed`, `notes` (thay đổi gì so với run gốc).

**Bước 5.3.2 — Chạy E1, E2, E3, E4**
```powershell
python -m src.train --config configs/e1_cnna.yaml
python -m src.train --config configs/e2_cnnb.yaml
python -m src.train --config configs/e3_cnnb_aug.yaml
python -m src.train --config configs/e4_cnnc_gap_aug.yaml
```

**Bước 5.3.3 — Chạy E5, E7**
- E5 chỉ cần 10 epoch để so sánh xu hướng, không cần hội tụ hoàn toàn; ghi rõ trong báo cáo.
- E7 dùng `class_weight = 1/sqrt(freq)`.

**Bước 5.3.4 — Sinh bảng so sánh**
```powershell
python -m src.summarize
```
- Kiểm tra `summary.csv` có đủ 7 run.
- Vẽ hình so sánh val acc / macro-F1 giữa các run.

**Bước 5.3.5 — Chọn mô hình**
- Tiêu chí: **val macro-F1 cao nhất**, ưu tiên mô hình có ít tham số hơn nếu chênh < 0,3%.
- Ghi lý do chọn vào `configs/final.yaml` (comment) và nhật ký.

### 5.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | E1 val acc | ≥ 90% (dưới → kiểm tra pipeline) |
| 2 | E3 val acc | ≥ 97% |
| 3 | E3 val macro-F1 | ≥ 0,95 |
| 4 | E3 > E1 | Có cải thiện rõ (≥ 2 điểm %) |
| 5 | Mỗi run | Có config, metrics.csv, best.pt, curves.png |
| 6 | Số run | ≥ 4 run CNN |
| 7 | Bảng so sánh | Đủ params, thời gian, epoch dừng |
| 8 | Chọn mô hình | Có lý do bằng số, không cảm tính |

### 5.5. Checklist test

- [x] Cả 4 run E1–E4 kết thúc, không run nào crash giữa chừng. → xong cả **7 run** (E1–E4, E5a, E5b, E7), exit code 0
- [x] Mỗi run có `metrics.csv` với số epoch ≥ 5. → E1 29 · E2 40 · E3 40 · E4 39 · E5a 10 · E5b 10 · E7 28 epoch
- [x] `best.pt` của mỗi run load lại được và cho output 43 lớp. → 7/7 load `weights_only=True`, forward `(2, 43)`
- [x] Trong `summary.csv`, không có ô trống ở val acc / val macro-F1. → 7 dòng CNN, 0 ô trống
- [x] Kiểm tra trực quan: curves của E3 mượt, không dao động bất thường. → đã mở `e3_cnnb_aug_s42/curves.png`: loss giảm đơn điệu, val acc tăng đều lên 0,9994, không nhảy bậc
- [x] E1 không có BN/dropout (đọc config xác nhận). → `use_bn=False`, `dropout=0.0`
- [x] E3 có aug, E2 không aug (đọc config xác nhận). → E2 `aug=False`, E3 `aug=True`
- [x] E4 có GAP và 4 khối conv (in `model` ra kiểm tra). → `gap=True`, `widths=[32,64,128,256]`, `features` có 4 khối
- [x] Số params E4 < E3 (xác nhận lập luận GAP). → **399.947 < 2.201.931** (giảm **81,8 %**)
- [x] Đã ghi lại epoch dừng của từng run (early stopping hay hết epoch). → E1 early 29/40 · E2 hết 40/40 · E3 hết 40/40 · E4 early 39/40 · E5a hết 10/10 · E5b hết 10/10 · E7 early 28/40
- [x] Chênh lệch val–train acc được ghi lại cho từng run (phát hiện overfit). → xem cột `Train–Val gap` ở 5.6 (E1 +0,37 pp … E5b −26,4 pp)

**Kết quả:** 11/11 mục PASS (`python -m src.summarize --verify` → `outputs/results/phase5_check.json`).

### 5.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 5**
>
> | Run | Params | Epoch dừng | Val Acc | Val Macro-F1 | Train–Val gap | Thời gian |
> |---|---|---|---|---|---|---|
> | E1 CNN-A | 2.201.707 | 29/40 (early stop, best 22) | 0,996301 | 0,995705 | +0,37 pp | 83,2 s (2,85 s/epoch) |
> | E2 CNN-B | 2.201.931 | 40/40 (hết epoch, best 33) | 0,998342 | 0,998392 | +0,12 pp | 120,8 s (3,01 s/epoch) |
> | E3 CNN-B + aug | 2.201.931 | 40/40 (hết epoch, best 36) | 0,999362 | 0,999471 | **−1,22 pp** (train 0,9871 < val) | 378,1 s (9,44 s/epoch) |
> | E4 CNN-C + GAP | **399.947** | 39/40 (early stop, best 32) | **0,999745** | **0,999763** | −0,07 pp | 369,5 s (9,46 s/epoch) |
> | E5a lr 3e-4 | 2.201.931 | 10/10 (hết epoch, best 10) | 0,989542 | 0,993387 | −6,33 pp | 107,4 s (10,7 s/epoch) |
> | E5b lr 3e-3 | 2.201.931 | 10/10 (hết epoch, best 9) | 0,771968 | 0,567247 | −26,38 pp | 107,8 s (10,74 s/epoch) |
> | E7 class weight | 2.201.931 | 28/40 (early stop, best 21) | 0,998470 | 0,998914 | −3,52 pp | 270,4 s (9,63 s/epoch) |
>
> **So sánh E5 ở cùng epoch 10** (E5 chỉ chạy 10 epoch nên chỉ so xu hướng, không so best epoch):
>
> | Run | lr | Val Acc @ep10 | Val Macro-F1 @ep10 | Nhận xét |
> |---|---|---|---|---|
> | E3 (tham chiếu) | 1e-3 | 0,993113 | 0,992170 | hội tụ tốt |
> | E5a | 3e-4 | 0,989542 | **0,993387** | tương đương E3 (chênh 0,12 pp < 0,3 pp) |
> | E5b | 3e-3 | 0,775666 | 0,564512 | **phân kỳ**, lr quá cao |
>
> | Hạng mục | Giá trị |
> |---|---|
> | Mô hình chọn | **E4 — CNN-C 4 khối conv + GAP + dropout 0,4 + augmentation** (`configs/e4_cnnc_gap_aug.yaml`, run `e4_cnnc_gap_aug_s42`) |
> | Lý do chọn | Val macro-F1 **cao nhất bảng (0,999763)**; so với E3 (0,999471) chênh **0,029 điểm % < 0,3 điểm %** → coi là **tương đương**, theo quy tắc ưu tiên mô hình **ít tham số hơn**: E4 chỉ **399.947** params so với 2.201.931 của E3 (**−81,8 %**), cùng tốc độ train (~9,5 s/epoch) và ít bộ nhớ hơn khi suy luận. E7 (0,998914) và E2 (0,998392) thấp hơn rõ hơn 0,1 điểm %, không chọn. |
> | Nhận xét chính (3 gạch đầu dòng) | • **Đều bão hoà ~99,6–99,97 % trên val random**: 100 % (7.841/7.841) ảnh val có **frame liền kề (±1) cùng track trong train** → val bị rò rỉ frame, nên các chênh lệch giữa run là rất nhỏ và **không dùng để tuyên bố mô hình "tốt hơn"**; Δ thật sẽ đo bằng `split_group` ở Giai đoạn 6.<br>• **Augmentation là yếu tố tăng thời gian rõ nhất** (3,0 → 9,4 s/epoch do transform PIL trên CPU) và làm train acc thấp hơn val acc (gap âm) — dấu hiệu augmentation đang hoạt động, không phải overfit.<br>• **lr 3e-3 phân kỳ** (F1 0,567), lr 3e-4 tương đương 1e-3 ở 10 epoch; **class weight giảm nhẹ F1** (0,998914 so với 0,999471) nên không dùng cho mô hình cuối. |
> | Trạng thái | ☐ Đạt ☑ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | Yêu cầu 5.4 #4 ("E3 > E1 ≥ 2 điểm %") **không thể đạt bằng chữ** vì E1 đã 99,63 % (trần do rò rỉ frame, xem 5.7): E3 hơn E1 **0,31 điểm %**. Đã kiểm tra lại split: giao index train/val = 0, giao track = 0 với split group, nhưng **gần trùng frame** trong random split là 100 % — đây là lý do, không phải bug pipeline. Không chạm test. `configs/final.yaml` **chưa tạo** (thuộc Giai đoạn 6, sau khi đo Δ random vs group). `summary.csv` còn dòng `smoke_cnnb_s42`, sẽ loại ở Giai đoạn 9. |
>
> **Kiểm tra 5.7 — nghi ngờ rò rỉ khi E1 đã 99 %** (đã làm, bằng số):
> - `set(train_idx) ∩ set(val_idx) = ∅`; số track train/val = 1.045/262 với split group, giao track = 0.
> - Với **split random** (dùng cho E1–E7): **100 %** ảnh val có ít nhất một frame **cách ±1** cùng `track_key` nằm trong train (7.841/7.841). Đây là rò rỉ frame đã dự liệu trong kế hoạch → phải đo Δ bằng split group ở Giai đoạn 6 trước khi kết luận.
>
> ### 5.6.1. Nhật ký thực hiện (todos)
>
> | # | Todo | Trạng thái | Bằng chứng |
> |---|---|---|---|
> | 5.1 | Tạo 7 config E1–E4, E5a/E5b, E7 (chỉ khai báo phần khác `default.yaml`) | ☑ | `configs/e{1_cnna,2_cnnb,3_cnnb_aug,4_cnnc_gap_aug,5a_lr3e4,5b_lr3e3,7_classweight}.yaml` |
> | 5.2 | Chạy tuần tự 7 run, ghi thời gian thật | ☑ | `outputs/runs/*_s42/` (log trong `%TEMP%\run_*.log`) |
> | 5.3 | `python -m src.summarize --figure` | ☑ | `outputs/results/summary.csv` (10 dòng), `outputs/figures/05_experiment_comparison.png` |
> | 5.4 | Chọn mô hình theo val macro-F1 + ưu tiên ít tham số khi chênh < 0,3 điểm % | ☑ | chọn **E4** (0,999763 · 399.947 params) — lý do ghi ở bảng trên |
> | 5.5 | Kiểm tra 5.7 (nghi ngờ rò rỉ khi E1 ≥ 99 %) | ☑ | giao index = 0, nhưng 100 % val có frame ±1 trong train → ghi rõ |
> | 5.6 | Checklist 5.5 + điền 5.6 + 5.6.1 + bảng tiến độ, commit local | ☑ | `phase5_check.json` 11/11 PASS; commit local (không push) |
>
> **Tổng kết: 6/6 todo đạt.** Lệnh tái tạo: `python -m src.train --config configs/<exp>.yaml` cho từng config → `python -m src.summarize --figure --verify`.

### 5.7. Ghi chú / lưu ý

- Nếu E1 đã đạt 99%: nghi ngờ rò rỉ (val trùng train) → kiểm tra lại split và hash.
- Nếu E2 kém hơn E1: có thể do dropout quá mạnh hoặc lr chưa phù hợp; ghi nhận, không giấu.
- Nếu E3 kém hơn E2: augmentation có thể quá mạnh với GTSRB; thử giảm rotation/perspective, nhưng **phải chạy lại và báo cáo**, không sửa số.
- Không so sánh các run có số epoch khác nhau mà không ghi rõ; ưu tiên so sánh ở cùng epoch hoặc ở best epoch.
- Luôn ghi **thời gian/epoch** — nếu bất thường (ví dụ 2 phút/epoch), kiểm tra cache và DataLoader.
- Đừng chọn mô hình theo test. Test chưa được phép chạm ở giai đoạn này.

---

# GIAI ĐOẠN 6 — Chốt mô hình, multi-seed, đánh giá test 1 lần

**Ngày dự kiến:** D7 · **Thời lượng:** 4 giờ · **Gate:** G3

### 6.1. Mục tiêu
Đóng băng cấu hình cuối, chạy 3 seed để có mean ± std, và **đánh giá test đúng một lần**.

### 6.2. Giải thích
Một seed không đủ để khẳng định mô hình A hơn mô hình B. Ba seed cho phép báo cáo `mean ± std`, thể hiện hiểu biết về độ bất định — điểm cộng lớn trong phản biện.

Đánh giá test một lần là **kỷ luật khoa học**: nếu xem test nhiều lần rồi chọn cấu hình, test không còn là test độc lập nữa.

### 6.3. Việc cần làm

**Bước 6.3.1 — Đóng băng config**
- Copy config mô hình thắng → `configs/final.yaml`.
- Ghi vào đầu file: ngày chốt, lý do, run tham chiếu.
- Commit: `git commit -m "freeze: final config <id>"`.

**Bước 6.3.2 — Chạy multi-seed**
```powershell
python -m src.train --config configs/final.yaml --seed 42
python -m src.train --config configs/final.yaml --seed 123
python -m src.train --config configs/final.yaml --seed 2024
```

**Bước 6.3.3 — Tổng hợp multi-seed**
- Tính mean, std của val acc và val macro-F1.
- Ghi vào `outputs/results/multiseed.json`.

**Bước 6.3.4 — Đánh giá test một lần**
```powershell
python -m src.evaluate --run outputs/runs/<final>_s42 --split test --confirm-freeze
```
- Sinh `test_report.json` + `predictions.csv` + `confusion_matrix.png`.
- Ghi `hash(best.pt)` + commit hash vào report.

**Bước 6.3.5 — Kiểm tra tính hợp lý của kết quả test**
- So sánh test vs val: nếu test thấp hơn val > 3 điểm % → ghi nhận và giải thích (rò rỉ frame).
- Nếu test cao hơn val nhiều: kiểm tra lỗi tính metric.

### 6.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | `configs/final.yaml` | Có, đã commit, có ghi chú lý do |
| 2 | 3 seed hoàn tất | Đủ 3 thư mục run |
| 3 | mean ± std | Có trong `multiseed.json` |
| 4 | Test chạy đúng 1 lần | Log chỉ 1 entry cho split test |
| 5 | `test_report.json` | Có acc, macro-F1, per-class, params, hash |
| 6 | `predictions.csv` | Đúng 12.630 dòng |
| 7 | Chênh val–test | Được ghi và giải thích nếu > 2 điểm % |

### 6.5. Checklist test

- [ ] `git log` có commit "freeze".
- [ ] 3 run seed khác nhau, `config.yaml` trong mỗi run giống nhau trừ seed.
- [ ] `multiseed.json` có mean và std (không phải chỉ 1 giá trị).
- [ ] `test_report.json` chứa `checkpoint_sha256` và `git_commit`.
- [ ] `predictions.csv` có `len == 12630`.
- [ ] `predictions.csv` không có dòng trùng `path`.
- [ ] Confusion matrix: tổng mỗi hàng = 1 (sau chuẩn hóa) hoặc = support (không chuẩn hóa).
- [ ] Test acc nằm trong khoảng hợp lý (90–100%); nếu < 90% → điều tra.
- [ ] File `test_final.json` **không bị ghi đè** nếu chạy lại (kiểm tra timestamp).

### 6.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 6**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày chốt config | |
> | Mô hình cuối | |
> | Seed 42: val acc / F1 | |
> | Seed 123: val acc / F1 | |
> | Seed 2024: val acc / F1 | |
> | Val mean ± std (acc) | |
> | Val mean ± std (macro-F1) | |
> | **Test acc** | |
> | **Test macro-F1** | |
> | Chênh val–test | |
> | hash checkpoint | |
> | commit hash | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 6.7. Ghi chú / lưu ý

- Chỉ chạy test **sau** khi đã commit freeze. Nếu chưa, dừng lại.
- Không "chạy test thử" để xem kết quả. Mỗi lần chạy test phải được ghi log.
- Nếu phát hiện bug sau khi đã chạy test: sửa, chạy lại **toàn bộ từ đầu**, và ghi rõ trong báo cáo "test đã được đánh giá lại lần 2 do bug X" — trung thực hơn là giấu.
- `std` của macro-F1 thường lớn hơn std của accuracy ở lớp hiếm; đừng bỏ qua.
- Lưu `predictions.csv` ngay; đây là đầu vào cho toàn bộ Giai đoạn 7.

---

# GIAI ĐOẠN 7 — Đánh giá sâu & phân tích lỗi

**Ngày dự kiến:** D8–D9 · **Thời lượng:** 2 ngày · **Gate:** G4

### 7.1. Mục tiêu
Có đủ bảng/hình cho Chương 4–5: per-class report, confusion matrix, top cặp nhầm, error rate theo nguyên nhân, và **≥10 mẫu sai được phân tích thật**.

### 7.2. Giải thích
Đây là phần phân biệt bài "chạy được" với bài "hiểu vấn đề". Yêu cầu môn học là phân tích lỗi, nhưng phân tích tốt phải:
- **Định lượng** (bao nhiêu % lỗi rơi vào ảnh mờ/tối/biển nhỏ), không chỉ kể.
- **Phân biệt quan sát và giả thuyết**.
- **Chỉ ra giới hạn của mô hình**, không chỉ đổ lỗi dữ liệu.

### 7.3. Việc cần làm

**Bước 7.3.1 — Per-class report**
- `classification_report(y_true, y_pred, output_dict=True)`.
- Bảng 43 dòng: precision, recall, F1, support.
- Bar chart F1 sắp tăng dần, chú thích support.

**Bước 7.3.2 — Confusion matrix**
- Chuẩn hóa theo hàng, 43×43.
- Bản đầy đủ (heatmap) + bản zoom 15 lớp yếu nhất.
- Trích top 10 cặp `(true, pred)` có số lượng lớn nhất (bỏ đường chéo).

**Bước 7.3.3 — Định lượng theo nguyên nhân**
- Từ `predictions.csv`, chia tertile cho `blur_laplacian`, `brightness_mean`, `roi_area_ratio`.
- Bảng error rate 3×3 + biểu đồ.
- Kiểm tra tương quan (point-biserial hoặc so sánh trung bình) giữa lỗi và từng biến.

**Bước 7.3.4 — Chọn và phân tích 10+ mẫu sai**
- Ưu tiên mẫu có confidence cao nhưng sai (lỗi "tự tin sai").
- Ưu tiên mẫu đại diện cho các nhóm nguyên nhân khác nhau.
- Cắt ảnh theo ROI để hiển thị.
- Điền bảng: ảnh, nhãn thật, dự đoán, confidence, margin, quan sát, giả thuyết.

**Bước 7.3.5 — Phân tích OOD**
- Chạy model trên 10–15 ảnh tự chụp (Giai đoạn 8 chuẩn bị).
- Ghi dự đoán + confidence + đúng/sai.
- Thử ngưỡng unknown: rejection rate trên ảnh không có biển.

**Bước 7.3.6 — (Nếu kịp) E8: ROI crop**
- Train/eval trên ảnh crop theo ROI.
- So sánh với full image: accuracy, macro-F1, thời gian.
- Ghi nhận ảnh hưởng và lý do.

### 7.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Per-class report | Đủ 43 lớp |
| 2 | Confusion matrix | 43×43, chuẩn hóa, đọc được |
| 3 | Top cặp nhầm | ≥ 10 cặp kèm số lượng |
| 4 | Error rate theo nguyên nhân | 3 biến × 3 nhóm |
| 5 | Mẫu sai | ≥ 10, có ảnh thật, có nhận xét |
| 6 | OOD | ≥ 10 ảnh, có kết quả |
| 7 | Phân biệt quan sát/giả thuyết | Mỗi nhận xét ghi rõ loại |
| 8 | Lớp F1 thấp nhất | Được nêu tên + đối chiếu support |

### 7.5. Checklist test

- [ ] Bảng per-class có `support` cộng lại = 12.630.
- [ ] `confusion_matrix.png` mở được, nhãn trục đọc được.
- [ ] Top 10 cặp nhầm không chứa cặp `(i, i)`.
- [ ] Mỗi cặp nhầm có ≥ 3 ảnh ví dụ (nếu có).
- [ ] 3 biểu đồ error rate theo tertile có trục và số liệu.
- [ ] 10 ảnh sai: mở từng ảnh, xác nhận nhãn thật đúng theo `predictions.csv`.
- [ ] Trong bảng 10 mẫu, cột "quan sát" khác cột "giả thuyết".
- [ ] OOD: có ít nhất 3 ảnh không có biển báo, ghi rõ model dự đoán gì.
- [ ] Đã thử ngưỡng τ và ghi acceptance/rejection rate.

### 7.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 7**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Số lớp có F1 < 0,90 | |
> | Lớp F1 thấp nhất (ID, F1, support) | |
> | Top 1 cặp nhầm (số lượng) | |
> | Error rate nhóm mờ nhất vs nét nhất | |
> | Error rate nhóm tối nhất vs sáng nhất | |
> | Error rate nhóm biển nhỏ nhất vs lớn nhất | |
> | Số mẫu sai đã phân tích | |
> | OOD: số ảnh / số dự đoán đúng | |
> | Ngưỡng τ / acceptance in-dist / rejection OOD | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 7.7. Ghi chú / lưu ý

- Không suy diễn nguyên nhân từ 1 ảnh. Nếu nói "do mờ", phải có số liệu blur của ảnh đó so với trung bình.
- Ảnh ROI có thể rất nhỏ (vài chục pixel) → phóng to khi hiển thị nhưng ghi rõ "đã phóng to".
- Khi chọn 10 mẫu, tránh chọn toàn ảnh tối/ mờ; cần đa dạng nguyên nhân.
- Nếu lớp hiếm có F1 thấp, đừng kết luận ngay "model kém" — kiểm tra support: 30 ảnh val thì 1 lỗi đã làm F1 giảm mạnh.
- Với OOD, tuyệt đối không tuyên bố "model nhận diện được biển Việt Nam".

---

# GIAI ĐOẠN 8 — Demo dữ liệu mới

**Ngày dự kiến:** D10 · **Thời lượng:** 4–5 giờ · **Gate:** G4

### 8.1. Mục tiêu
Có app Streamlit chạy được với ảnh ngoài dataset, dùng **đúng** pipeline tiền xử lý, có xử lý "không xác định", có bộ ảnh tự chụp đã ẩn danh và manifest.

### 8.2. Giải thích
Demo là nơi dễ mất điểm nhất:
- Lệch preprocessing → dự đoán sai dù test 99%.
- Ảnh thật có nền rộng, biển nhỏ → khác hoàn toàn phân phối train.
- Model không có lớp "không biết" → luôn trả lời, kể cả ảnh không có biển.

Làm đúng ba điểm trên sẽ biến demo từ "trình diễn may rủi" thành "minh chứng hiểu biết".

### 8.3. Việc cần làm

**Bước 8.3.1 — Viết `src/predict.py`**
- `load_model(run_dir)`.
- `preprocess(image)` — **import từ `src.data`**, không viết lại.
- `predict_topk(image, k=3)` → list `(class_id, name_vi, prob)`.
- `should_reject(max_prob, tau)`.

**Bước 8.3.2 — Viết `app.py`**
- Upload JPG/JPEG/PNG.
- Hiển thị ảnh + top-3.
- Cảnh báo nếu max-prob < τ.
- Xử lý lỗi: file hỏng, > 20 MB, checkpoint thiếu.
- Footer cảnh báo sử dụng.

**Bước 8.3.3 — Thu thập ảnh ngoài dataset**
- 10–15 ảnh biển công cộng, ưu tiên biển giới hạn tốc độ/cấm.
- Không chụp người, không chụp biển số rõ nét.
- Crop quanh biển hoặc blur vùng nhạy cảm.
- Ghi `external_test/manifest.csv`: file, nguồn, ngày, mô tả, đã ẩn danh, nhãn kỳ vọng.

**Bước 8.3.4 — Thử 5 kịch bản**
1. Ảnh test chưa dùng train → đúng, top-3 hợp lý.
2. Ảnh tự chụp đã crop → kết quả + nhận xét.
3. Ảnh cảnh thật biển nhỏ → có thể sai, ghi nhận.
4. Ảnh không có biển → model vẫn gán nhãn; bật τ → "không xác định".
5. File lỗi / quá lớn → app báo lỗi, không crash.

**Bước 8.3.5 — Chụp ảnh minh chứng**
- Screenshot 5 kịch bản cho báo cáo/slide.

### 8.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | App chạy | `streamlit run app.py` không lỗi |
| 2 | Top-3 hiển thị | Đúng tên tiếng Việt + xác suất |
| 3 | Preprocess dùng chung | Có `from src.data import preprocess` |
| 4 | Ảnh ngoài dataset | ≥ 10 ảnh, có manifest |
| 5 | Ẩn danh | 100% ảnh không có mặt/biển số rõ |
| 6 | Ngưỡng unknown | Có, được thử trên ảnh không biển |
| 7 | Xử lý lỗi | 3 loại lỗi được xử lý |
| 8 | Footer cảnh báo | Có, hiển thị rõ |

### 8.5. Checklist test

- [ ] Upload ảnh test bất kỳ → app trả kết quả trong < 2 giây.
- [ ] Kết quả top-3 có tổng xác suất ≤ 1 và giảm dần.
- [ ] Tên lớp hiển thị đúng tiếng Việt (đối chiếu `class_names.json`).
- [ ] Upload file `.txt` đổi đuôi `.jpg` → app báo lỗi thân thiện.
- [ ] Upload ảnh > 20 MB → app báo lỗi, không treo.
- [ ] Đổi tên checkpoint → app báo "không tìm thấy checkpoint".
- [ ] Ảnh không có biển + τ bật → hiển thị "không xác định".
- [ ] `manifest.csv` có đủ số dòng = số ảnh trong `external_test/`.
- [ ] Mở từng ảnh trong `external_test/` kiểm tra ẩn danh.
- [ ] Footer có câu "không dùng cho quyết định giao thông thực tế".

### 8.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 8**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | |
> | Số ảnh ngoài dataset | |
> | Số ảnh đã ẩn danh | |
> | Kịch bản 1 (ảnh test): đúng/sai | |
> | Kịch bản 2 (ảnh crop): đúng/sai | |
> | Kịch bản 3 (cảnh thật): đúng/sai | |
> | Kịch bản 4 (không biển): dự đoán / bị từ chối | |
> | Ngưỡng τ | |
> | Thời gian phản hồi trung bình | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 8.7. Ghi chú / lưu ý

- Nếu app dự đoán sai ảnh tự chụp: **đừng sửa ảnh cho đến khi đúng**. Ghi nhận và phân tích — đó là dữ liệu quý cho Chương 5.
- Không dùng ảnh từ Internet mà không rõ giấy phép trong báo cáo.
- Ảnh điện thoại thường có EXIF chứa GPS/thời gian → xóa EXIF trước khi đưa vào repo.
- Nếu dùng detect HSV, ghi rõ đây là module phụ, không phải phần chính của bài.
- Ngưỡng τ phải chọn trên **val**, không chọn trên ảnh demo.

---

# GIAI ĐOẠN 9 — Tái lập & đóng gói

**Ngày dự kiến:** D11 · **Thời lượng:** 4 giờ · **Gate:** G5

### 9.1. Mục tiêu
Một người khác (hoặc chính bạn sau 1 tuần) có thể clone repo, làm theo README và tái lập kết quả.

### 9.2. Giải thích
"Chạy được trên máy tôi" không phải tái lập. Tái lập cần: phiên bản thư viện cố định, đường dẫn tương đối, dữ liệu tải được bằng lệnh, seed cố định, và **bằng chứng đã thử trong môi trường sạch**.

### 9.3. Việc cần làm

**Bước 9.3.1 — Đóng băng requirements**
```powershell
pip freeze > requirements.txt
```
- Kiểm tra không có package lạ, không có đường dẫn local.

**Bước 9.3.2 — Viết README**
Bắt buộc có:
1. Giới thiệu ngắn + kết quả chính.
2. Yêu cầu hệ thống (GPU, RAM, OS).
3. Cài đặt (đúng lệnh cu128).
4. Tải dữ liệu.
5. Train / evaluate / demo.
6. Cấu trúc thư mục.
7. Tái lập kết quả (seed, config).
8. Hạn chế + cảnh báo sử dụng.

**Bước 9.3.3 — Kiểm tra đường dẫn**
- `grep` toàn repo tìm `C:\`, `/home/`, `D:\` → sửa hết thành `pathlib` tương đối.

**Bước 9.3.4 — Clean-room test**
- Tạo venv mới `.venv_test`.
- Cài từ `requirements.txt`.
- Chạy: tải data (hoặc trỏ vào data có sẵn) → train smoke 2 epoch → evaluate val → app.
- Ghi lại lỗi gặp phải và sửa README.

**Bước 9.3.5 — Sinh `summary.csv` cuối**
- Đảm bảo mọi run cần báo cáo đều có mặt.
- Kiểm tra không có run rác (smoke) trong bảng.

### 9.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | `requirements.txt` | Cài được trong venv sạch, không lỗi |
| 2 | README | Đủ 8 mục, lệnh copy-paste chạy được |
| 3 | Không đường dẫn tuyệt đối | 0 kết quả grep |
| 4 | Clean-room | Chạy hết luồng không lỗi |
| 5 | `summary.csv` | Đủ run, không có NaN |
| 6 | `env.json` | Khớp môi trường thực tế |

### 9.5. Checklist test

- [ ] `python -m venv .venv_test` + cài requirements thành công.
- [ ] Trong venv sạch: `python -c "import torch; print(torch.cuda.is_available())"` = True.
- [ ] Chạy đúng lệnh trong README từ đầu đến cuối, không phải sửa gì.
- [ ] `grep -r "C:\\" src/ app.py` → 0 kết quả.
- [ ] `grep -r "/home/" src/ app.py` → 0 kết quả.
- [ ] `python -m src.evaluate --run <best> --split val` chạy lại cho cùng số (sai khác < 0,2%).
- [ ] `summary.csv` mở bằng Excel/pandas, không lỗi encoding.
- [ ] README có cảnh báo "không dùng cho quyết định thực tế".

### 9.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 9**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày / người thực hiện | |
> | Thời gian clean-room install | |
> | Lỗi gặp khi clean-room | |
> | Đã sửa gì | |
> | Số run trong summary.csv | |
> | Kết quả evaluate lại (val acc) | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 9.7. Ghi chú / lưu ý

- `pip freeze` trên Windows có thể sinh đường dẫn `file:///` cho package cài local; kiểm tra và loại.
- Đừng pin `torch` bản CPU trong requirements — ghi rõ index cu128 trong README.
- Nếu README cần 2 lệnh pip (torch riêng + phần còn lại), ghi rõ thứ tự.
- Clean-room test tốn thời gian nhưng là bằng chứng mạnh nhất cho yêu cầu "tái lập".

---

# GIAI ĐOẠN 10 — Báo cáo

**Ngày dự kiến:** D12–D13 · **Thời lượng:** 2 ngày · **Gate:** G5

### 10.1. Mục tiêu
Báo cáo 6 chương, mọi số liệu lấy từ `summary.csv` / `test_report.json` / `data_report.json` / `multiseed.json`.

### 10.2. Giải thích
Báo cáo không phải bản mô tả code. Mỗi chương phải trả lời một câu hỏi:
- Ch1: bài toán là gì, cho ai?
- Ch2: dữ liệu có đáng tin không?
- Ch3: phương pháp có hợp lý không?
- Ch4: kết quả có tốt hơn baseline không, vì sao?
- Ch5: mô hình sai ở đâu, vì sao?
- Ch6: dùng được đến đâu?

### 10.3. Việc cần làm

**Bước 10.3.1 — Dựng khung theo mục 18 của tài liệu chủ**
- Tạo file, dán tiêu đề chương/mục, chưa viết nội dung.
- Chèn sẵn placeholder cho bảng/hình.

**Bước 10.3.2 — Viết Ch1–Ch2 (D12 sáng)**
- Ch1: phát biểu bài toán, input/output, đối tượng, giá trị, giới hạn.
- Ch2: nguồn, license, số liệu thật, EDA, split, augmentation.

**Bước 10.3.3 — Viết Ch3 (D12 chiều)**
- Baseline, kiến trúc, bảng params, loss/optimizer/schedule, seed, phần cứng.
- Chèn sơ đồ kiến trúc (vẽ bằng draw.io/Matplotlib).

**Bước 10.3.4 — Viết Ch4 (D13 sáng)**
- Learning curves, bảng so sánh, ablation, test final, multi-seed.

**Bước 10.3.5 — Viết Ch5–Ch6 (D13 chiều)**
- Ch5: confusion, per-class, 10 mẫu, OOD.
- Ch6: demo, tái lập, hạn chế, hướng phát triển.

**Bước 10.3.6 — Rà soát số liệu**
- Mọi bảng: đối chiếu từng ô với file nguồn.
- Mọi hình: có caption + nguồn dữ liệu.

### 10.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Đủ 6 chương | Đúng cấu trúc |
| 2 | Mọi bảng có số thật | 0 ô "điền sau" |
| 3 | Mọi hình có caption | 100% |
| 4 | Có bảng ablation | ≥ 4 run CNN |
| 5 | Có 10 mẫu sai | Đủ, có ảnh |
| 6 | Có phần hạn chế | Nêu ≥ 3 hạn chế thật |
| 7 | Tài liệu tham khảo | ≥ 4 nguồn, đúng định dạng |

### 10.5. Checklist test

- [ ] Không còn chuỗi "TODO", "điền sau", "..." trong báo cáo.
- [ ] Số test acc trong Ch4 khớp `test_report.json`.
- [ ] Số 39.209/12.630 khớp `data_report.json`.
- [ ] Bảng ablation khớp `summary.csv` từng dòng.
- [ ] 10 ảnh trong Ch5 mở được, đúng nhãn.
- [ ] Hình curves có đơn vị trục, legend.
- [ ] Có câu "không dùng cho quyết định giao thông thực tế" ở Ch1 và Ch6.
- [ ] Đánh số hình/bảng liên tục, có tham chiếu trong văn bản.

### 10.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 10**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày hoàn thành | |
> | Số trang | |
> | Số hình / số bảng | |
> | Chương còn yếu | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 10.7. Ghi chú / lưu ý

- Viết Ch2 ngay từ D2 (khi số liệu còn mới), đừng đợi đến D12.
- Không copy kết quả từ paper vào bảng của mình; nếu trích, ghi rõ "tham khảo".
- Ảnh trong báo cáo: crop quanh biển, không lộ người/biển số.
- Dùng cùng một định dạng số (ví dụ 2 chữ số thập phân) cho toàn báo cáo.

---

# GIAI ĐOẠN 11 — Slide & phản biện

**Ngày dự kiến:** D14 · **Thời lượng:** 1 ngày · **Gate:** G5

### 11.1. Mục tiêu
12 slide khớp 100% số liệu báo cáo, trình bày trong 8–10 phút, trả lời được các câu phản biện cơ bản.

### 11.2. Giải thích
Slide không phải bản rút gọn của báo cáo. Mỗi slide một thông điệp. Con số trên slide phải trùng báo cáo — lệch số là lỗi bị phát hiện ngay.

### 11.3. Việc cần làm

**Bước 11.3.1 — Dựng 12 slide theo mục 19 tài liệu chủ**
**Bước 11.3.2 — Chèn hình từ `outputs/figures/`**
**Bước 11.3.3 — Chuẩn bị demo dự phòng**
- Quay video 60–90 giây các kịch bản demo.
- Lưu `demo_backup.mp4`.
**Bước 11.3.4 — Chuẩn bị Q&A**
- Dùng bảng 15 câu ở mục 20 tài liệu chủ.
- Chuẩn bị 1 slide phụ (không đưa vào bản chính) chứa bảng kết quả đầy đủ.

### 11.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Số slide | 10–12 |
| 2 | Thời lượng nói | 8–10 phút |
| 3 | Số liệu khớp báo cáo | 100% |
| 4 | Có hình confusion + curves | Có |
| 5 | Có demo dự phòng | Video tồn tại, phát được |
| 6 | Có cảnh báo sử dụng | Slide cuối |

### 11.5. Checklist test

- [ ] Đọc toàn bộ slide trong 8–10 phút (bấm giờ).
- [ ] Mỗi slide có ≤ 6 dòng chữ chính.
- [ ] Ảnh không bị méo, chữ không bị cắt khi trình chiếu.
- [ ] Số test acc trên slide = số trong báo cáo.
- [ ] Video demo phát được trên máy khác.
- [ ] Trả lời được 5 câu phản biện bất kỳ trong danh sách mà không cần mở báo cáo.

### 11.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 11**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày hoàn thành | |
> | Số slide | |
> | Thời lượng trình bày thử | |
> | Câu phản biện chưa trả lời tốt | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 11.7. Ghi chú / lưu ý

- Slide kết quả nên có 1 câu kết luận, không chỉ dán bảng.
- Đừng đưa ảnh dự đoán sai mà không có nhận xét — phản tác dụng.
- Chuẩn bị sẵn câu trả lời cho "tại sao 99% mà demo sai" — đó là câu gần như chắc chắn bị hỏi.

---

# GIAI ĐOẠN 12 — Nghiệm thu & nộp

**Ngày dự kiến:** D15 · **Thời lượng:** 4 giờ · **Gate:** G5

### 12.1. Mục tiêu
Đóng gói đủ artifact, chạy checklist nghiệm thu, nộp đúng hạn.

### 12.2. Giải thích
Bước cuối dễ sai vì mệt: thiếu file, sai đường dẫn, quên seed, nộp nhầm bản báo cáo cũ. Dùng checklist thay vì trí nhớ.

### 12.3. Việc cần làm

**Bước 12.3.1 — Kiểm tra artifact**
- Đối chiếu mục 21.3 tài liệu chủ.
- Mở từng file, xác nhận không rỗng/hỏng.

**Bước 12.3.2 — Nén bài**
- `BTL_GTSRB_CNN.zip` gồm: `src/`, `configs/`, `app.py`, `requirements.txt`, `README.md`, `outputs/results/`, `outputs/figures/`, `external_test/`, `report.docx`, `presentation.pptx`.
- **Không** đưa `data/`, `outputs/runs/*.pt` quá lớn vào zip nếu có giới hạn dung lượng; ghi link checkpoint riêng.

**Bước 12.3.3 — Kiểm tra lần cuối trên bản nén**
- Giải nén ra thư mục tạm, chạy README bước 1–3.
- Mở báo cáo và slide từ bản nén.

**Bước 12.3.4 — Nộp**
- Nộp đúng kênh, đúng hạn.
- Lưu bản sao + hash file nộp.

### 12.4. Yêu cầu cần đạt

| # | Yêu cầu | Ngưỡng |
|---|---|---|
| 1 | Đủ artifact | 100% mục 21.3 |
| 2 | Bản nén mở được | Giải nén không lỗi |
| 3 | README chạy từ bản nén | Bước 1–3 OK |
| 4 | Báo cáo + slide mở được | Không hỏng font |
| 5 | Không có dữ liệu cá nhân | 0 ảnh vi phạm |

### 12.5. Checklist test

- [ ] Tất cả ô ở mục 21.1, 21.2, 21.3 tài liệu chủ đã tick.
- [ ] `report.docx` không còn "TODO".
- [ ] `presentation.pptx` mở được trên máy khác.
- [ ] Giải nén bản nộp, chạy `python -m src.evaluate` với checkpoint có sẵn → ra số.
- [ ] Không có file `.venv`, `data/`, `*.pt` quá lớn trong zip (hoặc đã ghi rõ).
- [ ] Đã lưu hash file nộp + ảnh chụp màn hình xác nhận nộp.

### 12.6. Kết quả thực hiện

> **KẾT QUẢ THỰC HIỆN — GIAI ĐOẠN 12**
>
> | Hạng mục | Giá trị thực tế |
> |---|---|
> | Ngày nộp | |
> | Tên file nộp | |
> | Dung lượng | |
> | hash (SHA256) | |
> | Kênh nộp | |
> | Trạng thái | ☐ Đạt ☐ Đạt có điều kiện ☐ Chưa đạt |
> | Ghi chú | |

### 12.7. Ghi chú / lưu ý

- Nộp sớm hơn hạn ít nhất 2 giờ.
- Giữ 1 bản sao offline (USB/ổ ngoài) — mạng có thể lỗi đúng lúc nộp.
- Nếu giảng viên yêu cầu link checkpoint, kiểm tra quyền truy cập trước khi nộp.

---

# PHỤ LỤC

## Phụ lục A — Lệnh chạy nhanh (copy-paste)

```powershell
# 0. Môi trường
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn streamlit pyyaml tqdm pillow imagehash

# 1. Kiểm tra GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"

# 2. Dữ liệu
python -m src.data --download --report

# 3. Train
python -m src.train --config configs/e1_cnna.yaml
python -m src.train --config configs/e2_cnnb.yaml
python -m src.train --config configs/e3_cnnb_aug.yaml
python -m src.train --config configs/e4_cnnc_gap_aug.yaml

# 4. Chốt + test
python -m src.train --config configs/final.yaml --seed 42
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze

# 5. Tổng hợp + demo
python -m src.summarize
streamlit run app.py
```

## Phụ lục B — Ngưỡng cảnh báo (sanity thresholds)

Dùng để phát hiện lỗi pipeline sớm. Đây **không phải** mục tiêu điểm số.

| Kiểm tra | Ngưỡng hợp lý | Nếu vượt ngưỡng |
|---|---|---|
| 1 epoch (64×64, batch 128) | 10–25 giây | Kiểm tra cache, num_workers, AMP |
| Majority accuracy | 5–7% | Kiểm tra đếm lớp |
| HOG+SVC val acc | 90–97% | Kiểm tra HOG/scale/nhãn |
| E1 val acc | ≥ 90% | Kiểm tra split, nhãn, loss |
| E3 val acc | ≥ 97% | Kiểm tra aug, BN, lr |
| Test acc | 95–99,5% | < 95% → điều tra; > 99,8% → nghi rò rỉ |
| Chênh train–val acc | < 8 điểm % | > 8 → overfit, xem lại regularization |
| Chênh val–test acc | < 2 điểm % | > 2 → nghi rò rỉ frame |
| Tổng thời gian P0+P1 | 2–3 giờ | > 5 giờ → kiểm tra pipeline |

## Phụ lục C — Mẫu ghi kết quả (dùng chung)

```
Ngày:
Người:
Giai đoạn:
Việc đã làm:
Số liệu chính:
File sinh ra:
Test đã chạy:            (liệt kê)
Test fail:               (nếu có, ghi rõ)
Cách xử lý:
Trạng thái:  ☐ Đạt  ☐ Đạt có điều kiện  ☐ Chưa đạt
Ghi chú cho lần sau:
```

## Phụ lục D — Liên kết tài liệu

| Tài liệu | Đường dẫn |
|---|---|
| Kế hoạch tổng quan (bản chủ) | `docs/kehoach/KE_HOACH_TONG_QUAN_GTSRB_CNN.md` |
| Các bước thực hiện (tài liệu này) | `docs/kehoach/Cac_buoc_thuc_hien_GTSRB_CNN.md` |
| Kế hoạch ban đầu (tham khảo) | `docs/kehoach/Ke_hoach_hoan_thien_bai_tap_lon_GTSRB_CNN.docx` |
| Các bước ban đầu (tham khảo) | `docs/kehoach/Cac_buoc_phan_loai_bien_bao_GTSRB.docx` |

## Phụ lục E — Lịch sử thay đổi

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 1.0 | 22/10/2026 | Bản đầu: 13 giai đoạn (0–12), mỗi giai đoạn có mục tiêu, giải thích, việc cần làm, yêu cầu, checklist test, mẫu ghi kết quả, ghi chú; kèm 5 gate, bảng tiến độ, ngưỡng cảnh báo |
