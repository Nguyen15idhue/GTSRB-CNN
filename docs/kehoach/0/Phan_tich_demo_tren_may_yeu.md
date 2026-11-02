# PHÂN TÍCH: HUẤN LUYỆN TRÊN MÁY MẠNH — DEMO TRÊN MÁY YẾU HƠN

> **Tài liệu thay thế để chạy thực tế:** `docs/HUONG_DAN_CHAY_MAY_MOI.md` — hướng dẫn từng bước
> từ máy trắng (cài đặt → dữ liệu → train → demo → test), dễ làm theo hơn tài liệu phân tích dưới đây.
> Tài liệu này giữ lại để tham khảo phân tích đóng gói/demo trên máy yếu.


| Thông tin | Giá trị |
|---|---|
| Môn học | Học sâu — 12INF028 |
| Đề tài | Phân loại biển báo giao thông Đức (GTSRB) bằng CNN tự xây dựng |
| Tài liệu liên quan | `KE_HOACH_TONG_QUAN_GTSRB_CNN.md`, `Cac_buoc_thuc_hien_GTSRB_CNN.md` |
| Phiên bản | 1.0 — 22/10/2026 |
| Nội dung | Đánh giá khả năng triển khai demo trên máy cấu hình thấp; phương án đóng gói, quy trình chuyển máy, kiểm chứng và xử lý sự cố |
| Kết luận | **Khả thi.** Huấn luyện cần GPU, demo chỉ cần CPU. Rủi ro nằm ở môi trường và tính nhất quán tiền xử lý, không ở tốc độ |

---

## 1. Kết luận ngắn

Bài toán tách thành hai phần có yêu cầu phần cứng rất khác nhau:

| Phần | Nơi chạy | Yêu cầu tài nguyên |
|---|---|---|
| Huấn luyện + ablation | Máy mạnh (RTX 5070) | GPU 12 GB, 64 GB RAM, ~2–3 giờ GPU |
| **Demo / suy luận** | Máy yếu | **CPU, ~1 GB RAM, không cần GPU, không cần dataset** |

Mô hình CNN khoảng **2,2 triệu tham số**, đầu vào 64×64 → mỗi ảnh chỉ vài triệu phép tính. Máy yếu vẫn suy luận được trong khoảng 0,1–0,3 giây/ảnh.

Ba điểm quyết định thành bại:

1. **Môi trường**: cài đúng PyTorch bản **CPU**, đúng phiên bản Python 64-bit.
2. **Nhất quán tiền xử lý**: demo phải dùng đúng hàm `preprocess` của lúc train.
3. **Kiểm chứng**: so sánh kết quả giữa hai máy trên cùng bộ ảnh (golden test).

---

## 2. Máy demo cần gì và không cần gì

### 2.1. Artifact cần mang theo

| Artifact | Kích thước ước tính | Vai trò |
|---|---|---|
| `best.pt` (chỉ `state_dict`) | ~9 MB (2,2 M × 4 byte) | Trọng số mô hình |
| `deploy_config.json` | vài KB | `image_size`, `mean`, `std`, kiến trúc, mapping |
| `class_names.json` | vài KB | 43 nhãn + tên tiếng Việt |
| `src/models.py` + `src/preprocess.py` + `src/predict.py` | vài KB | Định nghĩa mạng và tiền xử lý |
| `app.py` | vài KB | Giao diện Streamlit |
| `requirements-demo.txt` | vài KB | Chỉ thư viện cần cho suy luận |
| `samples/` (10–15 ảnh) | vài MB | Ảnh demo chưa có trên máy đích |
| `golden.json` | vài KB | Kết quả chuẩn để đối chiếu giữa hai máy |
| `README-DEMO.md` | vài KB | Lệnh chạy và xử lý sự cố |

Tổng dung lượng bundle (không tính môi trường Python): **< 50 MB**.

### 2.2. Không cần mang theo

- Bộ GTSRB đầy đủ (12.630 ảnh test, ~250 MB) — trừ khi muốn chạy lại đánh giá trên máy demo.
- GPU, CUDA, driver NVIDIA.
- Code huấn luyện, `sklearn`, `matplotlib`, `seaborn`, notebook.
- `outputs/runs/` đầy đủ, log trung gian, checkpoint của các ablation.

### 2.3. Có thể bỏ thêm để nhẹ hơn

| Thành phần | Cách bỏ | Lợi ích |
|---|---|---|
| `torchvision` | Viết `preprocess` bằng PIL + NumPy + torch | Giảm ~vài trăm MB cài đặt |
| `models.py` | Export TorchScript (`torch.jit.script`) | 1 file duy nhất, không lệch code |
| PyTorch hoàn toàn | Export ONNX + `onnxruntime` | Cài đặt ~50 MB, chạy nhanh trên CPU yếu |

---

## 3. Yêu cầu phần cứng và tốc độ thực tế

### 3.1. Cấu hình tối thiểu

| Thông số | Tối thiểu | Thoải mái |
|---|---|---|
| CPU | 2 nhân, 64-bit | 4 nhân trở lên |
| RAM | 2 GB (CLI) / 4 GB (Streamlit + trình duyệt) | 8 GB |
| Ổ đĩa trống | ~1,5 GB (PyTorch CPU + thư viện) | 3 GB |
| Hệ điều hành | Windows 10 64-bit | Windows 10/11 |
| Python | 3.10–3.12, **64-bit** | 3.12 |
| Màn hình | 1280×720 | 1920×1080 |
| Mạng | Không bắt buộc nếu cài offline | Có để cài trực tiếp |

### 3.2. Ước tính tốc độ (cần đo thực tế)

| Máy | Thời gian/ảnh (CPU) | Khởi động app |
|---|---|---|
| i9-14900KF | ~5–15 ms | 5–8 giây |
| Laptop i5 4 nhân | ~20–50 ms | 8–12 giây |
| Máy yếu 2 nhân / Celeron | ~100–300 ms | 15–25 giây |

Ngay cả máy yếu nhất vẫn dưới 1 giây/ảnh → không bị coi là "demo chậm". Nếu chậm hơn, nguyên nhân thường là trình duyệt/Streamlit chứ không phải mô hình.

---

## 4. Bốn phương án demo

| # | Phương án | Yêu cầu ở máy demo | Ưu điểm | Nhược điểm | Khi nào dùng |
|---|---|---|---|---|---|
| **A** | Copy bundle + venv CPU + `pip install` | Python 64-bit; internet hoặc wheels offline | Đơn giản, giữ nguyên code, dễ sửa | Tải PyTorch ~200–250 MB | **Mặc định** |
| **B** | Bundle offline: Python embeddable + wheels | Không cần cài, không cần internet | Chạy ngay, không cần quyền admin | Chuẩn bị lâu, phải khớp phiên bản Python | Máy không có mạng / bị khóa quyền |
| **C** | Export ONNX + `onnxruntime` + app nhỏ | Python hoặc file thực thi nhỏ | Nhẹ (~50 MB), nhanh, không phụ thuộc PyTorch | Cần kiểm tra khớp output | Máy rất yếu, ổ đĩa chật |
| **D** | Deploy web (Streamlit Cloud / HuggingFace Spaces) | **Chỉ cần trình duyệt** | Máy yếu nhất vẫn demo được | Cần internet; mô hình công khai | Máy quá yếu / bị khóa / demo từ xa |
| **E** | Video quay sẵn 60–90 giây | Bất kỳ | Luôn chạy được | Không phải demo trực tiếp | Phương án cuối |

### 4.1. Điều tuyệt đối tránh

**Không copy nguyên thư mục `.venv` sang máy khác.** `pyvenv.cfg` trỏ tới Python gốc và nhiều script chứa đường dẫn tuyệt đối → gần như chắc chắn lỗi. Luôn cài lại từ `requirements` hoặc dùng Python embeddable.

### 4.2. Sơ đồ quyết định

```
Máy demo có Python 64-bit + quyền cài + internet?
├── Có  → Phương án A (venv CPU + Streamlit)
└── Không
    ├── Có quyền chạy file, không cần cài?
    │   ├── Có  → Phương án B (embeddable + wheels offline)
    │   └── Không → Phương án C (ONNX + app nhỏ) hoặc D (web)
    └── Máy quá yếu / bị khóa hoàn toàn → Phương án D (web) + E (video)
```

---

## 5. Quy trình 6 bước chuyển từ máy mạnh sang máy yếu

### Bước 1 — Trên máy mạnh: tạo `deploy/`

Cấu trúc khuyến nghị:

```
deploy/
├── best.pt                 # state_dict thuần, KHÔNG pickle cả class model
├── deploy_config.json      # image_size, mean, std, kiến trúc, class mapping
├── class_names.json
├── src/
│   ├── models.py           # chỉ định nghĩa mạng
│   ├── preprocess.py       # chỉ hàm tiền xử lý, không import datasets/sklearn
│   └── predict.py          # load_model + predict_topk
├── app.py
├── requirements-demo.txt   # torch (CPU), pillow, numpy, streamlit, pyyaml
├── samples/                # 10–15 ảnh demo + 20 ảnh golden test
├── golden.json
└── README-DEMO.md
```

Hai quyết định kỹ thuật quan trọng:

1. **Lưu `state_dict` thay vì pickle toàn bộ model**: tránh lỗi khi phiên bản PyTorch khác nhau. Khi load:
   ```python
   ckpt = torch.load("best.pt", map_location="cpu", weights_only=True)
   model.load_state_dict(ckpt["state_dict"])
   ```
2. **Tách `preprocess.py` khỏi `data.py`**: file này không được import `torchvision.datasets`, `sklearn`, `matplotlib`. Càng ít phụ thuộc, càng dễ cài trên máy yếu.

**Phương án gọn hơn:** export TorchScript (`torch.jit.script(model)`) → một file `model_scripted.pt` chứa cả kiến trúc và trọng số; máy demo không cần `models.py`.

### Bước 2 — Tự giả lập máy yếu ngay trên máy mạnh

```powershell
python -m venv .venv_cpu
.\.venv_cpu\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy\requirements-demo.txt
```

Ép chạy CPU trong app/script (`device = "cpu"`, `map_location="cpu"`), dự đoán 5 ảnh.

**Nếu bước này fail thì sang máy khác chắc chắn fail.** Đây là "cổng chặn" quan trọng nhất.

### Bước 3 — Golden test (bắt buộc)

Mục đích: chứng minh hai máy cho **cùng kết quả** với cùng đầu vào.

1. Chọn 20 ảnh cố định, đặt trong `deploy/samples/golden/`.
2. Trên máy mạnh (GPU): chạy và lưu `golden.json`:
   ```json
   {
     "00001.png": {"pred_id": 12, "prob": 0.9987, "top3": [[12, 0.9987], [13, 0.0008], [38, 0.0003]]}
   }
   ```
3. Trên môi trường CPU (Bước 2) và trên máy đích: chạy lại, so sánh.
4. Tiêu chí đạt: **cùng nhãn**, xác suất lệch **< 1e-3**.

Nếu lệch nhiều, kiểm tra theo thứ tự:
- Quên `model.eval()`.
- Thiếu/khác normalize (`mean`, `std`).
- Khác interpolation khi resize (bilinear vs nearest).
- Sai thứ tự kênh (RGB vs BGR).
- Khác phiên bản PyTorch gây khác biệt số học (thường rất nhỏ, < 1e-4).

### Bước 4 — Đóng gói và chuyển

- Nén `deploy/` + `golden.json` thành `demo_bundle.zip`.
- Chuyển bằng USB / Git / cloud.

Nếu máy đích **không có internet**, chuẩn bị wheels trên máy mạnh:

```powershell
pip download -r deploy\requirements-demo.txt -d wheels `
  --only-binary=:all: --platform win_amd64 --python-version 3.12
```

Sau đó trên máy đích:

```powershell
pip install --no-index --find-links wheels -r requirements-demo.txt
```

**Bắt buộc khớp:** Windows 64-bit, đúng minor version Python (3.12 ≠ 3.11), kiến trúc `win_amd64`.

### Bước 5 — Cài và chạy trên máy yếu

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-demo.txt
streamlit run app.py
```

Lần chạy đầu, Windows Firewall sẽ hỏi → chọn **Allow** (ưu tiên Private network). Streamlit mở `http://localhost:8501`.

Nếu muốn máy khác trong cùng mạng truy cập:

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Bước 6 — Kiểm tra tại chỗ trước khi demo thật

- [ ] Chạy golden test trên máy yếu, so với `golden.json`.
- [ ] Thử 5 kịch bản: ảnh test, ảnh crop, ảnh cảnh thật, ảnh không có biển, file lỗi.
- [ ] Đo thời gian phản hồi thực tế.
- [ ] Chụp ảnh màn hình máy demo (bằng chứng cho báo cáo).
- [ ] Thử tắt mạng → app vẫn chạy (nếu dùng local).
- [ ] Thử khởi động lại máy → chạy lại đúng lệnh.

---

## 6. Lỗi thường gặp trên máy yếu và cách xử lý

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `no kernel image is available for execution on the device` | Cài nhầm PyTorch bản CUDA trên máy không GPU/GPU cũ | Cài bản CPU: `--index-url https://download.pytorch.org/whl/cpu` |
| `DLL load failed while importing _C` | Thiếu Microsoft Visual C++ Redistributable | Cài VC++ 2015–2022 x64 |
| `ModuleNotFoundError: sklearn` / `torchvision.datasets` | App import nhầm module dùng cho train | Tách `preprocess.py`, không import `data.py` đầy đủ |
| `streamlit: command not found` | Chưa activate venv | Dùng `python -m streamlit run app.py` |
| Cổng 8501 đã bị chiếm | Ứng dụng khác đang chạy | `--server.port 8502` |
| App mở rất chậm, treo | RAM thấp + nhiều tab trình duyệt | Đóng tab; dùng Edge; hoặc chuyển app Tkinter/CLI |
| Dự đoán khác máy mạnh | Tiền xử lý lệch | Golden test; kiểm tra `eval()`, mean/std, interpolation, thứ tự kênh |
| Antivirus chặn `.pt` / `.exe` | False positive | Nén kèm mật khẩu; hoặc dùng ONNX |
| Không có quyền cài Python | Máy trường/công ty bị khóa | Phương án B (embeddable) hoặc D (web) |
| Python 32-bit | Cài nhầm bản | Gỡ, cài bản 64-bit |
| `torch.load` lỗi `weights_only` | Phiên bản torch khác nhau | Lưu `state_dict` thuần; chỉnh cờ `weights_only` cho phù hợp |
| Màn hình projector bị cắt | Độ phân giải thấp | Thu nhỏ cửa sổ, dùng `--server.headless true` và trình duyệt full screen |

---

## 7. Ảnh hưởng đến yêu cầu môn học

Yêu cầu của môn là "demo chạy trên **dữ liệu chưa xuất hiện trong tập huấn luyện**", **không** yêu cầu demo phải chạy trên máy đã huấn luyện. Demo trên máy khác vẫn hợp lệ nếu:

1. Dùng **đúng checkpoint cuối** (ghi hash SHA-256 trong báo cáo).
2. Dùng **đúng pipeline tiền xử lý** (chứng minh bằng golden test).
3. Ảnh demo **không nằm trong train/val/test** (ảnh tự chụp).
4. Ghi rõ trong báo cáo: cấu hình máy demo, thời gian suy luận, kết quả.

Câu nên đưa vào Chương 6:

> "Mô hình được huấn luyện trên máy A (Intel i9-14900KF, RTX 5070 12 GB) và demo trên máy B (CPU, không GPU). Hai máy dùng cùng checkpoint và cùng pipeline tiền xử lý; sai khác xác suất trên bộ 20 ảnh kiểm chứng nhỏ hơn 1e-3."

Chi tiết này thể hiện tính tái lập — được đánh giá cao khi phản biện.

---

## 8. Khuyến nghị áp dụng

### 8.1. Ba lớp dự phòng

| Lớp | Nội dung | Khi nào dùng |
|---|---|---|
| **Chính** | Phương án A: `deploy/` + venv CPU + Streamlit | Mặc định |
| **Dự phòng 1** | TorchScript 1 file + `predict.py` CLI in top-3 | Streamlit/trình duyệt có vấn đề |
| **Dự phòng 2** | Video 60–90 giây + ảnh chụp màn hình | Máy hỏng, hết pin, bị khóa |

Nếu máy demo rất yếu hoặc không cho cài đặt: thêm **Phương án D** (Streamlit Community Cloud / HuggingFace Spaces). Mô hình ~9 MB, dưới giới hạn 100 MB của GitHub.

### 8.2. Năm việc nên làm ngay từ lúc bắt đầu code

1. Tách `preprocess` thành module riêng, không phụ thuộc `torchvision.datasets`.
2. Lưu `state_dict` + `deploy_config.json` ngay từ run đầu tiên.
3. Thêm cờ `--device cpu` cho `predict.py` và app.
4. Viết `requirements-demo.txt` riêng, không dùng chung `requirements.txt` đầy đủ.
5. Sinh `golden.json` ngay khi có checkpoint cuối (mất khoảng 2 phút).

Làm 5 việc này từ đầu thì việc chuyển sang máy yếu chỉ còn là **copy → cài → chạy**, khoảng 15–30 phút.

### 8.3. Ước tính thời gian chuẩn bị

| Công việc | Thời gian |
|---|---|
| Tạo `deploy/`, tách module, export TorchScript | 30–45 phút |
| Test trên venv CPU + golden test | 20–30 phút |
| Đóng gói + chuyển | 10 phút |
| Cài và kiểm tra trên máy đích | 15–30 phút |
| **Tổng** | **~1,5–2 giờ** |

---

## 9. Checklist tổng hợp

### 9.1. Trước khi rời máy mạnh

- [ ] `deploy/best.pt` load được bằng `weights_only=True`.
- [ ] `deploy_config.json` có đủ `image_size`, `mean`, `std`, kiến trúc, mapping.
- [ ] `class_names.json` đủ 43 lớp.
- [ ] `preprocess.py` không import `torchvision.datasets`, `sklearn`, `matplotlib`.
- [ ] `requirements-demo.txt` cài được trong venv sạch.
- [ ] App chạy được với `device="cpu"`.
- [ ] Golden test 20 ảnh: nhãn khớp 20/20, xác suất lệch < 1e-3.
- [ ] Bundle < 50 MB (không tính wheels).
- [ ] Có `README-DEMO.md` với lệnh copy-paste.
- [ ] Có video dự phòng.

### 9.2. Trên máy demo

- [ ] Python 64-bit đúng phiên bản.
- [ ] `torch.cuda.is_available()` = False nhưng app vẫn chạy.
- [ ] Golden test khớp.
- [ ] 5 kịch bản demo chạy đúng như mong đợi.
- [ ] Thời gian phản hồi < 1 giây/ảnh.
- [ ] Không có cảnh báo bảo mật/dữ liệu cá nhân trên ảnh demo.
- [ ] Có ảnh chụp màn hình máy demo để đưa vào báo cáo.

---

## 10. Câu hỏi thường gặp

**Máy demo không có GPU thì kết quả có khác không?**
Không đáng kể. Cùng trọng số, cùng tiền xử lý → cùng nhãn. Xác suất có thể lệch ở mức 1e-5–1e-4 do khác biệt số học CPU/GPU.

**Có cần mang theo dataset không?**
Không, trừ khi muốn chạy lại đánh giá test trên máy đó. Demo chỉ cần ảnh do người dùng tải lên.

**Có thể demo bằng cách mở web công khai không?**
Được, và là phương án tốt nếu máy yếu. Lưu ý: mô hình và app sẽ công khai; không đưa dữ liệu cá nhân lên.

**Nếu máy demo chỉ có 2 GB RAM?**
Dùng CLI (`predict.py`) thay vì Streamlit + trình duyệt. Streamlit + Chrome có thể chiếm 800 MB–1,2 GB.

**Nếu checkpoint quá lớn để gửi qua chat/email?**
~9 MB là bình thường. Nếu có nhiều file, nén lại hoặc dùng Git LFS.

**Có được tính là "demo trên dữ liệu mới" khi chạy trên máy khác?**
Được, miễn là ảnh demo chưa từng dùng để huấn luyện và checkpoint là bản cuối. Nên ghi rõ trong báo cáo.

---

## 11. Lịch sử thay đổi

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| 1.0 | 22/10/2026 | Phân tích khả năng demo trên máy yếu: yêu cầu tài nguyên, 4 phương án, quy trình 6 bước, golden test, xử lý sự cố, checklist |
