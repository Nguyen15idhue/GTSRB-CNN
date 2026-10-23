# CHẠY NHANH — DEMO STREAMLIT (KHÔNG TRAIN)

Một file duy nhất để mở demo từ máy mới. **Không train, không tải GTSRB, không cần GPU.**
Khoảng **5 phút**.

> File này gộp hai bản quickstart cũ (bản bundle và bản git clone) thành một.
> Đã chạy thật và kiểm chứng ngày 23/10/2026 (xem mục 6).
>
> Cần chạy đầy đủ (data → train → eval → test): xem `docs/HUONG_DAN_CHAY_MAY_MOI.md`.

---

## 1. Chọn cách lấy mã nguồn

| Cách | Khi nào dùng | Cần thêm |
|---|---|---|
| **A. git clone** | Có mạng, muốn mã nguồn đầy đủ | Không; repo đã kèm `checkpoints/final_s42/best.pt` (8,4 MB) |
| **B. Bundle** | Máy đích không có Git/mạng, chỉ cần demo | Giải nén `GTSRB_demo_bundle.zip` (~15 MB) từ máy có model |

Cả hai cách đều dùng chung các bước cài đặt và chạy ở mục 3–4.

### A. git clone

```powershell
git clone https://github.com/Nguyen15idhue/GTSRB-CNN.git
cd GTSRB-CNN
```

### B. Bundle

```powershell
# Trên máy đang có model (tuỳ chọn, nếu chưa có file zip):
python tests/package_demo.py     # → outputs/GTSRB_demo_bundle.zip

# Máy đích: giải nén rồi mở PowerShell tại thư mục vừa giải nén
```

Bundle chứa `app.py`, `src/`, checkpoint, `threshold.json`, `test_report.json`, `multiseed.json`,
`configs/class_names.json`, báo cáo DOCX, 16 ảnh demo và ảnh mẫu.

---

## 2. Yêu cầu

| Thành phần | Giá trị |
|---|---|
| OS | Windows 10/11 + PowerShell (Linux/macOS xem mục 8) |
| Python | **3.12 64-bit** (chuẩn của repo); đã chạy thật với **3.13.5** |
| GPU / CUDA | **Không cần** — torch CPU là đủ |
| Dữ liệu GTSRB | **Không cần** (chỉ cần nếu train/eval) |
| Dung lượng | ~1 GB cho `.venv`, ~250 MB tải về |

Kiểm tra Python:

```powershell
py -3.12 --version     # nếu không có 3.12, dùng bản đang có, ví dụ: py -3.13 --version
```

---

## 3. Cài đặt (một lần)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
py -3.12 -m venv .venv                 # hoặc py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-demo.txt   # torch CPU ~115 MB + streamlit, không cần CUDA
```

- **Không** cài `requirements.txt` (bản đó dành cho train, có torchvision/CUDA).
- **Không** cài torch từ index CUDA cho demo.
- Dấu hiệu đúng: đầu dòng lệnh có `(.venv)`.

> `requirements-demo.txt` ghim `torch==2.11.0` bản CPU trên PyPI; không cần GPU.

---

## 4. Chạy Streamlit

```powershell
streamlit run app.py
```

Mở **http://localhost:8501**. Dừng: `Ctrl+C`.

- Lần đầu Streamlit có thể hỏi **Email** → nhấn **Enter** (bỏ trống).
- Muốn bỏ hẳn bước hỏi email (dùng khi chạy nền/script):

```powershell
python -m streamlit run app.py --server.headless true
```

- Không nhận lệnh `streamlit`: dùng `python -m streamlit run app.py`.
- Cổng bận: thêm `--server.port 8502`.

### Kiểm tra nhanh trước khi mở trình duyệt (tuỳ chọn)

```powershell
# 1) Môi trường
python -c "import torch, streamlit; print(torch.__version__, streamlit.__version__)"

# 2) Nạp checkpoint + dự đoán 1 ảnh mẫu (không train)
$env:PYTHONIOENCODING="utf-8"
python -c "from PIL import Image; from src.predict import Predictor, load_tau; from src.config import PROJECT_ROOT; tau=load_tau(); p=Predictor(PROJECT_ROOT/'checkpoints'/'final_s42', tau=tau); r=p.predict(Image.open('external_test/ext_01_gtsrb_test_00000.png').convert('RGB'), k=3, tau=tau); print(r['topk'][0]['class_id'], r['topk'][0]['name_en'], round(r['topk'][0]['prob'], 4))"
```

Kỳ vọng: `16 Vehicles > 3.5 t prohibited 1.0`.

---

## 5. Demo trên giao diện

| Chọn ảnh | Kết quả mong đợi |
|---|---|
| “Ảnh test GTSRB — biển 16” | Cấm xe trên 3,5 tấn (conf ~1,00) |
| “Biển Hàn Quốc 30 km/h” | Giới hạn tốc độ 30 km/h |
| “Cảnh đường không có biển” | **Không xác định** |
| Upload biển Việt Nam | **Không xác định** hoặc sai — model chỉ biết 43 lớp biển ĐỨC |

Sidebar hiển thị: accuracy **98,56 %**, macro-F1 **0,9799**, top-3 **99,64 %**, τ = **0,9462**.

> **Lưu ý khi chưa có `data/`:** app tự ẩn ảnh mẫu không tồn tại, nên chỉ còn **3** mục
> (`ext_01`, biển Hàn Quốc, cảnh đường) thay vì 5. Đây là hành vi đúng, không phải lỗi.
> Bộ 16 ảnh PNG (`outputs/demo_images/`) không có trong repo; xem `docs/DEMO_DATA.md` nếu cần tạo.

---

## 6. Đã kiểm chứng thực tế (23/10/2026)

| Mục | Kết quả |
|---|---|
| Môi trường | Python 3.13.5 · torch 2.11.0+cpu · streamlit 1.66.1 |
| Checkpoint | `checkpoints/final_s42/best.pt`, sha256 `9c4dd0d6…be33` (khớp `checkpoints/README.md`) |
| Health check | `http://localhost:8501/_stcore/health` → `ok` |
| Chạy script | `streamlit.testing.v1.AppTest`: **0 exception, 0 error** |
| Sidebar | 98,56 % · 0,9799 · 99,64 % |
| Dự đoán ảnh mẫu | lớp 16 “Vehicles > 3.5 t prohibited”, xác suất 1,0000 |

---

## 7. Lỗi thường gặp

| Lỗi / triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Streamlit hỏi **Email** | onboarding lần đầu | nhấn Enter, hoặc chạy `--server.headless true` |
| `streamlit` không nhận lệnh | chưa activate `.venv` / chưa cài | activate lại, `pip install -r requirements-demo.txt` |
| `Activate.ps1 cannot be loaded` | PowerShell chặn script | `Set-ExecutionPolicy -Scope Process Bypass` rồi activate lại |
| `Không tìm thấy checkpoint` | thiếu `best.pt` | kiểm tra `checkpoints/final_s42/best.pt`; hoặc đặt `GTSRB_RUN_DIR` |
| `Port 8501 is already in use` | cổng bận | `streamlit run app.py --server.port 8502` |
| `ConnectionResetError: [WinError 10054]` | asyncio Windows Proactor khi health-check đóng kết nối | **vô hại**, không phải lỗi app; bỏ qua |
| `UnicodeEncodeError: 'charmap' codec…` khi in tiếng Việt | console dùng cp1258 | `$env:PYTHONIOENCODING="utf-8"` |
| `ModuleNotFoundError: No module named 'src'` | chạy sai thư mục | `cd` về thư mục gốc repo (có `src/`, `app.py`) |

### App tìm checkpoint theo thứ tự nào?

Theo `app.py` (đúng như code, không phải như mô tả cũ):

1. Biến môi trường `GTSRB_RUN_DIR` (nếu đặt).
2. `outputs/runs/final_s42/best.pt` (nếu đã train).
3. `checkpoints/final_s42/best.pt` (có sẵn sau khi clone).

Dùng checkpoint khác:

```powershell
$env:GTSRB_RUN_DIR = "D:\checkpoints\final_s42"
streamlit run app.py
```

---

## 8. Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
streamlit run app.py
```

macOS không có CUDA nhưng demo CPU chạy bình thường.

---

## 9. Không làm gì trong quy trình này

- **Không** chạy `python -m src.train` (train lại là ngoài phạm vi tài liệu này).
- **Không** chạy `python -m src.evaluate --split test` (test đã dùng đúng 1 lần, bị từ chối).
- **Không** tải `data/` nếu chỉ demo.
- **Không** sửa `preprocess`, `class_names.json` hay `threshold.json` để “chỉnh” kết quả demo.

---

## 10. Liên kết

- Hướng dẫn đầy đủ (data → train → eval → test): `docs/HUONG_DAN_CHAY_MAY_MOI.md`
- Vị trí ảnh val/test, xuất ảnh demo: `docs/DEMO_DATA.md`
- Báo cáo: `docs/bao_cao/BAO_CAO.md` / `.docx`
- Quy tắc làm việc: `AGENTS.md`
