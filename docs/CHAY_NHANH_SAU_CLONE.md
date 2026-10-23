# CHẠY NHANH SAU KHI CLONE (không train)

Máy mới → `git clone` → cài thư viện → mở Streamlit demo.
**Không train, không tải GTSRB, không cần GPU.** Khoảng **5 phút**.

> Repo **đã kèm checkpoint** tại `checkpoints/final_s42/best.pt` (8,4 MB) nên clone xong
> không cần copy gì thêm. App tự tìm checkpoint theo thứ tự:
> `outputs/runs/final_s42` → `checkpoints/final_s42` → biến môi trường `GTSRB_RUN_DIR`.

---

## 1. Chuẩn bị

- **Python 3.12 64-bit** — kiểm tra: `py -3.12 --version`
- Git (để clone)

## 2. Clone + cài (torch CPU, ~1 GB)

```powershell
git clone https://github.com/Nguyen15idhue/GTSRB-CNN.git
cd GTSRB-CNN
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
```

- Không cần CUDA, không cần `requirements.txt` (bản đó dành cho train).
- PowerShell chặn script: `Set-ExecutionPolicy -Scope Process Bypass` rồi activate lại.

## 3. Chạy

```powershell
streamlit run app.py
```

- Lần đầu hỏi **Email** → nhấn **Enter**.
- Mở **http://localhost:8501**. Dừng: `Ctrl+C`.
- Không nhận lệnh `streamlit`: dùng `python -m streamlit run app.py`.

## 4. Demo trong 2 phút

| Chọn ảnh | Kết quả mong đợi |
|---|---|
| “Ảnh test GTSRB — biển 16” | Cấm xe trên 3,5 tấn |
| “biển 50 km/h” / “STOP” | Giới hạn tốc độ 50 km/h / Dừng lại |
| “Biển Hàn Quốc 30 km/h” | Giới hạn tốc độ 30 km/h |
| “Cảnh đường không có biển” | **Không xác định** |
| Upload `outputs/demo_images/*.png` | top-3 + xác suất (nếu có thư mục này) |

Sidebar: accuracy **98,56 %**, macro-F1 **0,9799**, τ = 0,9462.
Biển Việt Nam → “Không xác định” hoặc sai: model chỉ biết 43 lớp biển ĐỨC.

## 5. Nếu muốn dùng checkpoint khác

```powershell
# cách 1: thay file
Copy-Item "D:\checkpoints\best.pt" checkpoints\final_s42\best.pt

# cách 2: trỏ tới thư mục có best.pt
$env:GTSRB_RUN_DIR = "D:\checkpoints\final_s42"
streamlit run app.py
```

`threshold.json`, `test_report.json`, `class_names.json` đã có trong repo.

## 6. Lỗi nhanh

| Lỗi | Cách sửa |
|---|---|
| `streamlit` không nhận lệnh | `python -m streamlit run app.py` |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass` |
| `Không tìm thấy checkpoint` | kiểm tra `checkpoints\final_s42\best.pt` còn đó, hoặc set `GTSRB_RUN_DIR` |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| Thiếu `torch`/`streamlit` | activate `.venv` rồi `pip install -r requirements-demo.txt` |

Muốn train/đánh giá lại: `docs/HUONG_DAN_CHAY_MAY_MOI.md`.
