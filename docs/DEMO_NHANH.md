# DEMO NHANH (5 phút)

Bản ngắn: **giải nén → cài thư viện → chạy Streamlit → demo**. Không cần train, không cần data, không cần GPU.

## 1. Cần mang theo

Một file: **`GTSRB_demo_bundle.zip`** (~15 MB) — xin từ máy có model, hoặc tạo bằng
`python tests/package_demo.py`. Trong đó đã có model, báo cáo và ảnh demo.

Máy mới chỉ cần **Python 3.12 (64-bit)**. Kiểm tra:

```powershell
py -3.12 --version
```

## 2. Chạy (Windows)

```powershell
# 1) giải nén zip, mở PowerShell tại thư mục vừa giải nén
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
streamlit run app.py
```

- Nếu hỏi **Email**: nhấn **Enter** (bỏ trống).
- Mở: **http://localhost:8501**
- Dừng: `Ctrl+C`

Linux/macOS: thay 2 lệnh đầu bằng `python3.12 -m venv .venv` và `source .venv/bin/activate`.

## 3. Demo trên giao diện

| Thao tác | Kết quả mong đợi |
|---|---|
| Tab **Dự đoán** → chọn “Ảnh test GTSRB — biển 16” | Cấm xe trên 3,5 tấn (conf ~1,00) |
| Chọn “biển 50 km/h” / “STOP” | Giới hạn tốc độ 50 km/h / Dừng lại |
| Chọn “Biển Hàn Quốc 30 km/h” | Giới hạn tốc độ 30 km/h (conf 0,95) |
| Chọn “Cảnh đường không có biển” | **Không xác định** |
| Upload ảnh trong `outputs/demo_images/*.png` | Kết quả + top-3 + xác suất |
| Upload biển Việt Nam | **Không xác định** hoặc sai — model chỉ biết 43 lớp biển ĐỨC |

Sidebar hiển thị: accuracy **98,56 %**, macro-F1 **0,9799**, top-3 **99,64 %**, τ = 0,9462.
Báo cáo đầy đủ: mở `docs/bao_cao/BAO_CAO.docx`.

## 4. Lỗi nhanh

| Lỗi | Cách sửa |
|---|---|
| `streamlit` không nhận lệnh | dùng `python -m streamlit run app.py` |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass` rồi activate lại |
| `Không tìm thấy checkpoint` | kiểm tra có `outputs/runs/final_s42/best.pt` (giải nén đủ file) |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| Thiếu `torch`/`streamlit` | đang sai venv → activate `.venv` rồi `pip install -r requirements-demo.txt` |

**Không cần** cài CUDA, không cần tải GTSRB. Nếu muốn train/đánh giá lại: xem
`docs/HUONG_DAN_CHAY_MAY_MOI.md`.
