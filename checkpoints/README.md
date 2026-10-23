# Checkpoint cho demo (commit theo yêu cầu để `git clone` chạy được ngay)

| File | Kích thước | SHA-256 |
|---|---:|---|
| `final_s42/best.pt` | 8,4 MB | `9c4dd0d677d655e25f36ab2b29f6d8c73e7754056b2ba93c398030b85cbcbe33` |
| `GTSRB_demo_bundle.zip` | 15,3 MB | `4b13404d158904ab621ebcb9147d5453509b6af04a1dfc18d3775f87058d0d26` |

## Cách dùng

App tự tìm theo thứ tự:
1. `outputs/runs/final_s42/best.pt` (nếu đã train)
2. `checkpoints/final_s42/best.pt` (có sẵn sau khi clone)
3. biến môi trường `GTSRB_RUN_DIR` (nếu đặt nơi khác)

```powershell
pip install -r requirements-demo.txt
streamlit run app.py
```

Chi tiết: `docs/CHAY_NHANH_SAU_CLONE.md`.

## Vì sao commit file nhị phân?

Repo mặc định không commit `*.pt` (AGENTS §11). Ngoại lệ này do yêu cầu
"clone về chạy demo ngay": chỉ 2 file, tổng ~24 MB, không chứa dữ liệu cá nhân.
Không thêm checkpoint/zip nào khác vào thư mục này.
