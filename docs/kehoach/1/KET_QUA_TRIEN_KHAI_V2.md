# KẾT QUẢ TRIỂN KHAI V2 — NHẬT KÝ CÓ BẰNG CHỨNG

> Mỗi mục chỉ ghi số **sinh ra từ artifact thật**. Đường dẫn file là bằng chứng.
> Số liệu tổng hợp cuối nằm ở `outputs/results/v2/summary.csv` và
> `outputs/results/v2/lockbox_result.json`.

## Phase 0 — Giao thức, split, audit, test

### 0.1. Pre-registration

- Ngày chốt: 02/11/2026.
- Commit: `18a785d` (`docs: chốt pre-registration thực nghiệm v2`).
- File: `docs/kehoach/1/PRE_REGISTRATION_V2.md` (8 so sánh chính, luật quyết định,
  luật dừng, quy tắc freeze/lockbox).

### 0.2. Split v2

Nguồn: `outputs/splits/v2/manifest.json` (SHA-256 file: xem `manifest.sha256`).

| Tập | Ảnh | Track | Sign-group | Seed |
|---|---:|---:|---:|---:|
| Pool | 39.201 | 1.307 | 1.183 | – |
| Lockbox | 8.302 | 277 | 238 | 2024 |
| Dev | 30.899 | 1.030 | 945 | – |
| Screening train | 24.719 | 824 | – | 42 |
| Screening val | 6.180 | 206 | – | 42 |
| Fold 0 train/val | 19.949 / 10.950 | – | – | 42 |
| Fold 1 train/val | 20.729 / 10.170 | – | – | 42 |
| Fold 2 train/val | 21.120 / 9.779 | – | – | 42 |

- Đơn vị nhóm: **sign-group** = track cùng một biển vật lý, gộp khi có cặp ảnh
  cùng nhãn khác track với MAD < 5 trên ảnh 64×64. Đã gộp **39.603 cặp**.
- Vì vậy số ảnh lockbox (8.302) và dev (30.899) khác bảng dự kiến trong kế hoạch
  (7.830 / 31.371); **manifest là nguồn chính thức**, không chép số dự kiến.
- mean/std tính riêng trên train của từng split, lưu trong `manifest.mean_std`
  (ví dụ screening: mean = 0,369832/0,339117/0,346847; std = 0,276523/0,267793/0,275659).

### 0.3. Audit C1–C12

Nguồn: `outputs/results/v2/audit/audit_report.json` + `checks.csv`.

| Mã | Kết quả | Đạt |
|---|---|---:|
| C1 track overlap lockbox/dev | 0 | ✓ |
| C2 sign-group overlap | 0 | ✓ |
| C3 lớp ở lockbox | 43 | ✓ |
| C4 lớp ở dev | 43 | ✓ |
| C5 lớp ở screening val | 43 | ✓ |
| C6 lớp ở 3 fold | 43/43/43 | ✓ |
| C7 overlap screening | 0 | ✓ |
| C8 overlap track 3 fold | 0/0/0 | ✓ |
| C9 test trong train/lockbox | 0 | ✓ |
| C10 lockbox gần trùng train (MAD < 5) | 0 | ✓ |
| C11 hash trùng trong pool | 0 | ✓ |
| C12 track/lớp nhỏ nhất ở dev | 6 | ✓ |
| HASH file split khớp manifest | khớp 6/6 | ✓ |
| MEANSTD khớp lại từ cache | 0,0e+00 | ✓ |
| COUNT manifest khớp file | khớp | ✓ |

Tổng: **15/15 PASS**. Ghi chú: audit dùng SHA-256 nội dung file thay MD5
(MD5 trùng ⇒ SHA-256 trùng), và chỉ gộp nhóm khi MAD < 5 nên C10 = 0 theo cấu trúc.

### 0.4. Test

- `pytest -q tests/test_split_v2.py` → **10 passed in 11,80 s**.
- Log: `outputs/results/v2/log_pytest_04.txt`.
- Bao gồm: forward A/B/C `(2,3,64,64)→(2,43)`, macro-F1/43 khớp sklearn
  (sai khác < 1e-9), đọc manifest/mean-std, `num_workers=2` trên Windows,
  3 fold phủ đúng toàn bộ dev một lần.

## Phase 1 — Baseline + factorial 2⁴

### 1.1. Baseline B1 (HOG + LinearSVC)

Nguồn: `outputs/results/v2/baseline.json`, log `outputs/results/v2/log_b1.txt`.

| Mục | Giá trị |
|---|---:|
| Train / val | 24.719 / 6.180 (screening) |
| HOG feature dim | 1.764 |
| C grid | 0,1; 1; 10 |
| **C chọn (theo val macro-F1)** | **0,1** |
| Val accuracy | 0,8727 |
| **Val macro-F1/43** | **0,8600** |
| Params (coef + intercept) | 75.895 |
| Thời gian trích HOG | 40,72 s |
| Thời gian fit (C=0,1) | 452,74 s |
| Latency predict-only | 0,0074 ms/ảnh |

- B0 majority (mốc dưới, cùng val): accuracy 0,0291 · macro-F1/43 0,0013.
- Ghi chú: liblinear báo `ConvergenceWarning` ở cả 3 mức C; đã ghi nhận, **không**
  tăng `max_iter` sau khi xem kết quả (tránh tune ngoài đăng ký).
- Không refit trên dev, không đánh giá trên lockbox.

