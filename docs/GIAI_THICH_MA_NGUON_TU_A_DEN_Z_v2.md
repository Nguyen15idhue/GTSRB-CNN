# GIẢI THÍCH MÃ NGUỒN GTSRB-CNN — PHIÊN BẢN V2 (A → Z)

> Tài liệu này giải thích **đúng hiện trạng mã nguồn v2** trong repo. Các thành
> phần cũ (split random, E1–E7, demo) chỉ được nhắc ngắn để phân biệt, không
> dùng để giải thích luồng chính. Mọi số liệu lấy từ artifact v2.

| Thông tin | Giá trị |
|---|---|
| Phiên bản | 2.0 — 03/11/2026 |
| Trạng thái | Khớp commit `6a0f63e` (sau nghiệm thu) |
| Kết quả chính | OOF macro-F1 0,970400; lockbox 0,975981; official clean 0,982499 |
| Báo cáo | `docs/bao_cao/BAO_CAO_v2.md` |
| Nhật ký 18 bước | `docs/kehoach/1/KET_QUA_TRIEN_KHAI_V2.md` |

---

## Mục lục

1. [Cách đọc tài liệu](#1-cách-đọc-tài-liệu)
2. [Sản phẩm và kết quả hiện tại](#2-sản-phẩm-và-kết-quả-hiện-tại)
3. [Bức tranh lớn: pipeline v2](#3-bức-tranh-lớn-pipeline-v2)
4. [Kiến thức tối thiểu cần có](#4-kiến-thức-tối-thiểu-cần-có)
5. [Môi trường và cài đặt](#5-môi-trường-và-cài-đặt)
6. [Cấu trúc repo hiện tại](#6-cấu-trúc-repo-hiện-tại)
7. [Tầng dữ liệu](#7-tầng-dữ-liệu)
8. [Vì sao dùng sign-group thay vì random](#8-vì-sao-dùng-sign-group-thay-vì-random)
9. [Tầng mô hình](#9-tầng-mô-hình)
10. [Huấn luyện](#10-huấn-luyện)
11. [Đánh giá: screening, OOF, lockbox, official test](#11-đánh-giá)
12. [Thống kê và phân tích](#12-thống-kê-và-phân-tích)
13. [Sinh bảng, hình, quyết định](#13-sinh-bảng-hình-quyết-định)
14. [Script điều phối](#14-script-điều-phối)
15. [Hợp đồng dữ liệu giữa các tầng](#15-hợp-đồng-dữ-liệu-giữa-các-tầng)
16. [Cấu hình YAML](#16-cấu-hình-yaml)
17. [Test và kiểm chứng](#17-test-và-kiểm-chứng)
18. [Các bẫy đã gặp và cách sửa](#18-các-bẫy-đã-gặp-và-cách-sửa)
19. [Chạy lại từ đầu](#19-chạy-lại-từ-đầu)
20. [Phần cũ còn trong repo](#20-phần-cũ-còn-trong-repo)
21. [Thuật ngữ](#21-thuật-ngữ)

---

## 1. Cách đọc tài liệu

- Nếu chỉ muốn chạy lại: đọc mục 5, 16, 19.
- Nếu muốn hiểu code: đọc 6 → 7 → 9 → 10 → 11 → 12.
- Nếu bảo vệ: đọc 2, 8, 12.2, 13.3 và Chương 5 của báo cáo.
- Quy ước: `src/...` là thư viện, `scripts/...` là điều phối, `configs/v2/...`
  là cấu hình, `outputs/...` là artifact. Số trong tài liệu là số thật từ
  `outputs/results/v2/`; không có số minh hoạ.

---

## 2. Sản phẩm và kết quả hiện tại

### 2.1. Sản phẩm

- Pipeline chia split, train, đánh giá, thống kê, sinh báo cáo v2.
- 111 run CNN + 1 baseline B1.
- Báo cáo Markdown và DOCX: `docs/bao_cao/BAO_CAO_v2.md`, `BAO_CAO_v2.docx`.
- Artifact máy đọc: `outputs/results/v2/`.
- Test: 22 test (split, stats, config) + kịch bản tái lập.

### 2.2. Kết quả cốt lõi

| Hạng mục | Giá trị |
|---|---|
| Split | lockbox 8.302; dev 30.899; screening 24.719/6.180; 3 fold |
| Winner | D2 — 4 khối 32/64/128/256, flatten, BN, DO 0,3, WD 1e-4, aug all |
| Params | 1.448.779 |
| OOF 31k | macro-F1 0,970400 ± 0,001869 |
| Lockbox | acc 0,988758; macro-F1 0,975981; top-3 0,998474; ECE 0,005328 |
| Official raw | acc 0,986989; macro-F1 0,980398 |
| Official clean 11.879 | acc 0,988692; macro-F1 0,982499 |

---

## 3. Bức tranh lớn: pipeline v2

```text
data/gtsrb (PPM gốc)
   │  src/data.py: build_cache / compute_stats / preprocess
   ▼
data/cache (X_train.npy 39.201 × 64×64×3 uint8, X_test.npy 12.630)
   │  src/split_v2.py: sign-group + lockbox/dev/screening/fold + audit
   ▼
outputs/splits/v2/*.json + manifest.json (hash, mean/std)
   │  src/data_v2.py: đọc manifest, loader, augmentation
   ▼
src/train_v2.py ──► outputs/runs/v2_<config>_s<seed>[_f<fold>]/
   │                 config.yaml, env.json, metrics.csv, best.pt,
   │                 curves.png, meta.json, predictions_val.csv
   ▼
src/summarize_v2.py ──► outputs/results/v2/summary.csv (111 run)
   │
   ├─ src/analysis_v2.py  (factorial, OOF, corruption)
   ├─ src/stats_v2.py     (bootstrap, Holm, variance)
   ├─ src/decision_cards_v2.py
   ├─ src/explain_v2.py   (hình cơ chế)
   └─ src/final_eval_v2.py (lockbox one-shot, official test)
   ▼
outputs/results/v2/{tables,figures,ACCEPTANCE.json}
   ▼
docs/bao_cao/BAO_CAO_v2.md → .docx
```

Năm phase tương ứng 18 bước:

| Phase | Việc | Run |
|---|---|---|
| 0 | Pre-registration, split, audit, test | 0 |
| 1 | Baseline + factorial 2⁴ | 1 + 48 |
| 2 | Kiến trúc, hyperparam, augmentation | 15 + 9 + 9 |
| 3 | OOF, freeze, final, lockbox, official test | 27 + 3 |
| 4 | Thống kê, hình, decision card, báo cáo | 0 |
| 5 | Nghiệm thu 12 điều kiện | 0 |

---

## 4. Kiến thức tối thiểu cần có

- **Tensor:** mảng nhiều chiều; ảnh batch là `(B, 3, H, W)`.
- **CNN:** conv 3×3 học đặc trưng cục bộ; BN chuẩn hóa activation; maxpool giảm
  không gian; flatten/GAP chuyển sang vector; FC cho logits.
- **Logits vs softmax:** model trả logits; `CrossEntropyLoss` tự làm softmax.
- **Epoch/batch:** một epoch duyệt hết train một lần; batch 128.
- **Overfit:** train tốt, val kém; đo bằng gap train−val.
- **Macro-F1:** trung bình F1 43 lớp, lớp hiếm quan trọng ngang lớp lớn.
- ** Leak:** val chứa thông tin của train (ở đây là frame cùng track).
- **Bootstrap theo track:** resample cả track, không resample từng ảnh, vì frame
  trong track tương quan mạnh.
- **Seed:** cố định ngẫu nhiên để tái lập.

---

## 5. Môi trường và cài đặt

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Kỳ vọng: `2.11.0+cu128 True (12, 0)` trên RTX 5070. Không cài torch từ PyPI
mặc định vì dễ ra bản CPU. `requirements.txt` là nguồn dependency duy nhất.

---

## 6. Cấu trúc repo hiện tại

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   ├── default.yaml, e*.yaml, final.yaml    # LEGACY (split cũ)
│   ├── class_names.json                     # 43 lớp en/vi/shape/color
│   └── v2/
│       ├── f00..f15.yaml                    # factorial
│       ├── d1,d2,d3,w1,w2.yaml              # kiến trúc
│       ├── l1,cw1,i2.yaml                   # hyperparam
│       ├── g1,g2,g3.yaml                    # augmentation
│       ├── x1,x2,x3.yaml                    # OOF (sinh bởi select_candidates)
│       ├── final.yaml                       # đóng băng
│       └── ALIASES.md
├── src/
│   ├── config.py        # đường dẫn, hằng số, seed, env, git
│   ├── data.py          # cache, preprocess, stats, split legacy
│   ├── split_v2.py      # sign-group + split 3 tầng + audit data
│   ├── data_v2.py       # manifest, cache 48/96, loader v2, aug
│   ├── models.py        # SmallCNN A/B/C/D2
│   ├── train_v2.py      # train/val, checkpoint, predictions
│   ├── baseline_v2.py   # HOG + LinearSVC
│   ├── audit_v2.py      # C1–C12 độc lập
│   ├── summarize_v2.py  # summary.csv
│   ├── stats_v2.py      # bootstrap, Holm, variance
│   ├── analysis_v2.py   # factorial, OOF, corruption
│   ├── decision_cards_v2.py
│   ├── explain_v2.py    # hình cơ chế
│   ├── final_eval_v2.py # lockbox, official test, near-dup test
│   ├── evaluate.py, train.py, baseline.py, predict.py   # LEGACY
│   └── error_analysis.py, preprocess.py, summarize.py   # LEGACY
├── scripts/
│   ├── gen_configs_v2.py, run_matrix.py
│   ├── select_candidates.py, run_oof_analysis.py
│   ├── freeze_v2.py, check_final_ckpt.py
│   ├── make_tables_v2.py, run_phase4.py, lockbox_figures.py
│   ├── verify_12.py, verify_41.py
│   ├── repro_check_v2.py, acceptance_v2.py
├── tests/
│   ├── test_split_v2.py, test_stats_v2.py, test_config_v2.py
│   └── test_pipeline.py, golden_test.py, ...            # LEGACY
├── outputs/
│   ├── splits/v2/          # COMMIT (nhỏ, có hash)
│   ├── results/v2/         # COMMIT (trừ log lớn)
│   ├── figures/v2/         # COMMIT
│   └── runs/               # KHÔNG commit
└── docs/
    ├── bao_cao/BAO_CAO_v2.md/.docx
    └── kehoach/1/KET_QUA_TRIEN_KHAI_V2.md
```

Điểm quan trọng: **v2 không sửa logic cũ**; nó thêm module `*_v2` và dùng cache
sẵn có. `src/data.py` vẫn là nơi định nghĩa cache, `preprocess`, `compute_stats`,
`_track_key` để v2 tái sử dụng — tránh hai bản tiền xử lý lệch nhau.

---

## 7. Tầng dữ liệu

### 7.1. `src/config.py` — hằng số và tiện ích

- `PROJECT_ROOT`, `DATA_ROOT`, `CACHE_DIR`, `RUNS_DIR`, `RESULTS_DIR`.
- `NUM_CLASSES = 43`, `IMG_SIZE = 64`, `DEFAULT_SEED = 42`.
- `set_seed(seed)`: seed Python/NumPy/torch/CUDA, bật deterministic.
- `seed_worker`: seed worker DataLoader.
- `get_device(prefer)`: auto/cuda/cpu.
- `git_commit`, `collect_env`: ghi hash và môi trường vào run.
- `load_config` + `validate_config`: dùng cho config legacy (merge `default.yaml`).
  **Config v2 là file đầy đủ**, nạp bằng `yaml.safe_load` trong `train_v2`, nên
  không đi qua `VALID_SPLITS` của legacy. Đây là lý do `configs/v2/final.yaml`
  có `data.split: dev_all` dù `VALID_SPLITS` cũ không có.

### 7.2. `src/data.py` — cache, thống kê, tiền xử lý

Các hàm v2 dùng lại:

| Hàm | Vai trò |
|---|---|
| `load_train_pool()` | Đọc `train_meta.csv`, chỉ giữ `usable_for_train=1` → 39.201 ảnh |
| `load_cache(64)` | Đọc `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, paths |
| `compute_stats(x, indices)` | mean/std 3 kênh trên đúng tập train |
| `_array_to_tensor(array, mean, std)` | uint8 HWC → float tensor (3,H,W) chuẩn hóa |
| `preprocess(image, size, mean, std)` | Hàm tiền xử lý **duy nhất** cho train/eval/demo |
| `_track_key(path, label)` | Khóa track `<label>_<track>` |
| `_read_train_meta()` | Đọc metadata 39.209 dòng |

`build_cache` gốc ghi vào `data/cache`; v2 **không gọi lại** cho 64×64 để tránh
ghi đè. Với 96×96, `data_v2.build_cache_size(96)` ghi vào `data/cache_96`.

### 7.3. `src/split_v2.py` — trái tim của split v2

Chạy: `python -m src.split_v2 --build --force`.

Các bước:

1. **Đọc pool** 39.201 ảnh + nhãn + track từ `train_meta.csv` và cache 64.
2. **Dựng sign-group:**
   - `_thumb(x)`: ảnh xám 16×16.
   - `_candidate_pairs`: trong mỗi lớp, so thumbnail; giữ cặp khác track có
     MAD thumbnail < 8. Thêm LSH aHash/dHash toàn cục để bắt cặp khác track.
   - `build_sign_groups`: kiểm chứng full 64×64, nếu MAD < 5 thì `union`.
   - Gộp mọi frame cùng track (track là một biển).
   - `name_groups`: đặt tên `<label>_g<k>` theo track nhỏ nhất.
3. **Chia lockbox:** mỗi lớp chọn `round(0.2 * n_group)` nhóm, seed 2024.
4. **Chia screening:** trong dev, mỗi lớp chọn 20% nhóm, seed 42.
5. **Chia 3 fold:** mỗi lớp, sắp nhóm theo số frame giảm dần, gán vào fold ít
   frame nhất (cân bằng).
6. **Audit nội bộ:** C1–C12, hash, mean/std; nếu fail → raise, không ghi manifest.
7. **Ghi artifact:** `lockbox.json`, `dev.json`, `screening.json`, `fold_0..2.json`,
   `manifest.json`, `manifest.sha256`, `audit/sign_groups.csv`,
   `audit/near_duplicate_lockbox.csv`, `audit/md5_duplicates_train.csv`.

Vì sao dùng **sign-group** chứ không phải track? Vì hai track khác nhau có thể
quay cùng một biển vật lý. Nếu chỉ tách track, ảnh của cùng biển vẫn có thể nằm
hai bên train/test → test dễ hơn thực tế. Sign-group gộp cả hai track đó.

Vì sao **không xóa** ảnh gần trùng? Vì xóa sẽ làm lệch phân bố và mất dữ liệu;
chỉ cần không cho chúng khác tập. Audit chỉ để kiểm chứng.

### 7.4. `src/data_v2.py` — đọc manifest và loader

| Hàm | Vai trò |
|---|---|
| `load_manifest()` | Đọc `outputs/splits/v2/manifest.json` |
| `get_indices(name, fold)` | Trả `{train, val}` cho screening/dev_all/lockbox/fold_k |
| `get_mean_std(name)` | Lấy mean/std đúng train của split |
| `load_images(size)` | Cache 64 hoặc 48/96 (`data/cache_48`, `cache_96`) |
| `build_cache_size(size)` | Dựng cache 48/96 từ ảnh gốc, không đụng cache 64 |
| `track_map()` | Danh sách index → path/label/track_key |
| `build_augmentation(profile, size, mean, std)` | Trả `Compose([...aug, ToTensor, Normalize])` |
| `V2Dataset` | `__getitem__` trả tensor + nhãn; train có aug, val không |
| `make_loaders_v2` | Tạo train/val loader, seed generator, `num_workers` |

Augmentation profile:

| Profile | Thành phần |
|---|---|
| `none` | không aug |
| `affine` | RandomAffine ±10°, translate 0,1, scale 0,9–1,1, shear 5 |
| `color` | ColorJitter 0,2/0,2/0,2/hue 0,02 |
| `color_strong` | ColorJitter 0,4/0,4/0,3/hue 0,03 (exploratory) |
| `perspective` | RandomPerspective 0,1, p=0,3 |
| `all` | affine + color + perspective |
| `affine_color`, `affine_perspective`, `color_perspective` | tổ hợp exploratory |

`build_augmentation` **luôn** kết thúc bằng `ToTensor` + `Normalize`; nếu thiếu,
DataLoader sẽ trả PIL và `default_collate` báo lỗi (đã từng gặp).

Biến môi trường `V2_NUM_WORKERS` cho phép tăng worker mà **không đổi
config_sha256**; số worker thực tế ghi vào `meta.json`.

---

## 8. Vì sao dùng sign-group thay vì random

Đây là lý do sống còn của v2. Số từ `outputs/results/leak_delta.json` (chẩn đoán
trên split cũ, cùng dữ liệu):

| Chỉ số | Random | Group/sign-group |
|---|---:|---:|
| % frame val có frame train cách ±1 | **95,22%** | 0% |
| Track train/val trùng | **1.306 / 1.307** | **0** |
| 1-NN 16×16, không học | **90,5%** | 31,0% |
| Val accuracy cùng model | 99,936% | 98,473% |
| Macro-F1/43 cùng model | 99,947% | 88,351%* |

\* Group val cũ thiếu 4 lớp nên bị phạt cấu trúc 9,06 điểm; phần do leak thật
khoảng 2,53 điểm. Đây cũng là lý do v2 kiểm tra đủ 43 lớp ở mọi tập.

Kết luận: random split khiến val **gần như là bản sao** của train ở mức frame.
Mọi so sánh trên val random bị thổi phồng và không dùng được. V2 dùng
sign-group, lockbox riêng, mean/std theo split, và báo official test chỉ như
benchmark phụ.

---

## 9. Tầng mô hình

`src/models.py` định nghĩa `SmallCNN`:

- `features`: danh sách block `Conv2d(3×3, padding=1) → [BatchNorm] → ReLU → MaxPool(2)`.
- `classifier`:
  - Nếu `gap=True`: `Dropout → Linear(width_cuối, 43)`.
  - Nếu `gap=False`: `Dropout → Linear(w*h*w, fc_hidden) → ReLU → Dropout → Linear(fc_hidden, 43)`.
- `forward` trả **logits**, không softmax.
- `count_parameters` chỉ tính tham số `requires_grad` (không tính buffer BN).
- `build_model(cfg)` đọc `cfg["model"]` và `cfg["data"]`.

Các cấu hình:

| Tên | Widths | BN | Dropout | GAP | FC | Params |
|---|---|---:|---:|---|---:|---:|
| CNN-A / F00 | 32/64/128 | Không | 0 | Không | 256 | 2.201.707 |
| CNN-B / F07,F15 | 32/64/128 | Có | 0,3 | Không | 256 | 2.201.931 |
| CNN-C / D3 | 32/64/128/256 | Có | 0,4 | Có | – | 399.947 |
| D2 (winner) | 32/64/128/256 | Có | 0,3 | Không | 256 | 1.448.779 |
| D1 | 32/64/128 | Có | 0,3 | Có | – | 99.019 |
| W1 | 16/32/64 | Có | 0,3 | Không | 256 | 1.083.579 |
| W2 | 48/96/192 | Có | 0,3 | Không | 256 | 3.366.363 |

Điểm dễ nhầm: D3 **không** dùng để kết luận cơ chế vì đổi cả depth, GAP và
dropout 0,4. D1/D2 mới tách từng yếu tố.

---

## 10. Huấn luyện

### 10.1. Luồng `src/train_v2.py`

```text
run_training_v2(config_path, seed, fold)
 ├─ load_yaml(config_path)
 ├─ resolve_split(cfg, fold) → split_name, used_fold
 ├─ cfg["data"]["fold"] = used_fold        # bug cũ: thiếu dòng này
 ├─ make_loaders_v2(...)                   # mean/std từ manifest
 ├─ build_model(cfg)                       # params
 ├─ AdamW + CosineAnnealingLR + GradScaler(AMP)
 ├─ for epoch 1..max_epochs:
 │    train_epoch → evaluate_epoch (nếu có val) → ghi metrics.csv (flush)
 │    nếu val F1 tốt hơn → lưu best.pt
 │    nếu patience → dừng
 ├─ load best.pt → predict_val → predictions_val.csv
 └─ meta.json + curves.png
```

### 10.2. Chế độ đặc biệt: final trên `dev_all`

`dev_all` không có val. `train_v2` phát hiện `len(val)==0`:

- không gọi `evaluate_epoch`, ghi `val_*` rỗng;
- không early stop;
- lưu `best.pt` mỗi epoch, nên checkpoint cuối = epoch cuối (26);
- `predictions_val.csv` chỉ có header;
- `meta.val_macro_f1 = None`.

Đây là lý do `summary.csv` của 3 run final để trống cột val; kết quả của chúng
nằm ở `lockbox_result.json`.

### 10.3. Artifact một run

```text
outputs/runs/v2_<config>_s<seed>[_f<fold>]/
├── config.yaml          # JSON dump của config đã resolve + _resolved
├── env.json             # Python/torch/CUDA/GPU/OS
├── metrics.csv          # schema 9 cột
├── best.pt              # dict, không pickle model
├── curves.png
├── meta.json            # metric, params, hash, latency, notes
└── predictions_val.csv  # image_index, track_key, true/pred, prob, fold
```

`metrics.csv`:

```text
epoch,lr,train_loss,train_acc,val_loss,val_acc,val_macro_f1,epoch_time_s,gpu_mem_mb
```

`best.pt`:

```python
{
  "state_dict": ..., "config": ..., "class_names": ...,
  "epoch": int, "val_macro_f1": float,
  "git_commit": str, "torch_version": str,
  "config_sha256": str, "split_sha256": str,
}
```

`config_sha256` = SHA-256 file YAML (không phải config đã resolve).
`split_sha256` = hash file split trong manifest. Vì vậy có thể kiểm tra một run
có đúng config/split đã đăng ký hay không.

### 10.4. Class weight và label smoothing

- `class_weight=none`: CrossEntropy thường.
- `inv_sqrt` / `inv_sqrt_freq`: `1/sqrt(freq)` chuẩn hóa trung bình 1.
- `inv_freq`: `1/freq` (exploratory).
- `label_smoothing` đọc từ config (mặc định 0).

### 10.5. Latency

Sau train, `_latency_ms` đo model-only batch 1, 100 lần, lấy median. Đây là
latency **model-only**, không phải thời gian UI/end-to-end. Báo cáo ghi rõ.

### 10.6. Chi phí huấn luyện được ghi ở đâu

- `metrics.csv`: mỗi epoch có `epoch_time_s` và `gpu_mem_mb`.
- `meta.json`: `train_time_s` = tổng vòng epoch; `infer_ms_per_img`.
- `scripts/training_cost_v2.py` đọc 111 run + B1, sinh
  `training_cost.csv`, `training_cost_summary.json`, `tables/training_cost.md`.
- Số thực tế: 111 run CNN = 3.411 epoch = **3,99 giờ** epoch loop; B1 = 8,2 phút;
  tổng ≈ **4,13 giờ** (chưa gồm phân tích). D2 screening 132,8 s/run; final
  26 epoch = 157,83–158,70 s/seed.
- Phạm vi: không gồm nạp cache/khởi động, dự đoán cuối, vẽ curve, đo latency.

---

## 11. Đánh giá

### 11.1. Screening val

Mỗi run screening có `predictions_val.csv` trên 6.180 ảnh. `summarize_v2` tính
lại macro-F1 và CI từ file này, không tin meta.

### 11.2. OOF

`src/analysis_v2.py:oof_predictions`:

- đọc 3 fold của một seed;
- kiểm tra không trùng ảnh và phủ đúng `dev_all`;
- trả `{seed: {image_index: {true, pred, fold, prob_true, prob_pred}}}`.

`oof_table`:

- ghép 3 fold → 30.899 ảnh/seed;
- macro-F1/43, CI cluster bootstrap B=2000;
- so sánh cặp bằng paired bootstrap;
- variance decomposition seed × fold.

`scripts/run_oof_analysis.py` xuất `oof_predictions.csv` (278.091 dòng),
`oof_summary.json`, `oof_comparisons.csv`, `variance.json`.

### 11.3. Lockbox — một lần

`src/final_eval_v2.py`:

- `_guard_freeze`: bắt buộc có `FREEZE.json`; `meta.config_sha256` của từng run
  final phải khớp `final_config_sha256`; `best.pt` phải tồn tại.
- `run_lockbox`: nếu `lockbox_result.json` đã có → từ chối, trừ `--allow-rerun`
  + lý do (sinh `_v2`).
- Ghi `lockbox_access.log`: thời gian, user, host, runs, n, acc, F1, rerun.
- `evaluate_checkpoint`: forward theo batch 256, softmax, metric đầy đủ
  (per-class, confusion, top-3, ECE, Brier).

### 11.4. Official test raw/clean

- `compute_test_near_duplicates`: so test với train theo lớp, thumbnail 16×16,
  kiểm chứng MAD < 5; ghi `audit/near_duplicate_test.csv` và `clean_test.json`.
- `run_official_test`: đánh giá 3 seed trên 12.630 raw và 11.879 clean; ghi
  `official_test_v2.json`.

### 11.5. Baseline

`src/baseline_v2.py`: HOG 64×64 (1764 chiều) + StandardScaler + LinearSVC,
C ∈ {0,1; 1; 10}, chọn trên screening val. Không refit dev, không chạm lockbox.

---

## 12. Thống kê và phân tích

### 12.1. `src/stats_v2.py`

| Hàm | Công dụng |
|---|---|
| `macro_f1_43` | F1 43 lớp, lớp vắng = 0 |
| `cluster_bootstrap_f1` | CI 95% theo track, B mặc định 2000 |
| `paired_bootstrap_delta` | Δ + CI + p_boot cho hai model cùng mẫu track |
| `bootstrap_contrast` | CI cho tổ hợp tuyến tính F1 (main effect/interaction) |
| `holm` | Holm–Bonferroni |
| `decision` | Áp luật 4.5 |
| `variance_decomposition` | ANOVA 2 chiều seed × fold, ICC |
| `read_predictions` | Đọc predictions CSV |

Tại sao bootstrap theo track? Vì 30 frame trong một track gần giống nhau; nếu
resample từng ảnh, CI sẽ hẹp giả và kết luận quá tự tin. Kết quả thực tế:
phương sai fold chiếm 91,0%, seed 0,9% — nếu chỉ chạy 1 fold/3 seed sẽ sai.

### 12.2. `src/analysis_v2.py`

- `_aligned_predictions`: căn nhiều config theo `image_index` (cùng val set).
- `factorial_effects`: main effect + interaction, bootstrap theo seed.
- `factorial_table`: 16 dòng mean ± sd, Δ vs F00, gap, params, time.
- `aggregate`: bảng mean ± sd + Δ vs reference cho một nhóm config.
- `oof_predictions` / `oof_table`: OOF, CI, so sánh, variance.
- `corrupt_images`: 6 loại × 3 mức (blur, noise, brightness, contrast, JPEG, occlusion).
- `corruption_table`: sinh ảnh một lần, đánh giá nhiều checkpoint.

### 12.3. `src/summarize_v2.py`

Quét `outputs/runs/v2_*` có `meta.json`, bỏ qua `v2_repro*`, ghi `summary.csv`
đúng schema 11.3 của kế hoạch. Cột `valid_for_conclusion` = 1 nếu chạy xong.
CI được tính từ `predictions_val.csv` nếu có; run final không có val thì để trống.

---

## 13. Sinh bảng, hình, quyết định

### 13.1. `src/decision_cards_v2.py`

- 4 card factorial (BN, dropout, WD, aug) lấy từ `factorial_effects`.
- 3 card kiến trúc (GAP=D1 vs F15, Depth=D2 vs F15, Width=W2 vs F15).
- 1 card LR (L1 vs F15).
- 2 card exploratory (CW1, I2).
- Holm cho đúng 8 so sánh chính; ghi `decision_cards.csv`,
  `main_comparisons.csv`.

### 13.2. `src/explain_v2.py`

| Hàm | Hình |
|---|---|
| `learning_curves` | 01 |
| `interaction_plot` | 02 |
| `pareto_plot` | 03 |
| `per_class_heatmap` | 04 + `per_class_delta.csv` |
| `reliability_plot` | 05 + `calibration.csv` |
| `corruption_curves` | 06 |
| `confusion_from_predictions` | 07 |
| `gradcam_grid` | 08 |
| `error_taxonomy` | 09 + CSV/JSON |

Grad-CAM hook vào **block conv cuối** (`model.features[-1]`), không hook vào
conv trong block vì ReLU inplace làm autograd báo lỗi view+inplace.

### 13.3. Hình lockbox

`scripts/lockbox_figures.py`: confusion trung bình 3 seed (10), per-class F1
(11), `lockbox_per_class.csv`.

### 13.4. Bảng markdown

`scripts/make_tables_v2.py`: factorial, main effects, architecture, hyperparam,
augmentation, corruption → `outputs/results/v2/tables/*.md`.

---

## 14. Script điều phối

| Script | Việc | Resume |
|---|---|---|
| `gen_configs_v2.py` | Sinh `configs/v2/*.yaml` từ `MA_TRAN_RUN.csv`, bỏ alias | – |
| `run_matrix.py` | Chạy danh sách config × seed × fold | Bỏ qua run có `meta.json`; lỗi ghi log, tiếp tục |
| `select_candidates.py` | Xếp hạng screening, chọn top 2 + control, sinh X1–X3 | Ghi lý do trước OOF |
| `run_oof_analysis.py` | Xuất OOF + CI + variance | – |
| `freeze_v2.py` | Sinh `final.yaml` + `FREEZE.json` | epoch = median best epoch |
| `check_final_ckpt.py` | Load/forward/hash check trước lockbox | – |
| `verify_12.py` | Artifact + mapping E1/E2/E3 + cửa F15 | – |
| `verify_41.py` | So macro-F1 với sklearn + hash đủ | – |
| `repro_check_v2.py` | 2 lần 3 epoch cùng seed | Ghi `reproducibility.json` |
| `acceptance_v2.py` | 12 điều kiện | Ghi `ACCEPTANCE.json` |
| `run_phase4.py` | Chạy summary, effects, hình, card | Bắt lỗi từng phần |

`run_matrix.py` dùng `run_training_v2` trực tiếp trong một process để giữ cache
trong RAM; mỗi run vẫn tạo thư mục riêng.

---

## 15. Hợp đồng dữ liệu giữa các tầng

### 15.1. Split

`screening.json`: `{name, seed, train: [...], val: [...]}`.
`fold_k.json`: `{name, seed, fold, train, val}`.
`lockbox.json`: `{name, seed, indices}`.
`dev.json`: `{name, indices}`.
`manifest.json`: số ảnh/track/group, seed, sha256 từng file, checks C1–C12,
`mean_std` từng split.

Chỉ số trong JSON là **index trong pool 39.201**, khớp thứ tự `X_train.npy`.

### 15.2. Run

Xem 10.3. `predictions_val.csv`:

```text
image_index,track_key,true_label,pred_label,prob_true,prob_pred,fold
```

`image_index` là index pool, dùng để ghép OOF.

### 15.3. Summary

Schema 11.3:

```text
protocol,block,config_id,run_id,seed,fold,split,model,params,aug,aug_profile,
epochs_run,best_epoch,val_acc,val_macro_f1,val_macro_f1_ci_low,val_macro_f1_ci_high,
test_acc,test_macro_f1,train_time_s,infer_ms_per_img,git_commit,config_sha256,
split_sha256,valid_for_conclusion,notes
```

### 15.4. OOF

```text
config_id,seed,image_index,track_key,true_label,pred_label,prob_true,prob_pred,fold
```

Mỗi `(config_id, seed, image_index)` đúng một dòng.

### 15.5. Decision cards

```text
factor,levels,reference,metric,mean,ci_low,ci_high,delta,delta_ci_low,
delta_ci_high,p_holm,mechanism,evidence,decision,tradeoff
```

### 15.6. Acceptance

`ACCEPTANCE.json`: 12 check, mỗi check có `value` và `evidence`.

---

## 16. Cấu hình YAML

### 16.1. Ví dụ F00 (CNN-A, không regularization)

```yaml
experiment: {id: f00, protocol: v2_clean, block: F_factorial, seed: 42}
data: {split: screening, fold: null, image_size: 64, num_classes: 43, num_workers: 4}
model: {name: smallcnn_3blk, widths: [32, 64, 128], use_bn: false,
        dropout: 0.0, gap: false, fc_hidden: 256}
augmentation: {enabled: false, profile: none}
train: {optimizer: adamw, lr: 0.001, weight_decay: 0.0, batch_size: 128,
        max_epochs: 40, patience: 7, scheduler: cosine, class_weight: none,
        label_smoothing: 0.0, amp: true, monitor: val_macro_f1}
output: {dir: outputs/runs}
```

### 16.2. Khác biệt của final

```yaml
experiment: {id: final, ...}
data: {split: dev_all, fold: null}
train: {max_epochs: 26, patience: 27}   # epoch cố định, không early stop
```

`configs/v2/final.yaml` được sinh bởi `freeze_v2.py`, không sửa tay sau freeze.

### 16.3. Alias

D4 = F15, L3 = F15, G0 = F07, G5 = F15; R1–R4 = F00/F07/F15/D3.
Không có file YAML cho alias để tránh chạy trùng.

---

## 17. Test và kiểm chứng

```powershell
pytest -q tests/test_split_v2.py tests/test_stats_v2.py tests/test_config_v2.py
```

- `test_split_v2.py` (10 test): manifest, hash, disjoint, đủ lớp, mean/std,
  forward A/B/C, macro-F1 khớp sklearn, không flip, `num_workers=2`, 3 fold phủ dev.
- `test_stats_v2.py` (5 test): macro-F1/43, Holm, luật quyết định, bootstrap CI,
  variance decomposition.
- `test_config_v2.py` (7 test): F00/E1, F07/E2, F15/E3, D3 legacy, protocol,
  final khớp freeze, không có file alias.

Kiểm chứng số:

- `verify_41.py`: macro-F1 OOF x2/42 lệch 0,0; summary F15 lệch 3,1e−07.
- `repro_check_v2.py`: 2 lần 3 epoch cùng seed, 4 metric lệch 0,0.
- `acceptance_v2.py`: 12/12.

---

## 18. Các bẫy đã gặp và cách sửa

| Bẫy | Biểu hiện | Sửa |
|---|---|---|
| Thiếu `ToTensor` trong aug v2 | DataLoader báo "batch must contain tensors ... PIL.Image" | `build_augmentation` luôn append `ToTensor` + `Normalize` |
| Không ghi `fold` override vào cfg | Cả 3 fold dùng `fold_0`, OOF báo trùng ảnh | `cfg["data"]["fold"] = used_fold` trước `make_loaders_v2` |
| Ghi JSON bằng `write_text` trên Windows | Hash manifest lệch do LF→CRLF | `write_bytes` + `.gitattributes -text` cho file split |
| `track_map()` gọi trong comprehension | Lockbox treo 30 phút vì đọc CSV 8.302 lần | Gọi một lần ngoài vòng lặp |
| `dev_all` không có val | `f1_score` báo empty array | Chế độ final: bỏ eval/early stop, lưu checkpoint cuối |
| Gradient hook vào conv có ReLU inplace | "view ... modified inplace" | Hook vào cả block `features[-1]` |
| Test protocol áp cho `final.yaml` | `max_epochs=26` khác 40 | Bỏ final khỏi test 40 epoch; thêm test khớp freeze |
| 2 worker Windows | Treo nếu không có `__main__` guard | Entry point `python -m`, test riêng với dataset nhỏ |

---

## 19. Chạy lại từ đầu

```powershell
# 0. Chuẩn bị dữ liệu + cache 64 (nếu chưa có)
python -m src.data --download --report
python -m src.data --prepare

# 1. Split v2 + audit + test
python -m src.split_v2 --build --force
python -m src.audit_v2
pytest -q tests/test_split_v2.py

# 2. Baseline + factorial
python -m src.baseline_v2
python scripts/run_matrix.py --block F_factorial --seeds 42,123,2024
python -m src.summarize_v2

# 3. Kiến trúc + hyperparam + augmentation
python scripts/run_matrix.py --configs d1,d2,d3,w1,w2 --seeds 42,123,2024
python scripts/run_matrix.py --configs l1,cw1,i2 --seeds 42,123,2024
python scripts/run_matrix.py --configs g1,g2,g3 --seeds 42,123,2024
python -m src.summarize_v2

# 4. Chọn top 2 + control, OOF
python scripts/select_candidates.py
python scripts/run_matrix.py --configs x1,x2,x3 --seeds 42,123,2024 --folds 0,1,2
python scripts/run_oof_analysis.py --configs x1,x2,x3

# 5. Freeze + final + lockbox + official test
python scripts/freeze_v2.py --winner x2 --source d2 --reason "..."
python scripts/run_matrix.py --configs final --seeds 42,123,2024
python scripts/check_final_ckpt.py
python -m src.final_eval_v2 --mode lockbox --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024
python -m src.final_eval_v2 --mode test --runs outputs/runs/v2_final_s42,outputs/runs/v2_final_s123,outputs/runs/v2_final_s2024

# 6. Phân tích + nghiệm thu
python scripts/run_phase4.py
python scripts/make_tables_v2.py
python scripts/acceptance_v2.py
```

Thời gian tham khảo trên RTX 5070: ~3–4 giờ cho toàn bộ (tùy augmentation).

---

## 20. Phần cũ còn trong repo

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| `src/data.py` split random/group/stratified_group | Giữ | v2 chỉ dùng cache/preprocess/stats/track_key |
| `src/train.py`, `src/evaluate.py`, `src/baseline.py` | Legacy | Dùng split cũ, không dùng cho kết luận v2 |
| `configs/e*.yaml`, `configs/final.yaml` | Legacy | Final cũ E3; v2 dùng `configs/v2/final.yaml` |
| `app.py`, `src/predict.py` | Demo | Vẫn trỏ checkpoint cũ `final_s42`; chưa cập nhật v2 |
| `outputs/results/summary.csv` | Legacy | Có 9 dòng split random; đã tách sang `legacy/` |
| `tests/test_pipeline.py`, `golden_test.py` | Legacy | Không thuộc nghiệm thu v2 |

Chiến lược là **không sửa logic cũ** để tránh phá demo/báo cáo cũ; v2 thêm module
song song và tự ghi artifact riêng. Khi trích dẫn, luôn dùng `outputs/results/v2/`.

---

## 21. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Pool | 39.201 ảnh train sau khi loại 8 ảnh trùng MD5 với test |
| Track | Đoạn video ngắn của một biển; khóa `<label>_<track>` |
| Sign-group | Nhóm track cùng một biển vật lý (MAD < 5) |
| Lockbox | 8.302 ảnh test nội bộ, mở một lần sau freeze |
| Dev | 30.899 ảnh còn lại dùng train/val |
| Screening | Chia 80/20 trong dev để so sánh nhanh |
| OOF | Mỗi ảnh dev được dự đoán đúng một lần qua 3 fold |
| Macro-F1/43 | F1 trung bình 43 lớp, lớp vắng = 0 |
| Cluster bootstrap | Resample theo track, không theo ảnh |
| Holm | Hiệu chỉnh p-value cho 8 so sánh chính |
| ECE | Expected Calibration Error, 15 bin |
| Decision card | Thẻ kết quả + cơ chế + quyết định cho một yếu tố |
| Legacy leaky | Kết quả cũ dùng split random, không dùng kết luận |
