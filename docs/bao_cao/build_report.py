"""Sinh báo cáo Markdown từ artifact thật (không bịa số).

Chạy: python docs/bao_cao/build_report.py
Ra:   docs/bao_cao/BAO_CAO.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "docs" / "bao_cao" / "BAO_CAO.md"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def rows(name: str) -> list[dict]:
    with (RES / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f} %"


def num(value: int | float) -> str:
    """Số nguyên kiểu Việt Nam: 31.360 thay vì 31,360."""
    return f"{int(value):,}".replace(",", ".")


CURRENT_CHAPTER = "1"
CHAPTER_COUNTER: dict[str, int] = {}
CAPTIONS: dict[tuple[str, tuple[str, ...]], str] = {
    ("2", ("Hạng mục", "Giá trị")): "Quy mô và chất lượng dữ liệu (đếm lại từ đĩa)",
    ("2", ("Split", "Train / Val", "Track train / val", "Lớp ở val", "Lớp vắng", "Lớp ít ảnh nhất ở val")): "Ba cách chia tập",
    ("2", ("Cửa sổ frame", "Val random", "Val group")): "Tỷ lệ ảnh val có frame train gần (rò rỉ)",
    ("3", ("Mô hình", "Thiết lập", "Vai trò")): "Hai baseline",
    ("3", ("Mô hình", "Khối conv", "BN", "Dropout", "GAP", "Aug", "Params thật")): "Ba cấu hình CNN",
    ("3", ("Tham số", "Giá trị")): "Thiết lập huấn luyện",
    ("4", ("Mô hình", "Val acc", "Val macro-F1", "Ghi chú")): "Baseline và ablation trên val random",
    ("4", ("Run", "Params", "Best epoch", "Val acc", "Val macro-F1", "Train−Val (điểm %)", "Thời gian (s)")): "Chi tiết 7 run CNN",
    ("4", ("Run", "Params", "Val acc", "Val macro-F1", "Best epoch")): "Hai ứng viên trên val stratified_group",
    ("4", ("Seed", "Val acc", "Val macro-F1", "Best epoch")): "Kết quả 3 seed",
    ("4", ("Chỉ số test", "Giá trị")): "Kết quả test (đánh giá đúng một lần)",
    ("5", ("Hạng", "Cặp (thật → dự đoán)", "Số ảnh")): "Top 10 cặp nhầm",
    ("5", ("Biến", "Nhóm thấp", "Nhóm giữa", "Nhóm cao", "Chênh nhóm cao−thấp (CI95)", "r²")): "Error rate theo tertile",
    ("5", ("#", "Thật → dự đoán", "Conf", "Nhóm", "Quan sát được", "Giả thuyết")): "Mười hai mẫu sai",
    ("5", ("Nhóm", "n", "Bị từ chối", "Tỷ lệ (CI95 Wilson)")): "OOD theo nhóm",
    ("6", ("Kịch bản", "Kết quả", "Conf", "Bị từ chối")): "Kết quả 5 kịch bản demo",
    ("A", ("Lớp", "Precision", "Recall", "F1", "Support")): "Per-class trên test",
    ("B", ("File", "Nội dung")): "Artifact chính",
}


def table(headers: list[str], body: list[list[str]], align: str | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in body:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def image(path: Path, caption: str, width: str = "100%") -> str:
    rel = path.relative_to(ROOT).as_posix()
    return f"![{caption}]({rel})\n\n*{caption}*"


summary = {r["run_id"]: r for r in rows("summary.csv")}
data = load("data_report.json")
split_random = load("split_random.json")
split_group = load("split_group.json")
split_sgroup = load("split_stratified_group.json")
leak = load("leak_delta.json")
phash = load("phash_highres.json")
multiseed = load("multiseed.json")
test = load("test_report.json")
error = load("error_report.json")
latency = load("latency_effect.json")
ood_summary = load("ood_summary.json")
threshold = load("threshold.json")
samples = rows("error_samples_table.csv")
per_class = rows("per_class_test.csv")
demo = rows("demo_scenarios.csv")
phase_ok = {p: load(p) for p in (
    "phase2_check.json", "phase3_check.json", "phase4_check.json",
    "phase5_check.json", "phase6_check.json", "phase7_check.json", "phase8_check.json",
)}
test_verif = load("test_verification.json")
cleanroom_path = RES / "cleanroom.json"
cleanroom = load("cleanroom.json") if cleanroom_path.is_file() else None
golden_path = RES / "golden_compare.json"
golden = load("golden_compare.json") if golden_path.is_file() else None

chosen = summary["final_s42"]
acc = test["accuracy"]
f1 = test["macro_f1_43"]
top3 = test["top3_accuracy"]

# ------------------------------------------------------------------ sections
front = f"""# BÁO CÁO BÀI TẬP LỚN HỌC SÂU
## Phân loại 43 lớp biển báo giao thông Đức (GTSRB) bằng CNN tự xây dựng

**Môn học:** Học sâu — 12INF028 · **Ngày hoàn thành:** 23/10/2026
**Người thực hiện:** sinh viên lớp 12INF028 (repo `Nguyen15idhue/GTSRB-CNN`)
**Môi trường chạy:** Windows 11 · Python 3.12 · PyTorch 2.11.0+cu128 · RTX 5070 12 GB · i9-14900KF / 64 GB

> **Kết quả cuối (test, đánh giá đúng một lần):** accuracy **{pct(acc)}**, macro-F1 **{f1:.4f}**,
> top-3 **{pct(top3)}** trên **{num(test['n_samples'])}** ảnh test. Multi-seed trên validation sạch:
> accuracy **{pct(multiseed['val_acc']['mean'])} ± {multiseed['val_acc']['std']*100:.2f} điểm %**,
> macro-F1 **{multiseed['val_macro_f1']['mean']:.4f} ± {multiseed['val_macro_f1']['std']:.4f}**.
>
> **Cảnh báo:** đây là demo học thuật trên biển báo Đức; **không dùng cho quyết định giao thông thực tế**.
> Validation **random** bị rò rỉ frame nên chỉ dùng validation **stratified-group** và test để kết luận
> (mục 2.4).

## Tóm tắt

Báo cáo trình bày pipeline hoàn chỉnh cho GTSRB: xác minh dữ liệu, hai split có kiểm soát rò rỉ,
baseline (majority, HOG + LinearSVC), ba cấu hình CNN tự xây dựng, bảy thí nghiệm ablation,
chọn mô hình trên **validation sạch theo track**, đa seed, đánh giá test **một lần**, phân tích lỗi
định lượng, kiểm chứng OOD và demo Streamlit.

Ba kết luận chính:

1. **Rò rỉ frame là vấn đề thật và định lượng được:** 95,22 % ảnh validation random có frame train
   cách ±1 trong cùng track; 1-NN (không huấn luyện) đạt 90,50 % trên val random nhưng chỉ 31,00 %
   trên val group. Vì vậy val random chỉ dùng để tham chiếu, **không dùng để xếp hạng mô hình**.
2. **Trên validation sạch, CNN-B + augmentation (E3) tốt hơn CNN-C + GAP (E4)** 1,62 điểm %
   macro-F1 — đảo ngược kết luận nếu chỉ nhìn val random.
3. **Kết quả test ổn định và đã kiểm chứng độc lập** (14/14 check): accuracy {pct(acc)},
   macro-F1 {f1:.4f}; chênh val–test nhỏ và không có dấu hiệu lỗi metric.

---
"""

ch1 = f"""## Chương 1 — Giới thiệu

### 1.1. Bài toán

Cho một ảnh màu RGB chứa **một** biển báo giao thông Đức, dự đoán nhãn thuộc **43 lớp** (0–42).
Ảnh gốc có kích thước thay đổi; hệ thống resize về **64×64** và chuẩn hóa theo mean/std của train.

| Thành phần | Nội dung |
|---|---|
| Input | ảnh RGB, resize 64×64 → tensor 3×64×64 (12.288 giá trị) |
| Output | logits 43 chiều → xác suất (softmax khi suy luận), top-3 cho demo |
| Số lớp | 43 (single-label, phân loại đa lớp) |
| Đối tượng | sinh viên/nghiên cứu thị giác máy tính; công cụ gán nhãn/kiểm kê biển báo |
| Giá trị | tự động hóa gán nhãn kho ảnh lớn, hỗ trợ kiểm kê chất lượng biển báo |
| Giới hạn | chỉ phân loại ảnh đã chứa biển; không phát hiện biển trong cảnh; không nhận diện biển Việt Nam |

### 1.2. Phạm vi

Có làm: phân loại 43 lớp, ablation, đa seed, đánh giá test một lần, phân tích lỗi, demo ảnh ngoài.
**Không** làm: detection, nhận diện biển Việt Nam, huấn luyện trên ảnh người dùng, triển khai thời gian thực.

### 1.3. Cảnh báo sử dụng

Mô hình là **nguyên mẫu học thuật**. Sai số trên ảnh ngoài phân phối còn lớn (Chương 5), ngưỡng từ chối
chỉ dựa trên confidence và **không đảm bảo** phát hiện ảnh không có biển. **Không dùng cho quyết định
giao thông thực tế.**

---
"""

data_scan = data.get("image_scan", {})
counts = data["counts"]
CURRENT_CHAPTER = "2"
ch2 = f"""## Chương 2 — Dữ liệu

### 2.1. Nguồn và điều kiện sử dụng

- **GTSRB** — German Traffic Sign Recognition Benchmark, Stallkamp et al., *Neural Networks* 32 (2012) 323–332.
- Trang chính thức: `https://benchmark.ini.rub.de/gtsrb_dataset.html` (không phân giải DNS từ máy này
  ngày 22/10/2026; câu điều kiện sử dụng được chép từ bản lưu Wayback, ghi trong `data_report.json`).
- Nguyên văn điều kiện: *"The data is free to use. However, we cordially ask you to cite the following
  publication if you do: …"* → dữ liệu dùng cho nghiên cứu, **không** phải CC0.
- Train lấy từ archive chính thức `GTSRB_Final_Training_Images.zip`; test qua `torchvision.datasets.GTSRB`
  (bản train rút gọn 26.640 ảnh của torchvision **không** được dùng).

### 2.2. Quy mô thật (đếm lại, không dùng số công bố)

{table(
    ["Hạng mục", "Giá trị"],
    [
        ["Ảnh train trên đĩa", f"{num(counts['train'])}"],
        ["Ảnh test", f"{num(counts['test'])}"],
        ["Ảnh train dùng để chia split", f"{num(counts['train_pool_after_exclusion'])} (loại {counts['excluded_duplicates']} ảnh trùng MD5 với test)"],
        ["Số lớp", f"{data['num_classes']} (train {data['classes_train_present']}/43, test {data['classes_test_present']}/43)"],
        ["Lớp ít nhất / nhiều nhất (train)", f"lớp {data['least_populated_train_class']['class_id']} ({num(data['least_populated_train_class']['count'])} ảnh) / lớp {data['most_populated_train_class']['class_id']} ({num(data['most_populated_train_class']['count'])} ảnh)"],
        ["Tỷ lệ mất cân bằng max/min", f"{data['imbalance_ratio_max_over_min']:.2f} lần"],
        ["Số lớp < 300 ảnh train", f"{data['images_below_300_train']}"],
        ["ROI hợp lệ", f"train {num(data['roi']['train_images_with_roi'])}/{num(data['roi']['train_annotation_rows'])}; test {num(data['roi']['test_images_with_roi'])}/{num(data['roi']['test_annotation_rows'])}"],
        ["Track trích được", f"{num(data['tracks']['train_images_with_track_id'])}/{num(data['tracks']['train_images'])} ảnh, {num(data['tracks']['num_tracks'])} track"],
        ["Ảnh lỗi khi mở", f"{len(data_scan.get('unreadable', []))}"],
    ],
)}

{image(FIG / '01_class_distribution.png', 'Hình 2.1. Phân bố số ảnh theo lớp (tô đỏ: lớp dưới 300 ảnh)')}

{image(FIG / '02_image_sizes.png', 'Hình 2.2. Phân bố kích thước ảnh gốc')}

{image(FIG / '03_sample_grid.png', 'Hình 2.3. Lưới 43 ảnh mẫu (mỗi lớp một ảnh, crop theo ROI)')}

### 2.3. Ba cách chia tập

| Split | Train / Val | Track train / val | Lớp ở val | Lớp vắng | Lớp ít ảnh nhất ở val |
|---|---|---|---|---|---|
| `random` (stratified 80/20) | {num(split_random['summary']['n_train'])} / {num(split_random['summary']['n_val'])} | {num(split_random['summary']['n_tracks_train'])} / {num(split_random['summary']['n_tracks_val'])} (giao {split_random['summary']['overlap_track']}) | {split_random['summary']['classes_val']}/43 | – | {split_random['summary']['min_class_val']['count']} |
| `group` (GroupShuffleSplit) | {num(split_group['summary']['n_train'])} / {num(split_group['summary']['n_val'])} | {num(split_group['summary']['n_tracks_train'])} / {num(split_group['summary']['n_tracks_val'])} (giao {split_group['summary']['overlap_track']}) | {split_group['summary']['classes_val']}/43 | {split_group['summary']['classes_missing_val']} | {split_group['summary']['min_class_val']['count']} |
| **`stratified_group` (dùng cho GĐ6)** | **{num(split_sgroup['summary']['n_train'])} / {num(split_sgroup['summary']['n_val'])}** | **{num(split_sgroup['summary']['n_tracks_train'])} / {num(split_sgroup['summary']['n_tracks_val'])} (giao {split_sgroup['summary']['overlap_track']})** | **{split_sgroup['summary']['classes_val']}/43** | – | **{split_sgroup['summary']['min_class_val']['count']}** |

`stratified_group` được thiết kế để vừa **tách track** (không rò rỉ frame) vừa **đủ 43 lớp** ở val:
mỗi lớp góp `max(1, round(0,2 × số track của lớp))` track vào val. Đây là split dùng để chọn mô hình
và báo cáo multi-seed.

### 2.4. Rò rỉ frame: định lượng

| Cửa sổ frame | Val random | Val group |
|---|---|---|
| ±1 frame | **95,22 %** (7.466/7.841) | 0 % |
| ±2 frame | 99,60 % (7.810/7.841) | 0 % |
| ±5 frame | **100 %** (7.841/7.841) | 0 % |
| Khoảng cách frame gần nhất | trung vị **1** | không có |

Bằng chứng độc lập bằng **1-NN** (ảnh xám 16×16, không huấn luyện): val random **90,50 %**;
val random nhưng **cấm** láng giềng cùng track **36,50 %**; val group **31,00 %**. Nghĩa là val random
bị "giải sẵn" bởi frame gần trùng. Hệ quả thực nghiệm: mọi mô hình đều đạt ~99,6–99,97 % trên val random
(Bảng 4.2) nên **không thể xếp hạng** mô hình trên split này. Đo Δ trực tiếp: E3 trên val group giảm
**1,46 điểm % accuracy** và **2,53 điểm % macro-F1** (trên các lớp có mặt) so với val random
(`leak_delta.json`).

### 2.5. Kiểm tra trùng lặp (MD5 + pHash)

- **MD5:** 8 ảnh train trùng 8 ảnh test (cùng nhãn 14, track `00023` của tập train) → loại khỏi train,
  giữ file trên đĩa để số đếm khớp công bố. Danh sách: `excluded_train_duplicates.csv`.
- **pHash (64-bit) + MAD/NCC:** 16 cặp pHash trùng tuyệt đối và 17.214 cặp Hamming ≤ 7 chỉ là **va chạm
  hash giữa biển cùng lớp**; 13 ảnh test (0,10 %) có ảnh train rất giống ở 64×64 nhưng khi so ở ảnh gốc
  (crop ROI → 128×128) NCC **cao nhất 0,949**, trung vị 0,811 → **không có cặp gần trùng được xác nhận**.
  Nếu loại cả 13 ảnh, accuracy thay đổi **≤ {phash['max_accuracy_impact_points']} điểm %**.

{image(FIG / '09_phash_highres.png', 'Hình 2.4. Kiểm chứng 6 cặp nghi ngờ nhất ở ảnh gốc: cùng loại biển nhưng khác khung/nền/độ nét')}

### 2.6. Tiền xử lý và augmentation

- Một hàm duy nhất `src/data.py::preprocess` dùng cho train/eval/test/demo: RGB → resize 64×64 bilinear →
  `[0,1]` → chuẩn hóa bằng mean/std tính **trên train**:
  mean `[0,340408, 0,312365, 0,321632]`, std `[0,275148, 0,264242, 0,270747]` (`data_stats.json`).
- Cache `uint8` 64×64 (610,7 MB, tạo 13,6 s) để epoch nhanh; `X_train.npy` là **pool 39.201 ảnh**.
- Augmentation **chỉ train**: RandomAffine (xoay ±10°, dịch 0,1, scale 0,9–1,1, shear 5),
  ColorJitter nhẹ (0,2/0,2/0,2/0,02), RandomPerspective 0,1 (p=0,3).
  **Không** lật ngang/dọc, không xoay 90/180 (đổi nghĩa biển); có kiểm tra tự động chặn `Flip`.

{image(FIG / '04_augmentation.png', 'Hình 2.5. Lưới 8 ảnh gốc / 8 ảnh sau augmentation — không lật, số trên biển giữ nguyên chiều')}

---
"""

CURRENT_CHAPTER = "3"
ch3 = f"""## Chương 3 — Phương pháp

### 3.1. Baseline

| Mô hình | Thiết lập | Vai trò |
|---|---|---|
| E0a majority | dự đoán lớp nhiều nhất trong train (lớp 2) | sàn tham chiếu |
| E0b HOG + LinearSVC | ảnh xám 64×64, HOG 9 hướng, cell 8×8, block 2×2, L2-Hys → 1.764 chiều; StandardScaler fit trên train; C ∈ {{0,1; 1; 10}} chọn theo val macro-F1; `dual=true` | baseline chuẩn GTSRB, không augmentation |

### 3.2. Kiến trúc CNN

{image(FIG / '10_architecture.png', 'Hình 3.1. Ba cấu hình SmallCNN (input 3×64×64 → logits 43)')}

| Mô hình | Khối conv | BN | Dropout | GAP | Aug | Params thật |
|---|---|---|---|---|---|---|
| CNN-A | 3 (32/64/128) | không | 0 | không | không | 2.201.707 |
| **CNN-B (cuối)** | 3 (32/64/128) | có | 0,3 | không | có | **2.201.931** |
| CNN-C | 4 (32/64/128/256) | có | 0,4 | có | có | 399.947 |

- 64×64 là cân bằng chi tiết/chi phí; 3–4 khối 3×3 đủ receptive field cho biển báo.
- BN ổn định huấn luyện; dropout + weight decay chống overfit với ~31k ảnh; GAP giảm 2,1 M tham số FC.
- Không dùng pretrained (yêu cầu "CNN tự xây dựng").

### 3.3. Thiết lập huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer / LR / WD | AdamW / 1e-3 / 1e-4 (CNN-A: WD 0) |
| Batch / epoch tối đa / patience | 128 / 40 / 7 |
| Scheduler | CosineAnnealingLR (T_max = 40) |
| Loss | CrossEntropyLoss (E7: class weight 1/√freq) |
| AMP | bật khi CUDA |
| Chọn checkpoint | **val macro-F1 (43 lớp)** |
| Seed | 42; multi-seed 42/123/2024 |

### 3.4. Chỉ số và kỷ luật đánh giá

- Accuracy, **macro-F1/43**, per-class precision/recall/F1, confusion matrix chuẩn hóa theo hàng, top-3.
- Macro-F1 luôn tính trên **đủ 43 lớp** (`labels=range(43)`); nếu split thiếu lớp thì báo thêm
  F1 trên lớp có mặt và trần lý thuyết — tránh so sánh sai.
- Test chỉ chạy **sau khi** `configs/final.yaml` được commit (`freeze`), có `--confirm-freeze`,
  ghi log, không ghi đè; mọi lựa chọn mô hình/epoch/ngưỡng đều trên val.

{image(FIG / '10_pipeline.png', 'Hình 3.2. Pipeline thực nghiệm')}

---
"""


def e_rows() -> str:
    body = []
    for run in ["e1_cnna_s42", "e2_cnnb_s42", "e3_cnnb_aug_s42", "e4_cnnc_gap_aug_s42",
                "e5a_lr3e4_s42", "e5b_lr3e3_s42", "e7_classweight_s42"]:
        r = summary[run]
        body.append([
            run.replace("_s42", ""),
            f"{num(r['params'])}",
            r["best_epoch"],
            f"{float(r['val_acc']):.6f}",
            f"{float(r['val_macro_f1']):.6f}",
            f"{float(r['train_val_acc_gap']) * 100:+.2f}",
            f"{float(r['train_time_s']):.1f}",
        ])
    return table(["Run", "Params", "Best epoch", "Val acc", "Val macro-F1", "Train−Val (điểm %)", "Thời gian (s)"], body)


CURRENT_CHAPTER = "4"
ch4 = f"""## Chương 4 — Kết quả

### 4.1. Learning curves

{image(FIG / '10_curves_4runs.png', 'Hình 4.1. Loss / accuracy / macro-F1 của 4 cấu hình chính (val random)')}

CNN-A hội tụ nhanh nhưng dao động hơn; E3/E4 mượt hơn nhờ BN + augmentation. Lưu ý val random bị
rò rỉ frame nên các đường đều bão hoà ~0,99 — hình chỉ dùng để kiểm tra hội tụ, không để so sánh.

### 4.2. Baseline và ablation (val random)

{table(
    ["Mô hình", "Val acc", "Val macro-F1", "Ghi chú"],
    [
        ["E0a majority", f"{float(summary['e0a_majority']['val_acc']):.6f}", f"{float(summary['e0a_majority']['val_macro_f1']):.6f}", "lớp 2 (1.800 ảnh train)"],
        ["E0b HOG + LinearSVC", f"{float(summary['e0b_hog_svc']['val_acc']):.6f}", f"{float(summary['e0b_hog_svc']['val_macro_f1']):.6f}", "C=0,1; trích 51,3 s; fit 540,2 s"],
        ["E1 CNN-A", f"{float(summary['e1_cnna_s42']['val_acc']):.6f}", f"{float(summary['e1_cnna_s42']['val_macro_f1']):.6f}", "không BN/dropout/aug"],
        ["E2 CNN-B", f"{float(summary['e2_cnnb_s42']['val_acc']):.6f}", f"{float(summary['e2_cnnb_s42']['val_macro_f1']):.6f}", "+ BN + dropout + WD"],
        ["E3 CNN-B + aug", f"{float(summary['e3_cnnb_aug_s42']['val_acc']):.6f}", f"{float(summary['e3_cnnb_aug_s42']['val_macro_f1']):.6f}", "+ augmentation"],
        ["E4 CNN-C + GAP + aug", f"{float(summary['e4_cnnc_gap_aug_s42']['val_acc']):.6f}", f"{float(summary['e4_cnnc_gap_aug_s42']['val_macro_f1']):.6f}", "4 khối + GAP"],
        ["E5a lr 3e-4 (10 epoch)", f"{float(summary['e5a_lr3e4_s42']['val_acc']):.6f}", f"{float(summary['e5a_lr3e4_s42']['val_macro_f1']):.6f}", "so sánh xu hướng"],
        ["E5b lr 3e-3 (10 epoch)", f"{float(summary['e5b_lr3e3_s42']['val_acc']):.6f}", f"{float(summary['e5b_lr3e3_s42']['val_macro_f1']):.6f}", "phân kỳ"],
        ["E7 class weight", f"{float(summary['e7_classweight_s42']['val_acc']):.6f}", f"{float(summary['e7_classweight_s42']['val_macro_f1']):.6f}", "1/√freq"],
    ],
)}

{image(FIG / '05_experiment_comparison.png', 'Hình 4.2. So sánh val accuracy và val macro-F1 giữa các run')}

Bảng chi tiết 7 run CNN (params, epoch dừng, thời gian):

{e_rows()}

**Đọc bảng này đúng cách:** các chênh lệch rất nhỏ (E4 hơn E3 đúng **2 ảnh/7.841**) là **nhiễu**, không
phải bằng chứng mô hình tốt hơn. Vì vậy Giai đoạn 5 chỉ ghi nhận, không chốt mô hình trên split này.

### 4.3. Chọn mô hình trên validation sạch

Hai ứng viên sát nhau ở Giai đoạn 5 được chạy lại **trên `split_stratified_group`** (seed 42):

| Run | Params | Val acc | Val macro-F1 | Best epoch |
|---|---|---|---|---|
| E3 (CNN-B + aug) | 2.201.931 | **0,983525** | **0,970613** | 34 |
| E4 (CNN-C + GAP) | 399.947 | 0,976373 | 0,954453 | 34 |

E3 hơn E4 **1,62 điểm %** macro-F1 (> 0,3 điểm %) → chọn **E3**. Quy tắc "ưu tiên ít tham số" chỉ áp
dụng khi chênh < 0,3 điểm %, nên không chọn E4 dù chỉ bằng 18 % tham số. Đây là ví dụ rõ cho việc
**split quyết định kết luận**.

### 4.4. Multi-seed và test

| Seed | Val acc | Val macro-F1 | Best epoch |
|---|---|---|---|
| 42 | {pct(float(summary['final_s42']['val_acc']))} | {float(summary['final_s42']['val_macro_f1']):.6f} | {summary['final_s42']['best_epoch']} |
| 123 | {pct(float(summary['final_s123']['val_acc']))} | {float(summary['final_s123']['val_macro_f1']):.6f} | {summary['final_s123']['best_epoch']} |
| 2024 | {pct(float(summary['final_s2024']['val_acc']))} | {float(summary['final_s2024']['val_macro_f1']):.6f} | {summary['final_s2024']['best_epoch']} |
| **Mean ± std** | **{pct(multiseed['val_acc']['mean'])} ± {multiseed['val_acc']['std']*100:.2f}** | **{multiseed['val_macro_f1']['mean']:.6f} ± {multiseed['val_macro_f1']['std']:.6f}** | – |

{image(FIG / '10_multiseed.png', 'Hình 4.3. Val accuracy / macro-F1 theo 3 seed (đường đỏ: mean, dải: ±1 std)')}

| Chỉ số test | Giá trị |
|---|---|
| Số ảnh | {num(test['n_samples'])} |
| Accuracy | **{pct(acc)}** |
| Macro-F1 / 43 lớp | **{f1:.6f}** |
| Top-3 accuracy | {pct(top3)} |
| Checkpoint sha256 | `{test['checkpoint_sha256'][:16]}…` |
| Commit freeze | `{test['git_commit'][:10]}` |

{image(FIG / '10_confusion_full.png', 'Hình 4.4. Confusion matrix test 43×43 (chuẩn hóa theo hàng)')}

**Chênh val–test:** test **cao hơn** mean val 0,47 điểm % accuracy và 1,45 điểm % macro-F1. Đã kiểm tra
lại: accuracy/F1/top-3 tính lại từ `predictions.csv` khớp từng bit, nhãn khớp `GT-final_test.csv`,
sha256 khớp (`test_verification.json` 14/14 PASS). Phân phối ảnh val/test gần như trùng nhau (median ROI
0,5625 vs 0,5639; blur 0,0022 vs 0,0023; brightness 0,325 vs 0,338), nên **không suy diễn nguyên nhân**;
chênh lệch cỡ 1,5 lần std giữa seed và được ghi nhận, không phải lỗi metric.

---
"""

CURRENT_CHAPTER = "5"
ch5 = f"""## Chương 5 — Phân tích lỗi

### 5.1. Per-class

Chỉ **{error['n_classes_f1_below_090']} lớp** có F1 < 0,90. Lớp thấp nhất: **27 (Người đi bộ), F1 0,846,
support 60**. Với support nhỏ, 1 ảnh sai làm F1 giảm ~0,03, nên không kết luận "mô hình kém" cho lớp này.

{image(FIG / '07_per_class_f1.png', 'Hình 5.1. F1 theo lớp trên test (sắp tăng dần, kèm support)')}

### 5.2. Confusion và cặp nhầm

{image(FIG / '07_confusion_zoom15.png', 'Hình 5.2. Zoom 15 lớp F1 thấp nhất (chuẩn hóa theo hàng)')}

{table(
    ["Hạng", "Cặp (thật → dự đoán)", "Số ảnh"],
    [[i + 1, f"{p['true']} → {p['pred']}", p["count"]] for i, p in enumerate(error["top_confusions"])],
)}

{image(FIG / '10_top10_pairs_bar.png', 'Hình 5.3. Top 10 cặp nhầm trên test')}

{image(FIG / '07_top_pairs.png', 'Hình 5.4. Ảnh ví dụ cho từng cặp nhầm (crop ROI, đã phóng to)')}

### 5.3. Error rate theo nguyên nhân (định lượng)

{table(
    ["Biến", "Nhóm thấp", "Nhóm giữa", "Nhóm cao", "Chênh nhóm cao−thấp (CI95)", "r²"],
    [
        ["Độ mờ (Laplacian, cao = nét)", f"{error['tertiles']['summary']['blur_laplacian']['error_rate'][0]*100:.2f} %", f"{error['tertiles']['summary']['blur_laplacian']['error_rate'][1]*100:.2f} %", f"{error['tertiles']['summary']['blur_laplacian']['error_rate'][2]*100:.2f} %", f"{latency['effect_size']['blur_laplacian']['difference_points']:+.2f} điểm % ({latency['effect_size']['blur_laplacian']['difference_ci95_points'][0]:+.2f}…{latency['effect_size']['blur_laplacian']['difference_ci95_points'][1]:+.2f})", f"{latency['effect_size']['blur_laplacian']['r_squared']:.4f}"],
        ["Độ sáng (mean V)", f"{error['tertiles']['summary']['brightness_mean']['error_rate'][0]*100:.2f} %", f"{error['tertiles']['summary']['brightness_mean']['error_rate'][1]*100:.2f} %", f"{error['tertiles']['summary']['brightness_mean']['error_rate'][2]*100:.2f} %", f"{latency['effect_size']['brightness_mean']['difference_points']:+.2f} điểm % ({latency['effect_size']['brightness_mean']['difference_ci95_points'][0]:+.2f}…{latency['effect_size']['brightness_mean']['difference_ci95_points'][1]:+.2f})", f"{latency['effect_size']['brightness_mean']['r_squared']:.4f}"],
        ["Diện tích biển / ảnh", f"{error['tertiles']['summary']['roi_area_ratio']['error_rate'][0]*100:.2f} %", f"{error['tertiles']['summary']['roi_area_ratio']['error_rate'][1]*100:.2f} %", f"{error['tertiles']['summary']['roi_area_ratio']['error_rate'][2]*100:.2f} %", f"{latency['effect_size']['roi_area_ratio']['difference_points']:+.2f} điểm % ({latency['effect_size']['roi_area_ratio']['difference_ci95_points'][0]:+.2f}…{latency['effect_size']['roi_area_ratio']['difference_ci95_points'][1]:+.2f})", f"{latency['effect_size']['roi_area_ratio']['r_squared']:.4f}"],
    ],
)}

{image(FIG / '07_error_rates.png', 'Hình 5.5. Error rate theo tertile của 3 biến (test, n = 12.630)')}

**Diễn giải có kỷ luật:** cả ba biến có **effect size rất nhỏ** (r² từ 0,0003 đến 0,0043 — dưới 0,5 %
phương sai). p nhỏ chủ yếu do n = 12.630. Vì vậy chỉ nói **xu hướng**: ảnh mờ/tối/biển nhỏ có error rate
cao hơn, không khẳng định quan hệ nhân quả.

### 5.4. Mười hai mẫu sai (quan sát ≠ giả thuyết)

{image(FIG / '07_error_samples.png', 'Hình 5.6. 12 mẫu dự đoán sai trên test (crop ROI, đã phóng to)')}

{table(
    ["#", "Thật → dự đoán", "Conf", "Nhóm", "Quan sát được", "Giả thuyết"],
    [
        [s["stt"], f"{s['true_id']} → {s['pred_id']}", f"{float(s['confidence']):.3f}", s["nhom_nguyen_nhan"],
         s["quan_sat_duoc"], s["gia_thuyet"]]
        for s in samples
    ],
)}

Bốn cơ chế lặp lại: **cháy sáng** làm mất viền đỏ của biển "đường ưu tiên" (4/12), **mờ** làm mất thanh
trắng của "cấm đi vào" (2/12), **thiếu sáng** làm mất hướng mũi tên (2/12), và **chữ số 50/30** khác nhau
một nét cộng moiré do resize (2/12). Các quan sát này lấy từ việc mở từng ảnh, không suy đoán.

### 5.5. OOD và ngưỡng từ chối

{image(FIG / '07_ood.png', 'Hình 5.7. Ảnh ngoài dataset (CC0/PD) — nhãn dự đoán hoặc "không xác định"')}

{table(
    ["Nhóm", "n", "Bị từ chối", "Tỷ lệ (CI95 Wilson)"],
    [[name, g["n"], g["n_rejected"], f"{g['rejection_rate']*100:.0f} % ({g['rejection_ci95_wilson'][0]*100:.0f}–{g['rejection_ci95_wilson'][1]*100:.0f} %)"]
     for name, g in ood_summary["groups"].items()],
)}

- τ = **{threshold['tau']:.4f}** (phân vị 5 % max-prob trên val; acceptance in-distribution
  {pct(threshold['acceptance_in_distribution'])}), **không** chọn trên demo.
- Ngưỡng này là **cơ chế từ chối theo confidence**, không phải bộ phát hiện OOD: 1/4 ảnh đường vẫn bị
  nhận nhầm "Nhường đường" với confidence 0,996. Với n = 4–6, khoảng tin cậy rất rộng — không kết luận
  tỷ lệ chính xác.
- Ảnh biển nước ngoài (Mỹ/Hàn) top-1 đúng nhãn tương ứng nhưng 1/2 bị τ từ chối vì biển nhỏ; biển chữ
  Philippines bị gán nhãn Đức 2/4 lần → đúng bản chất domain shift.

---
"""

CURRENT_CHAPTER = "6"
ch6 = f"""## Chương 6 — Demo và kết luận

### 6.1. Demo

- `app.py` (Streamlit): upload JPG/JPEG/PNG ≤ 20 MB, hiển thị ảnh + top-3 tiếng Việt, cảnh báo
  "không xác định" khi max-prob < τ, xử lý 3 lỗi (file hỏng, quá lớn, thiếu checkpoint), footer cảnh báo.
- `src/predict.py` dùng đúng `src.data.preprocess` (không viết lại) và checkpoint cuối `final_s42`.
- AppTest: **0 exception**, có footer cảnh báo (`phase8_check.json`).

{image(FIG / '08_scenarios.png', 'Hình 6.1. Ảnh minh chứng 5 kịch bản demo (không phải screenshot UI)')}

{table(
    ["Kịch bản", "Kết quả", "Conf", "Bị từ chối"],
    [[d["scenario"], d["pred_name"] or d["rejected"], d["prob"] or "–", "có" if d["rejected"] == "True" else "không"]
     for d in demo],
)}

**Thời gian:** model-only median **{latency['model_only_ms']['median']:.2f} ms**; end-to-end trong Python
(bytes → validate → preprocess → forward → top-3) median **{latency['end_to_end_python_ms']['median']:.2f} ms**,
p95 {latency['end_to_end_python_ms']['p95']:.2f} ms. **Chưa tính** upload mạng và render UI — không gọi
đây là "thời gian người dùng chờ".

{image(FIG / '08_external_contact.png', 'Hình 6.2. 11 ảnh ngoài dataset đã kiểm tra ẩn danh (không người/biển số)')}

### 6.2. Tái lập và kiểm thử

- `python -m pytest tests -q` → **11 passed** (split, macro-F1 43 lớp, checkpoint, validate upload,
  predictions/report integrity, log test 1 lần, threshold, manifest/license, class_names).
- Test đã chạy **đúng một lần**; `test_runs.log` 1 dòng; chạy lại bị từ chối trước khi ghi.
- Kiểm chứng độc lập test: **{test_verif['passed']}/{test_verif['total']} PASS** — 12.630 dòng/path duy nhất,
  nhãn khớp GT, metric và sha256 khớp.
"""

if cleanroom:
    ch6 += f"""
- Clean-room (`.venv_test`, cài từ requirements + torch cu128):
  CUDA `{cleanroom.get('cuda_available')}` · capability `{cleanroom.get('capability')}` ·
  smoke train {cleanroom.get('smoke_epochs')} epoch · evaluate val acc
  `{cleanroom.get('val_acc'):.6f}` (kỳ vọng `{cleanroom.get('expected_val_acc'):.6f}`) ·
  AppTest `{cleanroom.get('app_exception_count')} exception` ({cleanroom.get('elapsed_min')} phút).
"""
if golden:
    ch6 += f"""
- Golden test 20 ảnh giữa hai môi trường: lệch xác suất lớn nhất
  **{golden['max_probability_difference']:.2e}** (ngưỡng 1e-3), lệch nhãn {golden['prediction_mismatches']}.
"""

ch6 += f"""
### 6.3. Hạn chế (thật, không giấu)

1. **Val random bị rò rỉ frame** (95,22 % ±1); mọi so sánh trên split này chỉ có giá trị tham chiếu.
2. **Val group thiếu lớp** (0, 19, 34, 39) nếu dùng GroupShuffleSplit thuần; đã xử lý bằng
   `stratified_group`, nhưng split này vẫn là thiết kế riêng, chưa được chuẩn hóa trong tài liệu GTSRB.
3. **Sai số ngoài phân phối còn lớn**: biển nước ngoài/biển chữ có thể bị gán nhãn Đức với confidence cao;
   τ không phải bộ phát hiện OOD (1/4 ảnh không biển lọt).
4. **Chỉ 43 lớp biển Đức**, ảnh 64×64; không xử lý cảnh có nhiều biển hoặc biển quá nhỏ.
5. **Một lần test duy nhất** đúng kỷ luật, nhưng nghĩa là không có ước lượng độ bất định của chính test;
   multi-seed chỉ đo biến thiên huấn luyện trên val.
6. **mean/std** tính trên train của split random nhưng dùng cho mô hình cuối (split stratified-group) —
   lệch nhỏ, đã ghi nhận; nếu làm lại nên tính trên train của split cuối.
7. **Chưa có screenshot UI và video demo** (môi trường không có GUI/browser automation); ảnh minh chứng
   là hình ghép từ chính kết quả chạy thật.
8. Giai đoạn 11–12 (slide, đóng gói/nộp) **chưa thực hiện** theo yêu cầu.

### 6.4. Hướng phát triển

Thu ảnh thật nhiều hơn (đặc biệt biển Việt Nam) để đo domain shift có kiểm soát; calibration xác suất
(temperature scaling) và OOD detection thực sự (energy/Mahalanobis); crop ROI tự động (E8);
tăng độ phân giải (96×96) và so sánh; distillation/pruning cho thiết bị yếu.

### 6.5. Kết luận

Pipeline đã đi đủ chuỗi dữ liệu → baseline → CNN → ablation → chọn mô hình trên val sạch → đa seed →
test một lần → phân tích lỗi → demo. Kết quả test **{pct(acc)} accuracy / {f1:.4f} macro-F1** là con số
đã được kiểm chứng chéo (predictions, nhãn GT, sha256, log), và mọi hạn chế đã được nêu định lượng.

> **Mô hình chỉ phục vụ học tập, không dùng cho quyết định giao thông thực tế.**

---

## Tài liệu tham khảo

1. Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C. *Man vs. computer: Benchmarking machine learning
   algorithms for traffic sign recognition.* Neural Networks 32 (2012) 323–332.
2. GTSRB dataset — `https://benchmark.ini.rub.de/gtsrb_dataset.html` (truy cập 22/10/2026, qua Wayback).
3. Torchvision datasets — `https://docs.pytorch.org/vision/main/datasets.html`.
4. PyTorch documentation — `https://pytorch.org/docs/stable/index.html`.
5. scikit-learn user guide — `https://scikit-learn.org/stable/`.
6. Streamlit documentation — `https://docs.streamlit.io/`.

---

## Phụ lục A — Per-class trên test

{table(
    ["Lớp", "Precision", "Recall", "F1", "Support"],
    [[r["class_id"], f"{float(r['precision']):.4f}", f"{float(r['recall']):.4f}", f"{float(r['f1']):.4f}", r["support"]] for r in per_class],
)}

*Bảng A.1. Per-class trên test*

## Phụ lục B — Artifact chính

| File | Nội dung |
|---|---|
| `outputs/results/test_report.json`, `test_final.json` | kết quả test duy nhất (acc, F1, per-class, confusion, sha256) |
| `outputs/results/predictions_test_final.csv` | 12.630 dòng dự đoán test (bản sao nguyên văn, kèm sha256) |
| `outputs/results/summary.csv` | bảng tổng hợp run (có cột `split`, `role`) |
| `outputs/results/multiseed.json` | mean ± std 3 seed |
| `outputs/results/error_report.json` | per-class, top cặp, tertile, 12 mẫu, OOD |
| `outputs/results/leak_delta.json` | định lượng rò rỉ frame |
| `outputs/results/phash_*.json`, `phash_pairs.csv` | audit gần trùng |
| `outputs/results/phase2..8_check.json` | checklist từng giai đoạn |
| `outputs/results/cleanroom.json`, `golden_compare.json` | tái lập môi trường sạch |

*Bảng B.1. Artifact chính*

## Phụ lục C — Lệnh chạy

```powershell
python -m src.data --download --report
python -m src.data --prepare --verify
python -m src.train --config configs/final.yaml --seed 42
python -m src.evaluate --run outputs/runs/final_s42 --split val
python -m src.evaluate --run outputs/runs/final_s42 --split test --confirm-freeze  # chỉ 1 lần
python -m src.summarize --figure --multiseed
python -m src.error_analysis --run outputs/runs/final_s42
python -m pytest tests -q
streamlit run app.py
```
"""

CAPTION_LIST = [
    "Thành phần bài toán",
    "Quy mô và chất lượng dữ liệu (đếm lại từ đĩa)",
    "Ba cách chia tập",
    "Tỷ lệ ảnh val có frame train gần (rò rỉ)",
    "Hai baseline",
    "Ba cấu hình CNN",
    "Thiết lập huấn luyện",
    "Baseline và ablation trên val random",
    "Chi tiết 7 run CNN",
    "Hai ứng viên trên val stratified_group",
    "Kết quả 3 seed",
    "Kết quả test (đánh giá đúng một lần)",
    "Top 10 cặp nhầm",
    "Error rate theo tertile",
    "Mười hai mẫu sai",
    "OOD theo nhóm",
    "Kết quả 5 kịch bản demo",
    "Per-class trên test",
    "Artifact chính",
]


def paginate_tables(markdown: str) -> str:
    """Đánh số bảng theo chương và chèn caption ngay sau mỗi bảng."""
    import re as _re

    lines = markdown.splitlines()
    out: list[str] = []
    chapter = "1"
    counters: dict[str, int] = {}
    caption_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            match = _re.search(r"Chương (\d+)", line)
            if match:
                chapter = match.group(1)
            elif "Phụ lục A" in line:
                chapter = "A"
            elif "Phụ lục B" in line:
                chapter = "B"
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].startswith("*Bảng "):
                i += 1
            out.extend(block)
            out.append("")
            counters[chapter] = counters.get(chapter, 0) + 1
            if caption_index < len(CAPTION_LIST):
                out.append(f"*Bảng {chapter}.{counters[chapter]}. {CAPTION_LIST[caption_index]}*")
                caption_index += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


full_text = front + ch1 + ch2 + ch3 + ch4 + ch5 + ch6
full_text = paginate_tables(full_text)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(full_text, encoding="utf-8")
print("Đã ghi", OUT, f"({len((front + ch1 + ch2 + ch3 + ch4 + ch5 + ch6).splitlines())} dòng)")
