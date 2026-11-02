# BỘ PROMPT TRIỂN KHAI KẾ HOẠCH V2

> Dùng khi người dùng **đã yêu cầu rõ ràng** việc commit/push sau mỗi bước.
> Nguồn sự thật: `AGENTS.md`, `KE_HOACH_V2_DAY_DU.md`, `CHECKLIST_TRIEN_KHAI.md`, `MA_TRAN_RUN.csv`.
> `MAU_KET_QUA_GIA_DINH.md` chỉ để hình dung đầu ra; **mọi số trong đó là giả định, cấm chép vào báo cáo**.

## Cách dùng

| Cách | Việc cần làm |
|---|---|
| **Một lần, không gián đoạn (khuyến nghị)** | Dán **Prompt 0** rồi để agent chạy hết 18 bước. |
| **Chia theo phase** | Dán **Prompt 0**, sau đó dán lần lượt **Prompt 1 → 5**. Mỗi prompt vẫn giữ nguyên vòng lặp và không dừng giữa các bước con. |
| **Chỉ một bước** | Dán **Prompt 0**, rồi dán **Prompt bước n** ở mục 8 và thay `<ID>` bằng bước cần làm. |
| **Có sự cố nghiêm trọng** | Dùng **Prompt 7**, dừng đúng luật, không tự sửa số. |

---

## 1. Prompt 0 — Tổng đạo diễn

```text
Bạn là kỹ sư ML chính kiêm auditor tái lập của dự án GTSRB CNN. Hãy thực thi TOÀN BỘ kế hoạch v2 trong docs/kehoach/1/, liên tục từ Phase 0 đến nghiệm thu, không dừng giữa chừng.

ĐỌC TRƯỚC KHI LÀM, theo thứ tự:
1. AGENTS.md.
2. docs/kehoach/1/README.md.
3. docs/kehoach/1/KE_HOACH_V2_DAY_DU.md — đọc hết.
4. docs/kehoach/1/CHECKLIST_TRIEN_KHAI.md.
5. docs/kehoach/1/MA_TRAN_RUN.csv.
6. docs/kehoach/1/MAU_KET_QUA_GIA_DINH.md — chỉ để hình dung; mọi số là giả định, cấm chép.
7. git status; cấu trúc src/, tests/, configs/, outputs/results/, outputs/splits/.

MỤC TIÊU: hoàn thành 18 bước, có artifact thật, test pass, tài liệu cập nhật, commit/push, báo cáo cuối.

LUẬT BẤT BIẾN:
- Không bịa số. Mọi số phải trace được về file, log, metric hoặc hash.
- Không tune trên lockbox/official test. Lockbox chỉ mở một lần sau freeze.
- Không dùng split_random cho bất kỳ kết luận nào. Mọi kết luận dùng split v2/stratified_group.
- Không lật ảnh ngang/dọc; không đổi mapping nhãn; không viết lại preprocessing.
- Không thêm dependency nếu không cập nhật requirements.txt trong cùng commit.
- Không commit data/, outputs/runs/, *.pt, *.npy, .venv, .env.
- Không xóa/sửa công việc có sẵn của người khác khi chưa kiểm tra.
- Không tuyên bố mô hình đủ an toàn cho hệ thống giao thông thực tế.
- Chênh lệch < 0,3 điểm % là tương đương. CI chứa 0 thì không nói "tốt hơn".
- Macro-F1 luôn tính labels=range(43), average="macro", zero_division=0.
- Tối đa 40 epoch, early stopping patience 7 theo val macro-F1. Không so 10 epoch với 40 epoch.
- Mỗi run chỉ đổi một nhóm yếu tố; ghi rõ trong config/notes.

BẮT ĐẦU: tạo todos tổng đủ 18 mục (ID + tên + trạng thái pending) trước khi đọc chi tiết. Sau đó mới vào vòng lặp.

VÒNG LẶP BẮT BUỘC CHO MỖI BƯỚC n:
1. TODOS: tạo/cập nhật danh sách 18 bước bằng công cụ todo. Mỗi mục có ID, trạng thái, evidence. Cập nhật sau từng bước, không gộp nhiều bước.
2. ĐỌC: đọc đúng mục của bước n trong plan, checklist và code liên quan.
3. LÀM: thực hiện bước n; tạo code/config/artifact cần thiết.
4. TEST: chạy test theo checklist. Nếu fail, sửa và chạy lại đến khi pass; ghi log.
5. GHI: cập nhật kết quả thật vào checklist và tài liệu kết quả tương ứng. Chỉ tick khi có evidence.
6. TODO: cập nhật todos ngay, kể cả khi fail/blocked.
7. COMMIT/PUSH: commit đúng file của bước n bằng tiếng Việt, push origin main. Nếu không có gì để commit, ghi lý do và bỏ qua.
8. NEXT: sang bước n+1 ngay. Không hỏi "tiếp tục không", không tóm tắt dài.

DANH SÁCH 18 BƯỚC:
| ID | Bước | Checklist |
|---|---|---|
| 0.1 | Chốt giao thức/pre-registration | 0.1 |
| 0.2 | Code + sinh split v2 | 0.2 |
| 0.3 | Audit C1–C12 | 0.3 |
| 0.4 | Test split/dataloader | 0.4 |
| 1.1 | Baseline B1 | 1.1 |
| 1.2 | Factorial 2^4, F00–F15 × 3 seed | 1.2 |
| 2.1 | Kiến trúc D/C/W × 3 seed | 2.1 |
| 2.2 | Hyperparam L/P/W/CW/LS/I/BS × 3 seed | 2.2 |
| 2.3 | Augmentation G × 3 seed + corruption | 2.3 |
| 2.4 | Chọn top 3–4 + control | 2.4 |
| 3.1 | OOF 3 fold × 3 seed + bootstrap/Holm/variance | 3.1 |
| 3.2 | Chọn winner + freeze | 3.2 |
| 3.3 | Final dev × 3 + lockbox một lần + official test raw/clean | 3.3 |
| 4.1 | Thống kê/tổng hợp summary v2 | 4.1 |
| 4.2 | Hình cơ chế | 4.2 |
| 4.3 | Decision cards 10 yếu tố | 4.3 |
| 4.4 | Báo cáo v2 + phụ lục legacy | 4.4 |
| 5 | Nghiệm thu cuối 12 điều kiện | Nghiệm thu cuối |

CHI TIẾT TỪNG BƯỚC:
- 0.1: chốt 8 so sánh chính, luật quyết định, luật dừng; điền ngày/commit hash vào checklist. Commit tài liệu trước khi chạy run đầu.
- 0.2: bổ sung src/split_v2.py hoặc mở rộng src/data.py theo mục 3.4. Sinh lockbox 20% track seed 2024, dev 80%, screening 80/20 seed 42, 3 fold seed 42. Ghi outputs/splits/v2/manifest.json, hash từng file, mean/std từng split.
- 0.3: kiểm tra C1–C12; track/sign-group overlap; đủ 43 lớp; test không nằm trong train; MD5; lockbox không có ảnh gần trùng train (MAD < 5). Ghi outputs/results/v2/audit/*.csv. Nếu bất kỳ mục nào fail: dừng khẩn cấp, không sang Phase 1.
- 0.4: tests/test_split_v2.py pass; forward (2,3,64,64) → (2,43) cho A/B/C; macro-F1/43 khớp sklearn; đọc manifest/mean/std; num_workers=2 trên Windows không treo.
- 1.1: B0 majority, B1 HOG+LinearSVC chọn C trên screening val, B1b refit trên dev nhưng chưa chạm lockbox. Ghi thời gian trích/fit.
- 1.2: F00–F15 × 3 seed = 48 run. R1–R4 chỉ là alias của F00/F07/F15/D3; không chạy trùng. Mỗi run đủ config, env, metrics, best.pt, curves. Kiểm tra F00/F07/F15 tương ứng CNN-A/E2/E3. Tính main effect BN/DO/WD/Aug, interaction BN×Aug và DO×WD, bảng 2^4 đủ 16 dòng. Cửa kiểm tra: nếu F15 thấp hơn 97,06% quá xa, kiểm tra split/mean/std trước khi kết luận; không đổi luật.
- 2.1: D1, D2, D3, C1, W1, W2, W3 × 3 seed = 21 run; D4 = F15, không chạy lại. Bảng params vs macro-F1 (Pareto); so sánh D1 vs D4, D2 vs D4, C1 vs D3.
- 2.2: L1, L2, L4, L5, P1, P2, W4, CW1, CW2, LS1, I1, I2, BS1, BS2 × 3 seed = 42 run; L3 = F15. Đủ 40 epoch. 3e-3 NaN/không hội tụ là kết quả hợp lệ, phải báo cáo.
- 2.3: G1–G4, G6–G8 × 3 seed = 21 run; G0 = F07, G5 = F15. Corruption 8 loại × 3 mức trên screening val. Bảng clean + corruption; kết luận aug có cần, phần nào.
- 2.4: sắp xếp mean macro-F1; lấy nhóm trong 0,3 điểm so với tốt nhất; chọn top 3 theo tie-break + 1 control. Ghi lý do trước khi chạy OOF; điền X1–X4 vào MA_TRAN_RUN.csv.
- 3.1: top 3 + control × 3 fold × 3 seed = 36 run. Mỗi ảnh dev được dự đoán đúng một lần mỗi seed. oof_predictions.csv đủ dòng, không trùng. Macro-F1/43 trên OOF; cluster bootstrap theo track B ≥ 2.000; paired bootstrap cho Δ; Holm cho 8 so sánh chính; variance decomposition seed/fold/residual.
- 3.2: áp dụng luật 4.5. Nếu tương đương, chọn ít params/nhanh/đơn giản hơn. Ghi decision card; config, seed, epoch = median best epoch. Commit freeze; ghi config_sha256, split_sha256, git_commit. Chưa mở lockbox.
- 3.3: train final trên toàn dev × 3 seed; không early stop trên lockbox; kiểm tra checkpoint load/forward. Mở lockbox đúng một lần, ghi thời gian/lệnh/người chạy. lockbox_result.json có accuracy, macro-F1/43, per-class, confusion, top-3, ECE. Official test raw 12.630 và clean 12.499. Không chỉnh gì sau khi xem lockbox.
- 4.1: bảng summary v2 khớp artifact; CI cho 8 so sánh; variance decomposition; kiểm tra macro-F1/43 với sklearn trên một mẫu; không có số nào không trace được.
- 4.2: learning curves A/B/C; interaction plot BN×Aug, DO×WD; Pareto params vs macro-F1; per-class Δ heatmap; Grad-CAM 12 ảnh; reliability + ECE; corruption curves 8 panel; confusion matrix chuẩn hóa hàng; error taxonomy 12 mẫu.
- 4.3: decision cards cho BN, dropout, WD, aug, depth, GAP, width, LR, class weight, input size. Mỗi card có mức đã thử, mean ± sd, Δ [CI], cơ chế, kết luận, đánh đổi.
- 4.4: viết docs/bao_cao/BAO_CAO_v2.md, chương 2–6; phụ lục legacy leaky; xóa/sửa các câu sai mục 12.4; kiểm tra số khớp summary.csv/lockbox_result.json. Không ghi đè báo cáo cũ khi chưa được yêu cầu.
- 5: chạy 12 điều kiện nghiệm thu cuối; ký xác nhận; repo sạch; tái lập cùng seed; báo cáo cuối.

TEST GATE VÀ EVIDENCE:
- Mỗi bước phải có: lệnh đã chạy, file đầu ra, metric/hash, dòng checklist tương ứng.
- Không tick checklist nếu không có evidence.
- Test fail thì sửa code/test, không sửa số. Run thất bại hợp lệ (NaN, phân kỳ) thì ghi nhận và tiếp tục; không xóa.
- Nếu bước không có automated test, tạo smoke test tối thiểu hoặc ghi lệnh kiểm tra + output thật.

COMMIT/PUSH:
- Trước khi stage: git status --short. Chỉ stage file thuộc bước n; không git add -A mù quáng.
- Message tiếng Việt: feat:, fix:, exp:, docs:, chore: + mô tả ngắn.
- Body ghi: Bước <ID>; test <...>; artifact <...>.
- Push origin main. Nếu bị từ chối: git pull --rebase origin main, xử lý conflict, chạy lại test, push. Không force push.
- Không commit data/, outputs/runs/, *.pt, *.npy, .venv, .env. Được commit manifest/split JSON nhỏ, summary/decision card/corruption CSV và tài liệu.

CHỈ DỪNG KHI:
- Track/sign-group overlap > 0.
- Lockbox bị dùng trước freeze.
- Test xuất hiện trong train.
- Config hash/split hash không khớp.
- Metric không tái lập được với cùng seed.
- Có số trong báo cáo không trace được về artifact.
- Cần quyền đặc biệt, credential, hoặc thao tác phá hủy chưa được cho phép.
Ngoài các trường hợp trên: tự sửa, tự chạy lại, tự tiếp tục. Không hỏi lại giữa chừng.

ĐỊNH DẠNG TIẾT KIỆM TOKEN:
- Mỗi bước chỉ in một dòng: [n/18] <tên bước> — OK/FAIL — <artifact>.
- Log chi tiết ghi ra file, không paste vào chat.
- Tiếng Việt có dấu, không emoji, không giải thích dài.
- PowerShell: chcp 65001, $ErrorActionPreference='Stop', Write-Host tiếng Việt, chỉ in thông báo ngắn.

BÁO CÁO CUỐI:
- 18/18 bước, trạng thái pass/fail.
- Metric thật kèm CI, artifact, test đã chạy, commit đã push.
- Xác nhận: lockbox một lần, không tune test, không số giả định.
- Rủi ro/hạn chế và việc còn lại.
```

---

## 2. Prompt 1 — Phase 0

```text
Tiếp tục Prompt 0. Thực hiện Phase 0 (bước 0.1–0.4) liên tục, không dừng giữa các bước con.

Đọc: KE_HOACH_V2_DAY_DU.md mục 3.1–3.7, 4, 10.1, 11.1; CHECKLIST Phase 0; AGENTS.md mục 5, 8, 14.

Việc:
- 0.1: chốt pre-registration; điền ngày/commit hash.
- 0.2: tạo src/split_v2.py hoặc mở rộng src/data.py; sinh outputs/splits/v2/manifest.json + split JSON + hash + mean/std.
- 0.3: audit C1–C12, MAD < 5, sign-group, test-in-train, MD5; ghi outputs/results/v2/audit/.
- 0.4: tests/test_split_v2.py; forward A/B/C; macro-F1/43; manifest/mean-std; num_workers=2 Windows.

Gate: C1–C12 phải đạt 100%. Nếu fail: dừng khẩn cấp, ghi log, không sang Phase 1.
Sau mỗi bước: cập nhật checklist + docs/kehoach/1/KET_QUA_V2.md + todos; commit/push.
Hết Phase 0: nếu đang chạy chuỗi đầy đủ, sang Phase 1 ngay; nếu chỉ chạy Phase 0, báo hoàn tất Phase 0.
```

---

## 3. Prompt 2 — Phase 1

```text
Tiếp tục Prompt 0. Thực hiện Phase 1 (bước 1.1–1.2) liên tục.

Đọc: KE_HOACH_V2_DAY_DU.md mục 5, 6.0–6.2, 7, 10.2; CHECKLIST Phase 1; MA_TRAN_RUN.csv nhóm baseline và F_factorial.

Việc:
- 1.1: B0 majority; B1 HOG+LinearSVC chọn C trên screening val; B1b refit trên dev, chưa chạm lockbox.
- 1.2: F00–F15 × 3 seed = 48 run; R1–R4 là alias, không chạy trùng; đủ artifact từng run; kiểm tra F00/F07/F15; bảng 2^4; main effect; interaction BN×Aug, DO×WD; 4 curve đại diện.

Gate: F15 không thấp bất thường so với 97,06% cũ; nếu thấp, kiểm tra split/mean/std trước khi kết luận. Không đổi luật.
Sau mỗi bước: cập nhật checklist + KET_QUA_V2 + todos; commit/push.
Hết Phase 1: sang Phase 2 ngay.
```

---

## 4. Prompt 3 — Phase 2

```text
Tiếp tục Prompt 0. Thực hiện Phase 2 (bước 2.1–2.4) liên tục.

Đọc: KE_HOACH_V2_DAY_DU.md mục 6.3–6.5, 8.1–8.6, 10.3; CHECKLIST Phase 2; MA_TRAN_RUN.csv nhóm D, H, G.

Việc:
- 2.1: D1–D3, C1, W1–W3 × 3 seed = 21 run; D4 = F15; bảng Pareto; so sánh GAP/depth/width/dropout.
- 2.2: L1, L2, L4, L5, P1, P2, W4, CW1, CW2, LS1, I1, I2, BS1, BS2 × 3 seed = 42 run; L3 = F15; đủ 40 epoch; ghi NaN/phân kỳ là kết quả hợp lệ.
- 2.3: G1–G4, G6–G8 × 3 seed = 21 run; G0 = F07, G5 = F15; corruption 8 loại × 3 mức trên screening val; bảng clean + corruption.
- 2.4: chọn top 3–4 trong 0,3 điểm; tie-break; thêm control; ghi lý do trước OOF; điền X1–X4.

Gate: đủ bảng D/H/G; top 3–4 có lý do và control; không dùng lockbox.
Sau mỗi bước: cập nhật checklist + KET_QUA_V2 + todos; commit/push.
Hết Phase 2: sang Phase 3 ngay.
```

---

## 5. Prompt 4 — Phase 3

```text
Tiếp tục Prompt 0. Thực hiện Phase 3 (bước 3.1–3.3) liên tục.

Đọc: KE_HOACH_V2_DAY_DU.md mục 4.5–4.7, 6.6, 7; CHECKLIST Phase 3; MA_TRAN_RUN.csv nhóm X.

Việc:
- 3.1: top 3 + control × 3 fold × 3 seed = 36 run; OOF đủ mỗi ảnh dev đúng một lần mỗi seed; macro-F1/43; cluster bootstrap theo track B ≥ 2.000; paired bootstrap; Holm 8 so sánh; variance decomposition.
- 3.2: chọn winner theo luật 4.5; tie-break; decision card; freeze config/seed/median best epoch; ghi config_sha256, split_sha256, git_commit. Chưa mở lockbox.
- 3.3: final trên dev × 3 seed; không early stop trên lockbox; kiểm tra checkpoint; mở lockbox đúng một lần, ghi log; lockbox_result.json; official test raw 12.630 + clean 12.499.

Gate: OOF không trùng/thiếu; hash khớp; lockbox chỉ một lần; không chỉnh gì sau khi xem lockbox.
Nếu overlap > 0, lockbox mở sớm, test nằm trong train, hoặc hash lệch: dừng khẩn cấp.
Sau mỗi bước: cập nhật checklist + KET_QUA_V2 + todos; commit/push.
Hết Phase 3: sang Phase 4 ngay.
```

---

## 6. Prompt 5 — Phase 4

```text
Tiếp tục Prompt 0. Thực hiện Phase 4 (bước 4.1–4.4) liên tục.

Đọc: KE_HOACH_V2_DAY_DU.md mục 7, 8, 11, 12; CHECKLIST Phase 4; MAU_KET_QUA_GIA_DINH.md chỉ để hình dung, không chép số.

Việc:
- 4.1: summary v2 khớp artifact; CI 8 so sánh; variance decomposition; kiểm tra sklearn; traceability.
- 4.2: 9 nhóm hình theo checklist 4.2.
- 4.3: 10 decision cards theo schema 11.5.
- 4.4: viết docs/bao_cao/BAO_CAO_v2.md chương 2–6; phụ lục legacy leaky; sửa câu sai mục 12.4; kiểm tra số khớp summary.csv/lockbox_result.json.

Gate: không có số không trace được; không dùng số random cũ để kết luận; không ghi đè báo cáo cũ.
Sau mỗi bước: cập nhật checklist + KET_QUA_V2 + todos; commit/push.
Hết Phase 4: chạy Prompt 6 nghiệm thu cuối.
```

---

## 7. Prompt 6 — Nghiệm thu cuối

```text
Chạy nghiệm thu cuối theo CHECKLIST_TRIEN_KHAI.md mục "Nghiệm thu cuối" và KE_HOACH_V2_DAY_DU.md mục 14.

Kiểm tra đủ 12 điều kiện:
1. Lockbox 0 track/sign-group overlap; one-shot.
2. Mọi run kết luận dùng stratified_group.
3. Mỗi kết luận có Δ + CI + luật quyết định.
4. Factorial 2^4 đủ 16 config.
5. Kiến trúc tách GAP/depth/width/dropout.
6. Aug bật/tắt + corruption.
7. OOF 3 fold × 3 seed.
8. Decision card đủ 10 yếu tố.
9. Báo cáo không dùng số random cũ để kết luận.
10. Không tune trên lockbox/test.
11. Repo sạch; không commit data/checkpoint.
12. Tái lập: cùng seed → cùng kết quả (sai khác ghi rõ).

Nếu mục nào fail: quay lại bước tương ứng, sửa và chạy lại test; không sửa số.
Cập nhật checklist, ký xác nhận, commit/push.
Báo cáo cuối: 18/18, metric thật + CI, artifact, test, commit, rủi ro, xác nhận không dùng số giả định.
```

---

## 8. Prompt cho một bước n

```text
Tiếp tục Prompt 0. Thực hiện bước <ID> trong CHECKLIST_TRIEN_KHAI.md theo đúng vòng lặp 1–8.

Đọc: <mục plan>; <mục checklist>; code liên quan.
Việc: <mô tả ngắn>.
Gate: <tiêu chí pass>.
Cập nhật: <file kết quả>.
Commit: <loại>: <mô tả tiếng Việt>.
Sau khi xong: cập nhật todos, commit/push, tự động sang bước kế tiếp. Không dừng.
```

---

## 9. Prompt 7 — Sự cố nghiêm trọng

```text
DỪNG NGAY. Không sửa số, không xóa artifact, không chạy tiếp.

Lưu bằng chứng: lệnh, log, config, hash, thời gian, bước đang làm.
Tạo docs/kehoach/1/SU_CO_V2.md gồm:
- Thời gian, bước, người/lệnh chạy.
- Hiện tượng và phạm vi ảnh hưởng.
- File/hash liên quan.
- Đã thử gì.
- Đề xuất xử lý (không tự ý đổi luật).

Chỉ tiếp tục sau khi người phụ trách xử lý xong. Nếu phải chạy lại do bug, tạo bản _v2 và ghi lý do; không ghi đè kết quả cũ.
```

---

## 10. Phụ lục PowerShell tối giản

```powershell
chcp 65001>$null
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new()

$R='outputs/runs/v2'
Write-Host "Bước 1.2: factorial 2^4"
python scripts/run_matrix.py --csv docs/kehoach/1/MA_TRAN_RUN.csv --only F --resume
if($LASTEXITCODE -ne 0){throw 'Factorial lỗi'}
Write-Host "Xong: $R"
```

Quy ước:
- File `.ps1` lưu UTF-8 có BOM nếu dùng Windows PowerShell 5.1.
- Thông báo tiếng Việt có dấu, ngắn, không emoji.
- Không in bảng dài; ghi CSV/log rồi in đường dẫn.
- Dùng `if($LASTEXITCODE -ne 0){throw '...'}` sau mỗi lệnh Python quan trọng.
- Không hard-code `C:\`; dùng đường dẫn tương đối từ root repo.

---

## 11. Mẫu commit

```text
feat: thêm chia split v2 theo track
fix: sửa overlap screening val
exp: chạy factorial 2^4 trên split v2
docs: cập nhật kết quả factorial và checklist
chore: khóa cấu hình winner v2
```

Body gợi ý:

```text
Bước 1.2
Test: tests/test_train_smoke.py pass
Artifact: outputs/results/v2/factorial.csv
```

---

## 12. Mẫu evidence trong checklist

```text
- [x] F00–F15 × 3 seed (48 run)
  Evidence: outputs/results/v2/factorial.csv | lệnh: python scripts/run_matrix.py --only F | 48/48 run hợp lệ
```

Không tick nếu không có đường dẫn file, lệnh hoặc hash.
