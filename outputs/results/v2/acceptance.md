| # | Điều kiện | Đạt | Giá trị | Bằng chứng |
|---|---|---|---|---|
| 1 | Lockbox 0 track/sign-group overlap; one-shot | ✓ | {"overlap_track": 0, "overlap_sign_group": 0, "n_lockbox": 8302} | audit_report.json C1/C2 + lockbox_access.log |
| 2 | Mọi run kết luận dùng split sign-group (v2_clean) | ✓ | {"protocols": ["v2_clean"], "split_prefixes": ["dev", "fold", "screening"]} | summary.csv |
| 3 | Mỗi kết luận có Δ + CI + luật quyết định | ✓ | {"n_main_comparisons": 8, "holm_done": true} | main_comparisons.csv, decision_cards.csv |
| 4 | Factorial 2⁴ đủ 16 config | ✓ | {"n_runs": 48, "all_complete": true} | phase12_artifacts.json |
| 5 | Kiến trúc tách GAP/depth/width; D3 chỉ legacy | ✓ | {"configs": ["d1", "d2", "d3", "w1", "w2"], "d3_role": "legacy"} | tables/architecture.md |
| 6 | Aug bật/tắt + corruption 6×3 | ✓ | {"n_rows": 108, "n_types": 6, "n_severities": 3} | corruption.csv, tables/augmentation.md |
| 7 | OOF 3 fold × 3 seed | ✓ | {"rows": 278091, "expected": 278091} | oof_predictions.csv, variance.json |
| 8 | Decision card đủ 10 yếu tố | ✓ | {"n_cards": 10, "factors": ["BN", "Dropout", "Weight decay", "Augmentation", "GAP", "Depth", "Width", "LR", "Class weight", "Input size"]} | decision_cards.csv |
| 9 | Báo cáo không dùng số random cũ để kết luận | ✓ | {"legacy_rows": 9, "random_in_summary": 0} | legacy/README.md, BAO_CAO_v2.md Phụ lục A |
| 10 | Không tune trên lockbox/test | ✓ | {"freeze_before_open": true, "epoch": 26, "frozen_at": "2026-11-03 03:57:32"} | FREEZE.json, lockbox_access.log |
| 11 | Repo sạch; không commit data/checkpoint | ✓ | {"forbidden": [], "tracked_pt": ["checkpoints/final_s42/best.pt"]} | git ls-files |
| 12 | Tái lập: cùng seed → cùng kết quả | ✓ | {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0, "val_macro_f1": 0.0} | reproducibility.json, log_repro.txt |
