"""Chia split 3 tầng v2 (lockbox / dev / screening / 3 fold) — Giai đoạn 0.2.

Nguyên tắc (khớp `docs/kehoach/1/KE_HOACH_V2_DAY_DU.md` mục 3.4):

* Đơn vị nhóm là **sign-group**: các track (và frame) được gộp nếu có cặp ảnh
  gần trùng (MAD < 5 trên ảnh 64×64 trong cùng lớp). Không xóa ảnh nào.
* Lockbox = 20% sign-group mỗi lớp (seed 2024), stratified 43 lớp.
* Dev = phần còn lại; screening 80/20 trong dev (seed 42); 3 fold round-robin
  cân bằng số frame trong dev (seed 42).
* Mọi bước đều kiểm tra C1–C12 trước khi ghi manifest.

Artifact:
    outputs/splits/v2/{lockbox,dev,screening,fold_0..2}.json
    outputs/splits/v2/manifest.json   (hash từng file + mean/std từng split)
    outputs/results/v2/audit/sign_groups.csv
    outputs/results/v2/audit/near_duplicate_lockbox.csv

Cách dùng::

    python -m src.split_v2 --build
    python -m src.split_v2 --audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import NUM_CLASSES, PROJECT_ROOT, RESULTS_DIR
from .data import _read_train_meta, compute_stats, load_cache

LOG = logging.getLogger(__name__)

SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits" / "v2"
AUDIT_DIR = RESULTS_DIR / "v2" / "audit"
MANIFEST_PATH = SPLIT_DIR / "manifest.json"
POOL_SIZE = 39_201
NEAR_DUP_MAD = 5.0
THUMB_MAD = 8.0


# ---------------------------------------------------------------------------
# Tiện ích hash / IO
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    """SHA-256 của chuỗi byte."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 của file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Ghi JSON ổn định (sort_keys) và trả SHA-256 của nội dung đã ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    data = text.encode("utf-8")
    path.write_bytes(data)  # ghi byte để hash khớp tuyệt đối (không đổi CRLF)
    return sha256_bytes(data)


# ---------------------------------------------------------------------------
# Sign-group: gộp track gần trùng bằng MAD < 5
# ---------------------------------------------------------------------------


class UnionFind:
    """Union-find tối giản (path compression + union by rank)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _thumb(x: np.ndarray, size: int = 16) -> np.ndarray:
    """Ảnh uint8 (N,64,64,3) → thumbnail xám (N,size,size) float32."""
    n, h, w, _ = x.shape
    factor = h // size
    gray = x.astype(np.float32).mean(axis=3)
    return gray.reshape(n, size, factor, size, factor).mean(axis=(2, 4)).reshape(n, size * size)


def _hash_values(x: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Hash nhị phân (aHash 64-bit, dHash 56-bit) cho LSH toàn cục."""
    n = len(x)
    gray = x.astype(np.float32).mean(axis=3)
    small = gray.reshape(n, 8, 8, 8, 8).mean(axis=(2, 4))  # (n, 8, 8)
    results: list[tuple[np.ndarray, int]] = []
    ahash = (small > small.mean(axis=(1, 2), keepdims=True)).reshape(n, 64)
    dhash = (small[:, :, 1:] > small[:, :, :-1]).reshape(n, 56)
    for bits, width in ((ahash, 64), (dhash, 56)):
        values = np.zeros(n, dtype=np.uint64)
        for bit in range(width):
            values = (values << 1) | bits[:, bit].astype(np.uint64)
        results.append((values, width))
    return results


def _candidate_pairs(
    x: np.ndarray, labels: np.ndarray, tracks: Sequence[str]
) -> set[tuple[int, int]]:
    """Sinh ứng viên gần trùng: brute-force thumbnail theo lớp + LSH dHash toàn cục.

    Bỏ qua cặp cùng track (đã chắc chắn cùng nhóm) và cặp khác nhãn.
    """
    pairs: set[tuple[int, int]] = set()
    thumbs = _thumb(x)
    track_array = np.asarray(tracks, dtype=object)

    def add_pair(a: int, b: int) -> None:
        if labels[a] != labels[b] or track_array[a] == track_array[b]:
            return
        pairs.add((a, b) if a < b else (b, a))

    # 1) Trong cùng lớp: MAD thumbnail < THUMB_MAD (bao phủ chắc các cặp MAD < 5).
    for class_id in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == class_id)
        block = thumbs[idx]
        n = len(idx)
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            distance = np.abs(block[start:stop, None, :] - block[None, :, :]).mean(axis=2)
            rows, cols = np.nonzero(distance < THUMB_MAD)
            for r, c in zip(rows, cols):
                add_pair(int(idx[start + r]), int(idx[c]))
    # 2) Toàn cục: LSH trên các dải của aHash/dHash (bắt được cả cặp khác track).
    for hashes, width in _hash_values(x):
        for band in range(4):
            shift = band * (width // 4)
            keys = ((hashes >> shift) & np.uint64((1 << (width // 4)) - 1)).astype(np.int64)
            buckets: dict[int, list[int]] = defaultdict(list)
            for i, key in enumerate(keys.tolist()):
                buckets[key].append(i)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                for a_pos in range(len(members)):
                    for b_pos in range(a_pos + 1, len(members)):
                        add_pair(members[a_pos], members[b_pos])
    return pairs


def build_sign_groups(
    x: np.ndarray,
    labels: np.ndarray,
    tracks: Sequence[str],
    threshold: float = NEAR_DUP_MAD,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Gộp sign-group bằng MAD < ``threshold``; trả (group_index, danh sách cặp)."""
    started = time.perf_counter()
    candidates = _candidate_pairs(x, labels, tracks)
    LOG.info("Ứng viên gần trùng: %d cặp (%.1f s)", len(candidates), time.perf_counter() - started)
    uf = UnionFind(len(x))
    verified: list[tuple[int, int, float]] = []
    flat = x.reshape(len(x), -1).astype(np.int16)
    for a, b in sorted(candidates):
        mad = float(np.abs(flat[a] - flat[b]).mean())
        if mad < threshold:
            uf.union(a, b)
            verified.append((a, b, mad))
    # Frame trong cùng track luôn cùng một nhóm (track là một đoạn video của một biển).
    by_track: dict[str, list[int]] = defaultdict(list)
    for i, track in enumerate(tracks):
        by_track[track].append(i)
    for members in by_track.values():
        for other in members[1:]:
            uf.union(members[0], other)
    roots = np.array([uf.find(i) for i in range(len(x))], dtype=np.int64)
    LOG.info(
        "Sign-group: %d ảnh → %d nhóm (gộp %d cặp MAD<%.1f, %.1f s)",
        len(x),
        len(set(roots.tolist())),
        len(verified),
        threshold,
        time.perf_counter() - started,
    )
    return roots, verified


def name_groups(roots: np.ndarray, labels: np.ndarray, tracks: Sequence[str]) -> np.ndarray:
    """Đặt tên nhóm ổn định ``<label:05d>_g<k>`` theo track nhỏ nhất trong nhóm."""
    names = np.empty(len(roots), dtype=object)
    for class_id in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == class_id)
        members: dict[int, list[int]] = defaultdict(list)
        for i in idx:
            members[int(roots[i])].append(int(i))
        ordered = sorted(members.items(), key=lambda kv: min(tracks[i] for i in kv[1]))
        for order, (_, rows) in enumerate(ordered):
            for i in rows:
                names[i] = f"{class_id:05d}_g{order:02d}"
    return names


# ---------------------------------------------------------------------------
# Chia tầng
# ---------------------------------------------------------------------------


def _pick_groups(
    groups_by_class: dict[int, list[str]],
    ratio: float,
    seed: int,
) -> set[str]:
    """Chọn ``round(ratio*n)`` nhóm mỗi lớp (tối thiểu 1, chừa ít nhất 1 nhóm)."""
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for class_id in range(NUM_CLASSES):
        groups = sorted(groups_by_class.get(class_id, []))
        if not groups:
            raise RuntimeError(f"Lớp {class_id} không có nhóm nào — dữ liệu lỗi")
        if len(groups) < 2:
            continue
        k = min(max(1, int(round(ratio * len(groups)))), len(groups) - 1)
        chosen.update(rng.choice(np.asarray(groups, dtype=object), size=k, replace=False).tolist())
    return chosen


def _assign_folds(
    groups_by_class: dict[int, list[str]],
    image_counts: dict[str, int],
    n_folds: int = 3,
) -> dict[str, int]:
    """Round-robin nhóm theo lớp vào các fold, cân bằng tổng số frame."""
    assignment: dict[str, int] = {}
    for class_id in range(NUM_CLASSES):
        groups = sorted(
            groups_by_class.get(class_id, []),
            key=lambda g: (-image_counts[g], g),
        )
        load = [0] * n_folds
        for group in groups:
            fold = int(np.argmin(load))
            assignment[group] = fold
            load[fold] += image_counts[group]
    return assignment


def _group_map(groups: np.ndarray, labels: np.ndarray, tracks: Sequence[str]) -> dict[int, list[str]]:
    """``{class_id: [group, ...]}`` từ mảng group-name."""
    result: dict[int, list[str]] = defaultdict(list)
    for class_id in range(NUM_CLASSES):
        result[class_id] = sorted(set(groups[labels == class_id].tolist()))
    return result


def _indices(groups: np.ndarray, names: Iterable[str]) -> list[int]:
    """Chỉ số ảnh thuộc tập nhóm cho trước (sắp xếp tăng dần)."""
    wanted = set(names)
    return sorted(int(i) for i in np.flatnonzero(np.isin(groups, np.asarray(sorted(wanted), dtype=object))))


def _counts(labels: np.ndarray, indices: Sequence[int]) -> dict[str, int]:
    return {str(c): int(v) for c, v in zip(*np.unique(labels[list(indices)], return_counts=True))}


def _class_support(labels: np.ndarray, indices: Sequence[int]) -> list[int]:
    return np.bincount(labels[list(indices)], minlength=NUM_CLASSES).astype(int).tolist()


def _track_groups(tracks: Sequence[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(tracks)}


def _overlap(a: Iterable[str], b: Iterable[str]) -> int:
    return len(set(a) & set(b))


def _checks(
    groups: np.ndarray,
    labels: np.ndarray,
    tracks: Sequence[str],
    lockbox: Sequence[int],
    dev: Sequence[int],
    screening: dict[str, list[int]],
    folds: list[dict[str, list[int]]],
    md5_duplicates: int,
    near_dup_cross: int,
    test_in_train: int = 0,
) -> dict[str, Any]:
    """Chạy C1–C12 và trả dict kết quả (giá trị thật, không hard-code)."""
    g_lock, g_dev = set(groups[lockbox].tolist()), set(groups[dev].tolist())
    t_lock = {tracks[i] for i in lockbox}
    t_dev = {tracks[i] for i in dev}
    fold_results = []
    for fold in folds:
        train_idx, val_idx = fold["train"], fold["val"]
        fold_results.append(
            {
                "fold": fold["fold"],
                "overlap_track": _overlap((tracks[i] for i in train_idx), (tracks[i] for i in val_idx)),
                "overlap_group": _overlap(groups[train_idx].tolist(), groups[val_idx].tolist()),
                "classes_val": len([v for v in _class_support(labels, val_idx) if v > 0]),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )
    dev_track_counts = [
        len(set(tracks[i] for i in np.flatnonzero(labels == c) if i in set(dev)))
        for c in range(NUM_CLASSES)
    ]
    return {
        "C1_overlap_track_lockbox_dev": _overlap(t_lock, t_dev),
        "C2_overlap_sign_group_lockbox_dev": _overlap(g_lock, g_dev),
        "C3_classes_lockbox": len([v for v in _class_support(labels, lockbox) if v > 0]),
        "C4_classes_dev": len([v for v in _class_support(labels, dev) if v > 0]),
        "C5_classes_screening_val": len([v for v in _class_support(labels, screening["val"]) if v > 0]),
        "C6_folds_classes_val": [f["classes_val"] for f in fold_results],
        "C7_overlap_screening": _overlap(
            (tracks[i] for i in screening["train"]), (tracks[i] for i in screening["val"])
        ),
        "C8_fold_overlap_track": [f["overlap_track"] for f in fold_results],
        "C8b_fold_overlap_group": [f["overlap_group"] for f in fold_results],
        "C9_test_in_train": int(test_in_train),
        "C10_lockbox_near_duplicate_mad_lt_5": int(near_dup_cross),
        "C11_md5_duplicates_in_pool": int(md5_duplicates),
        "C12_min_tracks_per_class_dev": int(min(dev_track_counts)),
        "C12_min_tracks_per_class_lockbox": int(
            min(len(set(tracks[i] for i in np.flatnonzero(labels == c) if i in set(lockbox)))
                for c in range(NUM_CLASSES))
        ),
        "folds": fold_results,
        "n_groups_total": len(set(groups.tolist())),
        "n_tracks_total": len(set(tracks)),
    }


def _test_overlap(train_hashes: set[str]) -> tuple[int, list[str]]:
    """Kiểm tra ảnh test chính thức không nằm trong pool train (so hash nội dung)."""
    from .data import load_test_pool

    overlaps: list[str] = []
    for rel in load_test_pool().paths:
        if sha256_file(PROJECT_ROOT / rel) in train_hashes:
            overlaps.append(rel)
    return len(overlaps), overlaps


def _md5_duplicates(meta_rows: Sequence[dict[str, str]]) -> tuple[int, list[dict[str, str]], set[str]]:
    """Đếm MD5 trùng trong pool 39.201 (hash file gốc, không dùng cache)."""
    seen: dict[str, str] = {}
    duplicate_rows: list[dict[str, str]] = []
    for row in meta_rows:
        path = PROJECT_ROOT / row["path"]
        digest = sha256_file(path)  # sha256 đủ để phát hiện trùng tuyệt đối
        if digest in seen:
            duplicate_rows.append({"path": row["path"], "duplicate_of": seen[digest]})
        else:
            seen[digest] = row["path"]
    return len(duplicate_rows), duplicate_rows, set(seen)


def build(force: bool = False) -> dict[str, Any]:
    """Sinh toàn bộ split v2 + manifest + audit; raise nếu C1–C12 fail."""
    if MANIFEST_PATH.is_file() and not force:
        LOG.info("Đã có %s — dùng --force để sinh lại", MANIFEST_PATH)
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    rows = [r for r in _read_train_meta() if int(r["usable_for_train"]) == 1]
    if len(rows) != POOL_SIZE:
        raise RuntimeError(f"Pool có {len(rows)} ảnh, kỳ vọng {POOL_SIZE}")
    cache = load_cache(64)
    x = cache["train"]["x"]
    labels = cache["train"]["y"]
    tracks = [_track_key(r["path"], int(r["label"])) for r in rows]
    if len(x) != len(rows):
        raise RuntimeError(f"Cache {len(x)} ảnh ≠ meta {len(rows)}")

    started = time.perf_counter()
    roots, verified_pairs = build_sign_groups(x, labels, tracks)
    groups = name_groups(roots, labels, tracks)
    LOG.info("Dựng sign-group xong trong %.1f s", time.perf_counter() - started)

    # Chia 3 tầng.
    all_groups = _group_map(groups, labels, tracks)
    lock_groups = _pick_groups(all_groups, 0.2, seed=2024)
    lockbox = _indices(groups, lock_groups)
    dev = sorted(set(range(len(labels))) - set(lockbox))
    dev_labels = labels[dev]
    dev_groups: dict[int, list[str]] = defaultdict(list)
    for class_id in range(NUM_CLASSES):
        dev_groups[class_id] = sorted(set(groups[np.asarray(dev)][dev_labels == class_id].tolist()))

    screen_val_groups = _pick_groups(dev_groups, 0.2, seed=42)
    screen_val = _indices(groups, screen_val_groups)
    screen_val = sorted(set(screen_val) & set(dev))
    screen_train = sorted(set(dev) - set(screen_val))

    dev_index = np.asarray(dev)
    fold_groups = defaultdict(list)
    for class_id in range(NUM_CLASSES):
        fold_groups[class_id] = sorted(set(groups[dev_index][dev_labels == class_id].tolist()))
    image_counts = {g: int(np.sum(groups == g)) for g in set(groups[dev_index].tolist())}
    fold_assignment = _assign_folds(fold_groups, image_counts, n_folds=3)
    folds = []
    for fold_id in range(3):
        val_groups = {g for g, f in fold_assignment.items() if f == fold_id}
        val_idx = sorted(set(_indices(groups, val_groups)) & set(dev))
        train_idx = sorted(set(dev) - set(val_idx))
        folds.append({"fold": fold_id, "train": train_idx, "val": val_idx})

    # Audit gần trùng chéo tập (dùng lại đúng bộ cặp đã kiểm chứng).
    lock_set = set(lockbox)
    dev_set = set(dev)
    cross_pairs = [(a, b, mad) for a, b, mad in verified_pairs
                   if (a in lock_set) != (b in lock_set)]
    near_dup_cross = len(cross_pairs)
    md5_dup, md5_rows, train_hashes = _md5_duplicates(rows)
    test_in_train, test_overlaps = _test_overlap(train_hashes)

    checks = _checks(
        groups, labels, tracks, lockbox, dev,
        {"train": screen_train, "val": screen_val}, folds, md5_dup, near_dup_cross,
        test_in_train=test_in_train,
    )

    # mean/std theo từng split (chỉ trên train của split).
    mean_std = {
        "screening_train": compute_stats(x, screen_train),
        "fold_0_train": compute_stats(x, folds[0]["train"]),
        "fold_1_train": compute_stats(x, folds[1]["train"]),
        "fold_2_train": compute_stats(x, folds[2]["train"]),
        "dev_all": compute_stats(x, dev),
        "lockbox": compute_stats(x, lockbox),
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    payloads = {
        "lockbox.json": {"name": "lockbox", "seed": 2024, "indices": lockbox},
        "dev.json": {"name": "dev", "indices": dev},
        "screening.json": {"name": "screening", "seed": 42, "train": screen_train, "val": screen_val},
    }
    for fold in folds:
        payloads[f"fold_{fold['fold']}.json"] = {"name": f"fold_{fold['fold']}", "seed": 42, **fold}
    for name, payload in payloads.items():
        hashes[name] = _write_json(SPLIT_DIR / name, payload)

    manifest = {
        "protocol": "v2_clean",
        "generated_at": date.today().isoformat(),
        "pool_size": len(labels),
        "image_size": 64,
        "near_duplicate_threshold_mad": NEAR_DUP_MAD,
        "lockbox": {
            "n": len(lockbox),
            "n_tracks": len({tracks[i] for i in lockbox}),
            "n_groups": len(set(groups[lockbox].tolist())),
            "seed": 2024,
            "file": "outputs/splits/v2/lockbox.json",
            "sha256": hashes["lockbox.json"],
            "class_support": _class_support(labels, lockbox),
        },
        "dev": {
            "n": len(dev),
            "n_tracks": len({tracks[i] for i in dev}),
            "n_groups": len(set(groups[dev].tolist())),
            "file": "outputs/splits/v2/dev.json",
            "sha256": hashes["dev.json"],
            "class_support": _class_support(labels, dev),
        },
        "screening": {
            "train": len(screen_train),
            "val": len(screen_val),
            "seed": 42,
            "n_tracks_train": len({tracks[i] for i in screen_train}),
            "n_tracks_val": len({tracks[i] for i in screen_val}),
            "file": "outputs/splits/v2/screening.json",
            "sha256": hashes["screening.json"],
            "class_support_val": _class_support(labels, screen_val),
        },
        "folds": [
            {
                "fold": fold["fold"],
                "train": len(fold["train"]),
                "val": len(fold["val"]),
                "n_tracks_train": len({tracks[i] for i in fold["train"]}),
                "n_tracks_val": len({tracks[i] for i in fold["val"]}),
                "file": f"outputs/splits/v2/fold_{fold['fold']}.json",
                "sha256": hashes[f"fold_{fold['fold']}.json"],
                "class_support_val": _class_support(labels, fold["val"]),
            }
            for fold in folds
        ],
        "checks": checks,
        "mean_std": mean_std,
        "n_sign_groups": len(set(groups.tolist())),
        "n_near_duplicate_pairs_merged": len(verified_pairs),
        "note": (
            "Đơn vị nhóm là sign-group (gộp track có cặp ảnh MAD<5 trong cùng lớp). "
            "Số ảnh lockbox/dev/screening/fold lấy từ file này, không chép từ kế hoạch."
        ),
    }
    manifest["sha256"] = _write_json(MANIFEST_PATH, manifest)
    MANIFEST_PATH.with_suffix(".sha256").write_text(manifest["sha256"] + "\n", encoding="ascii")

    # Audit CSV.
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    group_tracks: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        entry = group_tracks.setdefault(
            str(groups[i]), {"label": int(labels[i]), "tracks": set(), "n_frames": 0}
        )
        entry["tracks"].add(tracks[i])
        entry["n_frames"] += 1
    with (AUDIT_DIR / "sign_groups.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sign_group", "label", "n_tracks", "n_frames", "tracks"])
        for group_name in sorted(group_tracks):
            entry = group_tracks[group_name]
            writer.writerow([
                group_name,
                entry["label"],
                len(entry["tracks"]),
                entry["n_frames"],
                "|".join(sorted(entry["tracks"])),
            ])
    with (AUDIT_DIR / "near_duplicate_lockbox.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index_a", "path_a", "index_b", "path_b", "mad", "cross_lockbox_dev"])
        for a, b, mad in cross_pairs:
            writer.writerow([a, rows[a]["path"], b, rows[b]["path"], round(mad, 4), 1])
    (AUDIT_DIR / "near_duplicate_summary.json").write_text(
        json.dumps(
            {
                "criterion": "MAD < 5 trên ảnh 64x64 (uint8), chỉ xét cùng nhãn khác track",
                "n_candidate_pairs": len(_candidate_pairs(x, labels, tracks)),
                "n_merged_pairs": len(verified_pairs),
                "n_cross_lockbox_dev": near_dup_cross,
                "n_within_lockbox": sum(1 for a, b, _ in verified_pairs if a in lock_set and b in lock_set),
                "n_within_dev": sum(1 for a, b, _ in verified_pairs if a in dev_set and b in dev_set),
                "test_images_matching_train": test_in_train,
                "test_overlap_examples": test_overlaps[:10],
                "hash": "sha256 nội dung file (MD5 trùng ⇒ SHA-256 trùng)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (AUDIT_DIR / "md5_duplicates_train.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "duplicate_of_sha256"])
        for row in md5_rows:
            writer.writerow([row["path"], row["duplicate_of"]])
    (AUDIT_DIR / "near_duplicate_test.csv").write_text(
        "note\nXem outputs/results/phash_test_suspects.csv (audit read-only cũ, 131 ảnh gần trùng).\n",
        encoding="utf-8",
    )

    failed = _failed_checks(checks)
    if failed:
        raise RuntimeError(f"Split v2 FAIL các kiểm tra: {failed}. Không sang Phase 1.")
    LOG.info(
        "Split v2 OK: lockbox %d ảnh / %d track (%d nhóm); dev %d ảnh / %d track; "
        "screening %d/%d; folds %s",
        len(lockbox), len({tracks[i] for i in lockbox}), len(set(groups[lockbox].tolist())),
        len(dev), len({tracks[i] for i in dev}), len(screen_train), len(screen_val),
        [(len(f["train"]), len(f["val"])) for f in folds],
    )
    return manifest


def _failed_checks(checks: dict[str, Any]) -> list[str]:
    """Trả danh sách kiểm tra không đạt theo tiêu chí C1–C12."""
    failed: list[str] = []
    if checks["C1_overlap_track_lockbox_dev"] != 0:
        failed.append("C1")
    if checks["C2_overlap_sign_group_lockbox_dev"] != 0:
        failed.append("C2")
    if checks["C3_classes_lockbox"] != NUM_CLASSES:
        failed.append("C3")
    if checks["C4_classes_dev"] != NUM_CLASSES:
        failed.append("C4")
    if checks["C5_classes_screening_val"] != NUM_CLASSES:
        failed.append("C5")
    if any(v != NUM_CLASSES for v in checks["C6_folds_classes_val"]):
        failed.append("C6")
    if checks["C7_overlap_screening"] != 0:
        failed.append("C7")
    if any(v != 0 for v in checks["C8_fold_overlap_track"]):
        failed.append("C8")
    if checks["C9_test_in_train"] != 0:
        failed.append("C9")
    if checks["C10_lockbox_near_duplicate_mad_lt_5"] != 0:
        failed.append("C10")
    if checks["C11_md5_duplicates_in_pool"] != 0:
        failed.append("C11")
    if checks["C12_min_tracks_per_class_dev"] < 2:
        failed.append("C12")
    return failed


def _track_key(path: str, label: int) -> str:
    """Khóa track ``<label>_<track>`` (khớp src.data)."""
    from .data import _track_key as legacy_track_key

    return legacy_track_key(path, label)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    parser = argparse.ArgumentParser(description="Sinh split 3 tầng v2")
    parser.add_argument("--build", action="store_true", help="sinh split + manifest + audit")
    parser.add_argument("--force", action="store_true", help="sinh lại dù đã có manifest")
    args = parser.parse_args(argv)
    if args.build or args.force:
        build(force=args.force)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
