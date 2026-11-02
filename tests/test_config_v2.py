"""Test cấu hình v2: alias R1–R4/E1–E4 và các bất biến giao thức."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "v2"


def load(config_id: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{config_id}.yaml").read_text(encoding="utf-8"))


def test_f00_is_cnn_a():
    cfg = load("f00")
    assert cfg["model"]["use_bn"] is False
    assert cfg["model"]["dropout"] == 0
    assert cfg["train"]["weight_decay"] == 0
    assert cfg["augmentation"]["enabled"] is False


def test_f07_is_cnn_b_without_aug():
    cfg = load("f07")
    assert cfg["model"]["use_bn"] is True
    assert cfg["model"]["dropout"] == 0.3
    assert cfg["train"]["weight_decay"] == 0.0001
    assert cfg["augmentation"]["enabled"] is False


def test_f15_is_cnn_b_with_aug():
    cfg = load("f15")
    assert cfg["model"]["use_bn"] is True
    assert cfg["augmentation"]["enabled"] is True
    assert cfg["augmentation"]["profile"] == "all"


def test_d3_is_cnn_c_legacy():
    cfg = load("d3")
    assert len(cfg["model"]["widths"]) == 4
    assert cfg["model"]["gap"] is True
    assert cfg["model"]["dropout"] == 0.4


def test_protocol_invariants():
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["train"]["max_epochs"] == 40, path.name
        assert cfg["train"]["patience"] == 7, path.name
        assert cfg["train"]["optimizer"] == "adamw", path.name
        assert cfg["data"]["num_classes"] == 43, path.name
        assert cfg["experiment"]["protocol"] == "v2_clean", path.name
        assert cfg["augmentation"]["profile"] != "horizontal_flip", path.name


def test_no_alias_config_files_are_duplicated():
    for alias in ("d4", "l3", "g0", "g5"):
        assert not (CONFIG_DIR / f"{alias}.yaml").exists(), f"{alias} phải là alias, không chạy lại"
