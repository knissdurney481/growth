from pathlib import Path

import pytest

from growth_twin.graticule.sheets import load_sheet, merge_sheet

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiment"


def test_main_sheet_values() -> None:
    cfg = load_sheet(EXPERIMENTS / "main.yaml")
    assert cfg.train.lr == 3.0e-4
    assert cfg.train.batch_size == 32
    assert cfg.train.epochs == 100
    assert cfg.physics.lambda_pb == 0.1
    assert cfg.model.hpo_terms == 847
    assert cfg.model.d_model == 768


def test_ablation_inherits_defaults() -> None:
    cfg = load_sheet(EXPERIMENTS / "ablation_physics_no_pb.yaml")
    assert cfg.ablate_pb is True
    assert cfg.model.d_model == 768
    assert cfg.train.lr == 3.0e-4


def test_merge_override() -> None:
    cfg = load_sheet(EXPERIMENTS / "main.yaml")
    merged = merge_sheet(cfg, {"train": {"lr": 1.0e-4}})
    assert merged.train.lr == 1.0e-4
    assert merged.model.d_model == cfg.model.d_model


def test_unknown_key_raises() -> None:
    cfg = load_sheet(EXPERIMENTS / "main.yaml")
    with pytest.raises(KeyError):
        merge_sheet(cfg, {"not_a_field": 1})


def test_every_experiment_loads() -> None:
    for sheet in EXPERIMENTS.glob("*.yaml"):
        assert load_sheet(sheet).name
