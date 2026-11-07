import torch

from growth_twin.datum.benchmarks import PB_REFERENCE, gh_igf1_rates, hill, preece_baines_height
from growth_twin.datum.leveling import preece_baines_residual
from growth_twin.graticule.schema import PhysicsCfg


def test_preece_baines_residual_zero_on_curve() -> None:
    ages = torch.linspace(2.0, 18.0, 8)
    male = preece_baines_height(ages, PB_REFERENCE["male"]).view(1, -1)
    sex = torch.ones(1)
    residual = preece_baines_residual(male, ages, sex)
    assert float(residual) < 1e-6


def test_preece_baines_reaches_adult_height() -> None:
    tall = preece_baines_height(torch.tensor([40.0]), PB_REFERENCE["male"])
    assert abs(float(tall) - PB_REFERENCE["male"].h1) < 0.5


def test_hill_monotone_and_bounded() -> None:
    x = torch.linspace(0.0, 500.0, 50)
    y = hill(x, km=100.0, n=2.0)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)
    assert torch.all(y[1:] - y[:-1] >= -1e-6)


def test_gh_feedback_sign() -> None:
    cfg = PhysicsCfg()
    gh = torch.tensor([[5.0]])
    low = torch.tensor([[50.0]])
    high = torch.tensor([[300.0]])
    ghrh = torch.tensor([[1.0]])
    dgh_low, _ = gh_igf1_rates(gh, low, ghrh, cfg)
    dgh_high, _ = gh_igf1_rates(gh, high, ghrh, cfg)
    assert float(dgh_high) < float(dgh_low)
