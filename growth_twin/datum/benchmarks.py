from dataclasses import dataclass

import torch
from torch import Tensor

from growth_twin.graticule.schema import PhysicsCfg


@dataclass(frozen=True)
class PreeceBainesParams:
    h1: float
    h_theta: float
    theta: float
    s0: float
    s1: float


PB_REFERENCE: dict[str, PreeceBainesParams] = {
    "male": PreeceBainesParams(h1=176.8, h_theta=154.2, theta=14.0, s0=0.145, s1=1.628),
    "female": PreeceBainesParams(h1=163.7, h_theta=148.6, theta=12.1, s0=0.182, s1=2.147),
}


def preece_baines_height(t_years: Tensor, params: PreeceBainesParams) -> Tensor:
    delta = t_years - params.theta
    denom = torch.exp(params.s0 * delta) + torch.exp(params.s1 * delta)
    return params.h1 - 2.0 * (params.h1 - params.h_theta) / denom


def preece_baines_height_by_sex(t_years: Tensor, sex: Tensor) -> Tensor:
    male = preece_baines_height(t_years, PB_REFERENCE["male"])
    female = preece_baines_height(t_years, PB_REFERENCE["female"])
    is_male = sex.view(-1, 1) > 0.5
    return torch.where(is_male, male, female)


def hill(x: Tensor, km: float, n: float) -> Tensor:
    xn = torch.clamp(x, min=0.0) ** n
    return xn / (km**n + xn)


def growth_plate_rate(n_density: Tensor, igf1: Tensor, gh: Tensor, cfg: PhysicsCfg) -> Tensor:
    logistic = cfg.proliferation_rate * n_density * (1.0 - n_density)
    return (
        logistic
        * hill(igf1, cfg.km_igf1, cfg.hill_coefficient)
        * hill(gh, cfg.km_gh, cfg.hill_coefficient)
    )


def gh_igf1_rates(gh: Tensor, igf1: Tensor, ghrh: Tensor, cfg: PhysicsCfg) -> tuple[Tensor, Tensor]:
    dgh = cfg.k_ghrh * ghrh - cfg.k_deg * gh - cfg.k_fb * igf1
    digf1 = cfg.k_prod * gh - cfg.k_clear * igf1
    return dgh, digf1
