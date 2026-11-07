import torch
from torch import Tensor, nn

from growth_twin.datum.benchmarks import (
    gh_igf1_rates,
    growth_plate_rate,
    preece_baines_height_by_sex,
)
from growth_twin.graticule.schema import PhysicsCfg


def preece_baines_residual(height: Tensor, ages: Tensor, sex: Tensor) -> Tensor:
    target = preece_baines_height_by_sex(ages.view(1, -1), sex)
    return torch.mean((height - target) ** 2)


def growth_plate_residual(
    n_density: Tensor, igf1: Tensor, gh: Tensor, dt: float, cfg: PhysicsCfg
) -> Tensor:
    dndt = (n_density[:, 1:] - n_density[:, :-1]) / dt
    rhs = growth_plate_rate(n_density[:, :-1], igf1[:, :-1], gh[:, :-1], cfg)
    return torch.mean((dndt - rhs) ** 2)


def ghs_residual(gh: Tensor, igf1: Tensor, ghrh: Tensor, dt: float, cfg: PhysicsCfg) -> Tensor:
    dghdt = (gh[:, 1:] - gh[:, :-1]) / dt
    rhs, _ = gh_igf1_rates(gh[:, :-1], igf1[:, :-1], ghrh[:, :-1], cfg)
    return torch.mean((dghdt - rhs) ** 2)


class PhysicsLoss(nn.Module):
    def __init__(self, cfg: PhysicsCfg, ablate_pb: bool, ablate_gp: bool, ablate_ghs: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.ablate_pb = ablate_pb
        self.ablate_gp = ablate_gp
        self.ablate_ghs = ablate_ghs

    def forward(
        self,
        height: Tensor,
        n_density: Tensor,
        gh: Tensor,
        igf1: Tensor,
        ghrh: Tensor,
        ages: Tensor,
        sex: Tensor,
        dt: float,
    ) -> dict[str, Tensor]:
        zero = height.new_zeros(())
        pb = zero if self.ablate_pb else preece_baines_residual(height, ages, sex)
        gp = zero if self.ablate_gp else growth_plate_residual(n_density, igf1, gh, dt, self.cfg)
        ghs = zero if self.ablate_ghs else ghs_residual(gh, igf1, ghrh, dt, self.cfg)
        total = self.cfg.lambda_pb * pb + self.cfg.lambda_gp * gp + self.cfg.lambda_ghs * ghs
        return {"pb": pb, "gp": gp, "ghs": ghs, "total": total}
