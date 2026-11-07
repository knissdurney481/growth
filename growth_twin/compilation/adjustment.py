import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ObjectiveCfg


class Objective(nn.Module):
    def __init__(self, cfg: ObjectiveCfg) -> None:
        super().__init__()
        self.mode = cfg.mode
        self.log_sigma_ba = nn.Parameter(torch.zeros(()))
        self.log_sigma_rank = nn.Parameter(torch.zeros(()))

    def forward(
        self, bone_age: Tensor, rank: Tensor, physics: Tensor, lambda_rank: float
    ) -> dict[str, Tensor]:
        if self.mode == "uncertainty":
            ba_term = 0.5 * torch.exp(-2.0 * self.log_sigma_ba) * bone_age + self.log_sigma_ba
            rank_term = 0.5 * torch.exp(-2.0 * self.log_sigma_rank) * rank + self.log_sigma_rank
            total = ba_term + rank_term + physics
        else:
            total = bone_age + lambda_rank * rank + physics
        return {"total": total, "bone_age": bone_age, "rank": rank, "physics": physics}
