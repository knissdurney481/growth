from collections.abc import Callable

import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class BoneAgeHead(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.net(fused).squeeze(-1)


def bone_age_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(prediction - target))


def enable_dropout(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()


def mc_dropout_predict(forward_fn: Callable[..., Tensor], samples: int) -> tuple[Tensor, Tensor]:
    draws = torch.stack([forward_fn() for _ in range(samples)], dim=0)
    mean = draws.mean(dim=0)
    var = draws.var(dim=0, unbiased=False)
    return mean, var
