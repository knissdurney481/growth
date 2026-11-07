from dataclasses import dataclass

import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg

AGE_GRID_START = 2.0
AGE_GRID_END = 18.0


@dataclass
class GrowthSeries:
    latent: Tensor
    observation: Tensor
    height: Tensor
    n_density: Tensor
    gh: Tensor
    igf1: Tensor
    ghrh: Tensor
    ages: Tensor
    dt: float


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelCfg, noise_std: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.steps = cfg.rollout_steps
        self.noise_std = noise_std
        self.to_latent = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.transition = nn.GRUCell(cfg.action_dim, cfg.latent_dim)
        self.decoder = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.growth_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim, 5),
        )
        ages = torch.linspace(AGE_GRID_START, AGE_GRID_END, self.steps)
        self.register_buffer("ages", ages)
        self.ages: Tensor

    def _decode_growth(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raw = self.growth_head(latent)
        height = 110.0 + 40.0 * raw[..., 0]
        n_density = torch.sigmoid(raw[..., 1])
        gh = 20.0 * torch.sigmoid(raw[..., 2])
        igf1 = 400.0 * torch.sigmoid(raw[..., 3])
        ghrh = 10.0 * torch.sigmoid(raw[..., 4])
        return height, n_density, gh, igf1, ghrh

    def forward(self, fused: Tensor, actions: Tensor | None = None) -> GrowthSeries:
        b = fused.shape[0]
        device = fused.device
        z = self.to_latent(fused)
        if actions is None:
            actions = torch.zeros(b, self.steps, self.cfg.action_dim, device=device)
        states = [z]
        for t in range(self.steps - 1):
            z = self.transition(actions[:, t], z)
            if self.noise_std > 0.0 and self.training:
                z = z + self.noise_std * torch.randn_like(z)
            states.append(z)
        latent = torch.stack(states, dim=1)
        observation = self.decoder(latent)
        height, n_density, gh, igf1, ghrh = self._decode_growth(observation)
        dt = float((AGE_GRID_END - AGE_GRID_START) / max(1, self.steps - 1))
        return GrowthSeries(
            latent=latent,
            observation=observation,
            height=height,
            n_density=n_density,
            gh=gh,
            igf1=igf1,
            ghrh=ghrh,
            ages=self.ages.to(device),
            dt=dt,
        )
