import os
from collections.abc import Iterable

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from growth_twin.atlas import DigitalGrowthTwin
from growth_twin.compilation.adjustment import Objective
from growth_twin.datum.leveling import PhysicsLoss
from growth_twin.graticule.schema import ExperimentCfg
from growth_twin.projection.meridian import GrowthSeries
from growth_twin.relief.elevation import bone_age_loss
from growth_twin.relief.gazetteer import pairwise_ranking_loss
from growth_twin.survey.fieldbook import Batch


def curriculum_lambda_rank(epoch: int, phase1: int, phase2: int, lambda_max: float) -> float:
    span = max(1, phase2 - phase1)
    fraction = (epoch - phase1) / span
    return float(min(1.0, max(0.0, fraction)) * lambda_max)


def physics_gate(epoch: int, phase1: int, phase2: int) -> tuple[bool, bool, bool]:
    return True, epoch > phase2, epoch > phase2


class Trainer:
    def __init__(self, model: DigitalGrowthTwin, cfg: ExperimentCfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.objective = Objective(cfg.objective).to(device)
        self.physics = PhysicsLoss(cfg.physics, cfg.ablate_pb, cfg.ablate_gp, cfg.ablate_ghs).to(
            device
        )
        params = list(self.model.parameters()) + list(self.objective.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.train.lr,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.eta_min
        )
        self.use_amp = cfg.train.amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.best_val = float("inf")

    def _physics_total(self, growth: GrowthSeries, sex: torch.Tensor, epoch: int) -> torch.Tensor:
        comp = self.physics(
            growth.height,
            growth.n_density,
            growth.gh,
            growth.igf1,
            growth.ghrh,
            growth.ages,
            sex,
            growth.dt,
        )
        pb_on, gp_on, ghs_on = physics_gate(
            epoch, self.cfg.train.curriculum_phase1, self.cfg.train.curriculum_phase2
        )
        phys = comp["pb"].new_zeros(())
        if pb_on:
            phys = phys + self.cfg.physics.lambda_pb * comp["pb"]
        if gp_on:
            phys = phys + self.cfg.physics.lambda_gp * comp["gp"]
        if ghs_on:
            phys = phys + self.cfg.physics.lambda_ghs * comp["ghs"]
        return phys

    def train_step(self, batch: Batch, epoch: int) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)
        lambda_rank = curriculum_lambda_rank(
            epoch,
            self.cfg.train.curriculum_phase1,
            self.cfg.train.curriculum_phase2,
            self.cfg.objective.lambda_rank_max,
        )
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model(batch)
            ba = bone_age_loss(out.bone_age, batch.bone_age)
            rank = pairwise_ranking_loss(
                out.rank, batch.variant_label, batch.variant_mask, self.cfg.objective.rank_margin
            )
            phys = self._physics_total(out.growth, batch.sex, epoch)
            losses = self.objective(ba, rank, phys, lambda_rank)
        total = losses["total"]
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: float(v.detach()) for k, v in losses.items()}

    def fit(
        self, train_loader: Iterable[Batch], val_loader: Iterable[Batch] | None = None
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.train.epochs + 1):
            records: list[dict[str, float]] = [
                self.train_step(batch, epoch) for batch in train_loader
            ]
            self.scheduler.step()
            mean = _mean_records(records)
            if val_loader is not None:
                mean["val_mad"] = self.validate(val_loader)
                if mean["val_mad"] < self.best_val:
                    self.best_val = mean["val_mad"]
                    self.save(os.path.join(self.cfg.train.out_dir, "best.ckpt"), epoch)
            history.append(mean)
        return history

    @torch.no_grad()
    def validate(self, val_loader: Iterable[Batch]) -> float:
        self.model.eval()
        errors: list[float] = []
        for batch in val_loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            errors.append(float(torch.mean(torch.abs(out.bone_age - batch.bone_age))))
        return sum(errors) / max(1, len(errors))

    def save(self, path: str, epoch: int) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "seed": self.cfg.train.seed,
            "best_val": self.best_val,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: sum(r[k] for r in records) / len(records) for k in keys}
