from dataclasses import dataclass

import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ExperimentCfg, ModelCfg, ObjectiveCfg
from growth_twin.projection.meridian import GrowthSeries, WorldModel
from growth_twin.relief.elevation import BoneAgeHead
from growth_twin.relief.gazetteer import RankingHead, variant_score
from growth_twin.survey.fieldbook import Batch
from growth_twin.triangulation.alternates import (
    CnnEncoder,
    EarlyFusion,
    LateFusion,
    MeanVariantEncoder,
    MlpPhenotypeEncoder,
)
from growth_twin.triangulation.bearings import PhenotypeEncoder
from growth_twin.triangulation.cadastre import VariantEncoder
from growth_twin.triangulation.intersection import CrossModalFusion
from growth_twin.triangulation.sightlines import RadiographEncoder


@dataclass
class DgtOutput:
    bone_age: Tensor
    raw_score: Tensor
    rank: Tensor
    growth: GrowthSeries
    fused: Tensor
    variant_tokens: Tensor


class DigitalGrowthTwin(nn.Module):
    def __init__(self, cfg: ExperimentCfg) -> None:
        super().__init__()
        model = cfg.model
        self.cfg = cfg
        self.objective: ObjectiveCfg = cfg.objective
        self.use_image = cfg.use_image
        self.use_phenotype = cfg.use_phenotype
        self.use_variant = cfg.use_variant
        self.d_model = model.d_model

        self.image_encoder = (
            RadiographEncoder(model) if cfg.image_encoder == "vit" else CnnEncoder(model)
        )
        self.phenotype_encoder = (
            PhenotypeEncoder(model)
            if cfg.phenotype_encoder == "gat"
            else MlpPhenotypeEncoder(model)
        )
        self.variant_encoder = (
            VariantEncoder(model)
            if cfg.variant_encoder == "transformer"
            else MeanVariantEncoder(model)
        )
        self.fusion = _make_fusion(cfg.fusion, model)
        self.world_model = WorldModel(model)
        self.bone_age_head = BoneAgeHead(model)
        self.ranking_head = RankingHead(model)

    def _zeros(self, batch: Batch, *shape: int) -> Tensor:
        return torch.zeros(*shape, device=batch.image.device)

    def encode(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b = len(batch)
        m = batch.variant_feats.shape[1]
        if self.use_image:
            image_rep = self.image_encoder(batch.image)
        else:
            image_rep = self._zeros(batch, b, self.d_model)
        if self.use_phenotype:
            phen_rep = self.phenotype_encoder(batch.phenotype, batch.adjacency)
        else:
            phen_rep = self._zeros(batch, b, self.d_model)
        if self.use_variant:
            tokens, var_rep = self.variant_encoder(batch.variant_feats, batch.variant_mask)
        else:
            tokens = self._zeros(batch, b, m, self.d_model)
            var_rep = self._zeros(batch, b, self.d_model)
        return image_rep, phen_rep, tokens, var_rep

    def forward(self, batch: Batch) -> DgtOutput:
        image_rep, phen_rep, variant_tokens, var_rep = self.encode(batch)
        fused_tokens, fused = self.fusion(image_rep, phen_rep, var_rep)
        growth = self.world_model(fused)
        bone_age = self.bone_age_head(fused)
        raw_score = variant_score(
            batch.variant_feats, batch.phenotype, batch.gene_pheno, self.objective
        )
        fused_phenotype = fused_tokens[:, 1]
        rank = self.ranking_head(raw_score, variant_tokens, fused_phenotype)
        return DgtOutput(
            bone_age=bone_age,
            raw_score=raw_score,
            rank=rank,
            growth=growth,
            fused=fused,
            variant_tokens=variant_tokens,
        )


def _make_fusion(kind: str, model: ModelCfg) -> nn.Module:
    if kind == "early":
        return EarlyFusion(model)
    if kind == "late":
        return LateFusion(model)
    return CrossModalFusion(model)


def build_dgt(cfg: ExperimentCfg) -> DigitalGrowthTwin:
    return DigitalGrowthTwin(cfg)
