import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class CnnEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        widths = [cfg.in_channels, 64, 128, 256, 512]
        blocks: list[nn.Module] = []
        for i in range(len(widths) - 1):
            blocks.append(nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1))
            blocks.append(nn.BatchNorm2d(widths[i + 1]))
            blocks.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(widths[-1], cfg.d_model)

    def forward(self, image: Tensor) -> Tensor:
        feat = self.pool(self.body(image)).flatten(1)
        return self.proj(feat)


class MlpPhenotypeEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hpo_terms, cfg.gat_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gat_hidden, cfg.d_model),
        )

    def forward(self, phenotype: Tensor, adjacency: Tensor) -> Tensor:
        return self.net(phenotype)


class MeanVariantEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.embed = nn.Linear(cfg.variant_feat_dim, cfg.d_model)

    def forward(self, variant_feats: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.embed(variant_feats)
        weight = mask.unsqueeze(-1)
        pooled = (tokens * weight).sum(1) / weight.sum(1).clamp(min=1.0)
        return tokens, pooled


class EarlyFusion(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * cfg.d_model, cfg.d_model)

    def forward(self, image: Tensor, phenotype: Tensor, variant: Tensor) -> tuple[Tensor, Tensor]:
        fused = self.proj(torch.cat([image, phenotype, variant], dim=-1))
        tokens = torch.stack([image, phenotype, variant], dim=1)
        return tokens, fused


class LateFusion(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, image: Tensor, phenotype: Tensor, variant: Tensor) -> tuple[Tensor, Tensor]:
        tokens = torch.stack([image, phenotype, variant], dim=1)
        weight = torch.softmax(self.scale, dim=0).view(1, 3, 1)
        fused = (tokens * weight).sum(dim=1)
        return tokens, fused
