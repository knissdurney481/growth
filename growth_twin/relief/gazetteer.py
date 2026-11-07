import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg, ObjectiveCfg
from growth_twin.survey.instruments import FEATURE_INDEX


def pathogenicity_score(feats: Tensor, weights: tuple[float, float, float]) -> Tensor:
    block = feats[..., FEATURE_INDEX["pathogenicity"]]
    coeff = feats.new_tensor(weights)
    return (block * coeff).sum(-1)


def phenotype_score(phenotype: Tensor, gene_pheno: Tensor) -> Tensor:
    p = phenotype.unsqueeze(1)
    num = (p * gene_pheno).sum(-1)
    denom = p.norm(dim=-1) * gene_pheno.norm(dim=-1) + 1.0e-8
    return num / denom


def gene_score(feats: Tensor, weights: tuple[float, float]) -> Tensor:
    block = feats[..., FEATURE_INDEX["gene_constraint"]]
    coeff = feats.new_tensor(weights)
    return (block * coeff).sum(-1)


def variant_score(
    feats: Tensor,
    phenotype: Tensor,
    gene_pheno: Tensor,
    cfg: ObjectiveCfg,
) -> Tensor:
    s_path = pathogenicity_score(feats, cfg.path_weights)
    s_phen = phenotype_score(phenotype, gene_pheno)
    s_gene = gene_score(feats, cfg.gene_weights)
    w1, w2, w3 = cfg.score_weights
    return w1 * s_path + w2 * s_phen + w3 * s_gene


class RankingHead(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, score: Tensor, variant_tokens: Tensor, phenotype_repr: Tensor) -> Tensor:
        m = variant_tokens.shape[1]
        phen = phenotype_repr.unsqueeze(1).expand(-1, m, -1)
        features = torch.cat([score.unsqueeze(-1), variant_tokens, phen], dim=-1)
        return self.net(features).squeeze(-1)


def pairwise_ranking_loss(rank: Tensor, label: Tensor, mask: Tensor, margin: float) -> Tensor:
    valid = mask > 0.5
    pos = (label > 0.5) & valid
    neg = (label < 0.5) & valid
    diff = margin + rank.unsqueeze(1) - rank.unsqueeze(2)
    hinge = torch.clamp(diff, min=0.0)
    pair_mask = pos.unsqueeze(2) & neg.unsqueeze(1)
    count = pair_mask.sum().clamp(min=1)
    return (hinge * pair_mask).sum() / count


def prioritize_variants(rank: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    masked = rank.masked_fill(mask < 0.5, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True)
    sorted_scores = torch.gather(rank, -1, order)
    return order, sorted_scores
