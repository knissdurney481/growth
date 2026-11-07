import torch

from growth_twin.graticule.schema import ObjectiveCfg
from growth_twin.relief.gazetteer import (
    pairwise_ranking_loss,
    pathogenicity_score,
    prioritize_variants,
    variant_score,
)
from growth_twin.survey.instruments import FEATURE_INDEX, VARIANT_FEAT_DIM


def test_pairwise_loss_zero_when_ordered() -> None:
    rank = torch.tensor([[5.0, 0.0, 0.0]])
    label = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    loss = pairwise_ranking_loss(rank, label, mask, margin=1.0)
    assert float(loss) == 0.0


def test_pairwise_loss_positive_when_misordered() -> None:
    rank = torch.tensor([[0.0, 2.0, 2.0]])
    label = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    loss = pairwise_ranking_loss(rank, label, mask, margin=1.0)
    assert float(loss) > 0.0


def test_pathogenicity_score_linear() -> None:
    feats = torch.zeros(1, 2, VARIANT_FEAT_DIM)
    feats[0, 0, FEATURE_INDEX["pathogenicity"]] = torch.tensor([1.0, 1.0, 1.0])
    score = pathogenicity_score(feats, (0.4, 0.3, 0.3))
    assert abs(float(score[0, 0]) - 1.0) < 1e-6
    assert abs(float(score[0, 1]) - 0.0) < 1e-6


def test_variant_score_prefers_pathogenic() -> None:
    feats = torch.zeros(1, 2, VARIANT_FEAT_DIM)
    feats[0, 0, FEATURE_INDEX["pathogenicity"]] = torch.tensor([0.9, 0.9, 0.9])
    feats[0, 0, FEATURE_INDEX["gene_constraint"]] = torch.tensor([0.9, 0.1])
    pheno = torch.ones(1, 4)
    gene_pheno = torch.zeros(1, 2, 4)
    gene_pheno[0, 0] = torch.ones(4)
    score = variant_score(feats, pheno, gene_pheno, ObjectiveCfg())
    assert float(score[0, 0]) > float(score[0, 1])


def test_prioritize_sorts_and_masks() -> None:
    rank = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    order, _scores = prioritize_variants(rank, mask)
    assert order[0, 0].item() == 1
    assert order[0, -1].item() == 3
