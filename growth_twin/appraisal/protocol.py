from collections.abc import Iterable

import numpy as np
import torch

from growth_twin.appraisal.inference_stats import coverage
from growth_twin.appraisal.metrics import (
    auprc,
    auroc,
    mad,
    mrr,
    pearson_r,
    precision_at_k,
    recall_at_k,
    rmse,
    within_months,
)
from growth_twin.atlas import DigitalGrowthTwin
from growth_twin.relief.elevation import enable_dropout, mc_dropout_predict
from growth_twin.survey.fieldbook import Batch


@torch.no_grad()
def evaluate_bone_age(
    model: DigitalGrowthTwin, loader: Iterable[Batch], device: torch.device, mc_samples: int
) -> dict[str, float]:
    model.eval()
    enable_dropout(model)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        mean, var = mc_dropout_predict(lambda b=batch: model(b).bone_age, mc_samples)
        std = torch.sqrt(var.clamp(min=0.0))
        preds.append(mean.cpu().numpy())
        targets.append(batch.bone_age.cpu().numpy())
        lowers.append((mean - 1.96 * std).cpu().numpy())
        uppers.append((mean + 1.96 * std).cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    lo = np.concatenate(lowers)
    hi = np.concatenate(uppers)
    return {
        "mad": mad(p, t),
        "rmse": rmse(p, t),
        "pearson": pearson_r(p, t),
        "within12": within_months(p, t, 12.0),
        "coverage95": coverage(lo, hi, t),
        "avg_width": float(np.mean(hi - lo)),
    }


@torch.no_grad()
def evaluate_variants(
    model: DigitalGrowthTwin, loader: Iterable[Batch], device: torch.device
) -> dict[str, float]:
    model.eval()
    rows_score: list[np.ndarray] = []
    rows_label: list[np.ndarray] = []
    rows_mask: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        rows_score.append(out.rank.cpu().numpy())
        rows_label.append(batch.variant_label.cpu().numpy())
        rows_mask.append(batch.variant_mask.cpu().numpy())
    scores = np.concatenate(rows_score, axis=0)
    labels = np.concatenate(rows_label, axis=0)
    mask = np.concatenate(rows_mask, axis=0)
    valid = mask > 0.5
    flat_scores = scores[valid]
    flat_labels = labels[valid]
    return {
        "auroc": auroc(flat_scores, flat_labels),
        "auprc": auprc(flat_scores, flat_labels),
        "p_at_10": precision_at_k(scores, labels, 10),
        "recall_at_10": recall_at_k(scores, labels, 10),
        "mrr": mrr(scores, labels),
    }
