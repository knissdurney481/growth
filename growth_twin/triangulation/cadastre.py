import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        logits = self.score(tokens).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        beta = torch.softmax(logits, dim=1)
        return torch.einsum("bm,bmd->bd", beta, tokens)


class VariantEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.embed = nn.Linear(cfg.variant_feat_dim, cfg.variant_hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.variant_hidden,
            nhead=cfg.variant_heads,
            dim_feedforward=cfg.variant_hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.variant_layers)
        self.proj = nn.Linear(cfg.variant_hidden, cfg.d_model)
        self.pool = AttentionPool(cfg.d_model)

    def forward(self, variant_feats: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        key_padding = mask < 0.5
        h = self.embed(variant_feats)
        h = self.encoder(h, src_key_padding_mask=key_padding)
        tokens = self.proj(h)
        pooled = self.pool(tokens, mask)
        return tokens, pooled
