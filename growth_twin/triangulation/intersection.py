import math

import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        q = self._split(self.q(query))
        k = self._split(self.k(context))
        v = self._split(self.v(context))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous()
        b, length, _, _ = ctx.shape
        return self.out(ctx.view(b, length, self.heads * self.head_dim))


class _FusionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm_q(tokens), self.norm_kv(tokens))
        return tokens + self.ff(self.norm_ff(tokens))


class CrossModalFusion(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.type_embed = nn.Parameter(torch.zeros(1, 3, cfg.d_model))
        self.layers = nn.ModuleList(
            _FusionLayer(cfg.d_model, cfg.fusion_heads, cfg.dropout)
            for _ in range(cfg.fusion_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, image: Tensor, phenotype: Tensor, variant: Tensor) -> tuple[Tensor, Tensor]:
        tokens = torch.stack([image, phenotype, variant], dim=1) + self.type_embed
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)
