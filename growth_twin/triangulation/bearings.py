import torch
import torch.nn.functional as F
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, concat: bool) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: Tensor, adjacency: Tensor) -> Tensor:
        b, n, _ = h.shape
        wh = self.lin(h).view(b, n, self.heads, self.out_dim)
        src = (wh * self.a_src).sum(-1).permute(0, 2, 1)
        dst = (wh * self.a_dst).sum(-1).permute(0, 2, 1)
        scores = src.unsqueeze(3) + dst.unsqueeze(2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        mask = adjacency.unsqueeze(0).unsqueeze(0) > 0
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhjk,bkhd->bjhd", alpha, wh)
        if self.concat:
            return out.reshape(b, n, self.heads * self.out_dim)
        return out.mean(dim=2)


class PhenotypeEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.term_embed = nn.Embedding(cfg.hpo_terms, cfg.d_model)
        self.b_present = nn.Parameter(torch.zeros(cfg.d_model))
        per_head = max(1, cfg.gat_hidden // cfg.gat_heads)
        layers: list[GraphAttentionLayer] = []
        in_dim = cfg.d_model
        for _ in range(cfg.gat_layers - 1):
            layers.append(
                GraphAttentionLayer(in_dim, per_head, cfg.gat_heads, cfg.gat_dropout, True)
            )
            in_dim = per_head * cfg.gat_heads
        layers.append(GraphAttentionLayer(in_dim, cfg.d_model, 1, cfg.gat_dropout, False))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ELU()
        nn.init.trunc_normal_(self.b_present, std=0.02)

    def forward(self, phenotype: Tensor, adjacency: Tensor) -> Tensor:
        b, n = phenotype.shape
        ids = torch.arange(n, device=phenotype.device)
        base = self.term_embed(ids).unsqueeze(0).expand(b, -1, -1)
        h = base + phenotype.unsqueeze(-1) * self.b_present
        for i, layer in enumerate(self.layers):
            h = layer(h, adjacency)
            if i < len(self.layers) - 1:
                h = self.act(h)
        weight = phenotype.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp(min=1.0)
        return (h * weight).sum(dim=1) / denom
