from typing import Any

import torch
from torch import Tensor, nn

from growth_twin.graticule.schema import ModelCfg


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class RadiographEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.proj = nn.Conv2d(cfg.in_channels, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            _Block(cfg.d_model, cfg.vit_heads, cfg.vit_mlp_ratio, cfg.dropout)
            for _ in range(cfg.vit_depth)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        if cfg.pretrained_vit:
            _load_imagenet21k(self)

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.proj(image).flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


def _load_imagenet21k(encoder: RadiographEncoder) -> None:
    try:
        import timm

        donor: Any = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=True)
    except Exception as error:
        raise RuntimeError("Pretrained image encoder weights could not be loaded.") from error
    src = donor.blocks
    with torch.no_grad():
        for dst_m, ref in zip(encoder.blocks, src, strict=False):
            dst: Any = dst_m
            dst.norm1.weight.copy_(ref.norm1.weight)
            dst.norm1.bias.copy_(ref.norm1.bias)
            dst.attn.in_proj_weight.copy_(ref.attn.qkv.weight)
            dst.attn.in_proj_bias.copy_(ref.attn.qkv.bias)
            dst.attn.out_proj.weight.copy_(ref.attn.proj.weight)
            dst.attn.out_proj.bias.copy_(ref.attn.proj.bias)
            dst.norm2.weight.copy_(ref.norm2.weight)
            dst.norm2.bias.copy_(ref.norm2.bias)
            dst.mlp[0].weight.copy_(ref.mlp.fc1.weight)
            dst.mlp[0].bias.copy_(ref.mlp.fc1.bias)
            dst.mlp[3].weight.copy_(ref.mlp.fc2.weight)
            dst.mlp[3].bias.copy_(ref.mlp.fc2.bias)
        enc: Any = encoder
        enc.norm.weight.copy_(donor.norm.weight)
        enc.norm.bias.copy_(donor.norm.bias)
