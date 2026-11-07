import math

import torch
import torch.nn.functional as F
from torch import Tensor

INTENSITY_MEAN = 127.3
INTENSITY_STD = 63.2

FEATURE_INDEX: dict[str, slice] = {
    "pathogenicity": slice(0, 3),
    "allele_frequency": slice(3, 4),
    "conservation": slice(4, 6),
    "splicing": slice(6, 8),
    "gene_constraint": slice(8, 10),
    "consequence": slice(10, 14),
    "gene_phenotype": slice(14, 15),
}
VARIANT_FEAT_DIM = 15


def normalize_intensity(image: Tensor) -> Tensor:
    return (image - INTENSITY_MEAN) / INTENSITY_STD


def resize_pad(image: Tensor, size: int) -> Tensor:
    _b, _c, h, w = image.shape
    scale = size / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    left = pad_w // 2
    return F.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)


def _rotate(image: Tensor, angles_deg: Tensor) -> Tensor:
    theta = angles_deg * math.pi / 180.0
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    matrix = torch.stack(
        [
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin, cos, zeros], dim=-1),
        ],
        dim=-2,
    )
    grid = F.affine_grid(matrix, list(image.shape), align_corners=False)
    return F.grid_sample(image, grid, align_corners=False, padding_mode="zeros")


def augment_image(image: Tensor, generator: torch.Generator) -> Tensor:
    b = image.shape[0]
    device = image.device
    angles = (torch.rand(b, generator=generator).to(device) * 2.0 - 1.0) * 15.0
    out = _rotate(image, angles)
    flip = torch.rand(b, generator=generator).to(device) < 0.5
    flipped = torch.flip(out, dims=[3])
    out = torch.where(flip.view(b, 1, 1, 1), flipped, out)
    brightness = 1.0 + (torch.rand(b, 1, 1, 1, generator=generator).to(device) * 2.0 - 1.0) * 0.2
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, generator=generator).to(device) * 2.0 - 1.0) * 0.2
    mean = out.mean(dim=(2, 3), keepdim=True)
    return (out - mean) * contrast + mean * brightness


def patchify(image: Tensor, patch_size: int) -> Tensor:
    b, c, _h, _w = image.shape
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    return patches.view(b, patches.shape[1], c * patch_size * patch_size)


def build_variant_features(parts: dict[str, Tensor]) -> Tensor:
    sample = next(iter(parts.values()))
    m = sample.shape[0]
    out = torch.zeros(m, VARIANT_FEAT_DIM, dtype=torch.float32)
    for name, span in FEATURE_INDEX.items():
        if name in parts:
            block = parts[name]
            if block.dim() == 1:
                block = block.unsqueeze(-1)
            out[:, span] = block.to(torch.float32)
    return out
