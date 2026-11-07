from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class Batch:
    image: Tensor
    phenotype: Tensor
    adjacency: Tensor
    variant_feats: Tensor
    variant_mask: Tensor
    gene_pheno: Tensor
    bone_age: Tensor
    age_years: Tensor
    sex: Tensor
    variant_label: Tensor

    def to(self, device: torch.device) -> "Batch":
        return replace(
            self,
            image=self.image.to(device),
            phenotype=self.phenotype.to(device),
            adjacency=self.adjacency.to(device),
            variant_feats=self.variant_feats.to(device),
            variant_mask=self.variant_mask.to(device),
            gene_pheno=self.gene_pheno.to(device),
            bone_age=self.bone_age.to(device),
            age_years=self.age_years.to(device),
            sex=self.sex.to(device),
            variant_label=self.variant_label.to(device),
        )

    def __len__(self) -> int:
        return int(self.image.shape[0])


def slice_batch(batch: Batch, index: Tensor) -> Batch:
    return Batch(
        image=batch.image[index],
        phenotype=batch.phenotype[index],
        adjacency=batch.adjacency,
        variant_feats=batch.variant_feats[index],
        variant_mask=batch.variant_mask[index],
        gene_pheno=batch.gene_pheno[index],
        bone_age=batch.bone_age[index],
        age_years=batch.age_years[index],
        sex=batch.sex[index],
        variant_label=batch.variant_label[index],
    )


class BatchLoader:
    def __init__(
        self, batch: Batch, batch_size: int, shuffle: bool, generator: torch.Generator
    ) -> None:
        self.batch = batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.n = len(batch)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        order = (
            torch.randperm(self.n, generator=self.generator)
            if self.shuffle
            else torch.arange(self.n)
        )
        for start in range(0, self.n, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield slice_batch(self.batch, idx)
