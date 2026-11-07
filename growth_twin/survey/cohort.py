from growth_twin.graticule.schema import DataCfg, ModelCfg
from growth_twin.survey.fieldbook import Batch


class Cohort:
    def __init__(self, batch: Batch) -> None:
        self.batch = batch

    def whole(self) -> Batch:
        return self.batch


class BoneAgeCohort(Cohort):
    pass


class VariantBenchmark(Cohort):
    pass


class FusionCohort(Cohort):
    pass


def build_cohort(cfg: DataCfg, model: ModelCfg, split: str, seed: int) -> Cohort:
    raise NotImplementedError(
        "Patient image, phenotype, and variant data loaders must be integrated."
    )
