# Digital Growth Twin

Digital Growth Twin provides research components for pediatric growth modelling, bone-age estimation, phenotype encoding, and variant prioritization.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

Patient images, phenotype records, variant labels, matched multimodal datasets, and trained checkpoints are not included. Dataset-specific loaders must be implemented before training, evaluation, and inference can run. The command-line workflows stop when data are unavailable. The manuscript's reported results have not been reproduced using this repository.

When pretrained image weights are requested, the external weights and optional dependency must be available; loading failures raise an error. The outputs require independent validation.

`survey` defines data structures and preprocessing; `triangulation` contains encoders and fusion; `projection` contains the growth model; `datum` contains physical constraints; `compilation` contains training; `appraisal` contains evaluation metrics. Configuration files are in `configs/`.

`scripts/prepare_data.sh` lists potential data sources. Their access and usage terms apply.

```bash
pytest -q
```

This is research software and has not been validated for clinical use.
