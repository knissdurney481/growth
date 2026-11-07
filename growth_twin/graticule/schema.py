from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCfg:
    d_model: int = 768
    img_size: int = 512
    patch_size: int = 16
    in_channels: int = 1
    vit_depth: int = 12
    vit_heads: int = 12
    vit_mlp_ratio: float = 4.0
    pretrained_vit: bool = True
    hpo_terms: int = 847
    gat_hidden: int = 256
    gat_layers: int = 3
    gat_heads: int = 4
    gat_dropout: float = 0.3
    variant_feat_dim: int = 15
    variant_hidden: int = 512
    variant_layers: int = 4
    variant_heads: int = 8
    max_variants: int = 100
    fusion_layers: int = 2
    fusion_heads: int = 8
    latent_dim: int = 128
    action_dim: int = 4
    rollout_steps: int = 8
    dropout: float = 0.2


@dataclass(frozen=True)
class DataCfg:
    dataset: str = "fusion"
    n_train: int = 700
    n_val: int = 150
    n_test: int = 150
    candidates_per_case: int = 50
    image_size: int = 512
    root: str = ""


@dataclass(frozen=True)
class PhysicsCfg:
    lambda_pb: float = 0.1
    lambda_gp: float = 0.05
    lambda_ghs: float = 0.05
    proliferation_rate: float = 0.05
    carrying_capacity: float = 1.0e6
    hill_coefficient: float = 2.0
    km_igf1: float = 100.0
    km_gh: float = 5.0
    k_ghrh: float = 0.8
    k_deg: float = 0.3
    k_fb: float = 0.2
    k_prod: float = 0.5
    k_clear: float = 0.25


@dataclass(frozen=True)
class ObjectiveCfg:
    mode: str = "fixed"
    lambda_rank_max: float = 0.5
    lambda_reg: float = 1.0e-5
    rank_margin: float = 1.0
    score_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    path_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    gene_weights: tuple[float, float] = (0.6, 0.4)


@dataclass(frozen=True)
class TrainCfg:
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    world_size: int = 4
    grad_accum: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    eta_min: float = 1.0e-6
    grad_clip: float = 1.0
    amp: bool = False
    seed: int = 1234
    mc_samples: int = 30
    curriculum_phase1: int = 20
    curriculum_phase2: int = 50
    steps_per_epoch: int = 0
    log_every: int = 10
    out_dir: str = "runs/main"


@dataclass(frozen=True)
class ExperimentCfg:
    name: str = "main"
    task: str = "fusion"
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    objective: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    ablate_pb: bool = False
    ablate_gp: bool = False
    ablate_ghs: bool = False
    image_encoder: str = "vit"
    phenotype_encoder: str = "gat"
    variant_encoder: str = "transformer"
    fusion: str = "cross_modal"
    use_image: bool = True
    use_phenotype: bool = True
    use_variant: bool = True
