import json
import os

import torch
import tyro

from growth_twin.atlas import DigitalGrowthTwin, build_dgt
from growth_twin.compilation.declination import seed_everything
from growth_twin.compilation.traverse import Trainer
from growth_twin.graticule.schema import ExperimentCfg
from growth_twin.graticule.sheets import load_sheet
from growth_twin.relief.gazetteer import prioritize_variants
from growth_twin.survey.cohort import build_cohort
from growth_twin.survey.fieldbook import BatchLoader


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(cfg: ExperimentCfg, split: str, shuffle: bool) -> BatchLoader:
    cohort = build_cohort(cfg.data, cfg.model, split, cfg.train.seed)
    gen = torch.Generator()
    gen.manual_seed(cfg.train.seed + 13)
    return BatchLoader(cohort.whole(), cfg.train.batch_size, shuffle, gen)


def _load_model(
    sheet: str, checkpoint: str, device: torch.device
) -> tuple[DigitalGrowthTwin, ExperimentCfg]:
    cfg = load_sheet(sheet)
    model = build_dgt(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    return model, cfg


def chart(sheet: str = "configs/experiment/main.yaml", steps: int = 0) -> None:
    cfg = load_sheet(sheet)
    seed_everything(cfg.train.seed)
    device = _device()
    train_loader = _loader(cfg, "train", shuffle=True)
    model = build_dgt(cfg)
    trainer = Trainer(model, cfg, device)
    if steps > 0:
        done = 0
        while done < steps:
            for batch in train_loader:
                trainer.train_step(batch, cfg.train.epochs)
                done += 1
                if done >= steps:
                    break
    else:
        val_loader = _loader(cfg, "val", shuffle=False)
        trainer.fit(train_loader, val_loader)
    trainer.save(os.path.join(cfg.train.out_dir, "final.ckpt"), cfg.train.epochs)


def assess(sheet: str, checkpoint: str) -> None:
    device = _device()
    model, cfg = _load_model(sheet, checkpoint, device)
    from growth_twin.appraisal.protocol import evaluate_bone_age, evaluate_variants

    test_loader = _loader(cfg, "test", shuffle=False)
    bone = evaluate_bone_age(
        model, _loader(cfg, "test", shuffle=False), device, cfg.train.mc_samples
    )
    variant = evaluate_variants(model, test_loader, device)
    print(json.dumps({"bone_age": bone, "variant": variant}, indent=2))


def locate(sheet: str, checkpoint: str) -> None:
    device = _device()
    model, cfg = _load_model(sheet, checkpoint, device)
    model.eval()
    loader = _loader(cfg, "test", shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        out = model(batch)
        order, _scores = prioritize_variants(out.rank, batch.variant_mask)
    payload = {
        "bone_age_months": [round(float(v), 2) for v in out.bone_age[:5].tolist()],
        "top_variants_case0": order[0, :10].tolist(),
    }
    print(json.dumps(payload, indent=2))


class _BoneAgeExport(torch.nn.Module):
    def __init__(self, model: DigitalGrowthTwin) -> None:
        super().__init__()
        self.image_encoder = model.image_encoder
        self.fusion = model.fusion
        self.head = model.bone_age_head
        self.d_model = model.d_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rep = self.image_encoder(image)
        zeros = torch.zeros_like(rep)
        _, fused = self.fusion(rep, zeros, zeros)
        return self.head(fused)


def engrave(sheet: str, checkpoint: str, out: str = "growth_twin_boneage.onnx") -> None:
    device = torch.device("cpu")
    model, cfg = _load_model(sheet, checkpoint, device)
    wrapper = _BoneAgeExport(model).eval()
    example_input = torch.zeros(1, cfg.model.in_channels, cfg.model.img_size, cfg.model.img_size)
    torch.onnx.export(
        wrapper,
        (example_input,),
        out,
        input_names=["image"],
        output_names=["bone_age"],
        dynamic_axes={"image": {0: "batch"}, "bone_age": {0: "batch"}},
        opset_version=17,
    )
    print(out)


def entrypoint() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {"chart": chart, "assess": assess, "locate": locate, "engrave": engrave}
    )


if __name__ == "__main__":
    entrypoint()
