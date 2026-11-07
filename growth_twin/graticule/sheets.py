import dataclasses
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

from growth_twin.graticule.schema import ExperimentCfg

T = TypeVar("T")


def _coerce(tp: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return _build(cast(type[Any], tp), value)
    if tp is tuple or getattr(tp, "__origin__", None) is tuple:
        return tuple(value)
    return value


def _build(cls: type[T], data: dict[str, Any]) -> T:
    kinds = {f.name: f.type for f in dataclasses.fields(cast(Any, cls))}
    out: dict[str, Any] = {}
    for key, raw in data.items():
        if key not in kinds:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        out[key] = _coerce(kinds[key], raw)
    return cls(**out)


def load_sheet(path: str | Path) -> ExperimentCfg:
    text = Path(path).read_text()
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment sheet must be a mapping")
    return _build(ExperimentCfg, payload)


def merge_sheet(base: ExperimentCfg, overrides: dict[str, Any]) -> ExperimentCfg:
    names = {f.name for f in dataclasses.fields(ExperimentCfg)}
    unknown = set(overrides) - names
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    top: dict[str, Any] = {}
    for f in dataclasses.fields(ExperimentCfg):
        if f.name in overrides:
            top[f.name] = _coerce(f.type, overrides[f.name])
    return dataclasses.replace(base, **top)
