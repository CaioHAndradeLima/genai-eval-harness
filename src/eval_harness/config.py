"""Load and validate YAML experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eval_harness.models import EvalDataset, ExperimentMatrix, ExperimentVariant


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"expected mapping at root of {path}")
    return data


def load_experiment_matrix(path: Path | str) -> ExperimentMatrix:
    """Parse an experiment matrix YAML into validated models."""
    data = _load_yaml(Path(path))
    return ExperimentMatrix.model_validate(data)


def load_variant(path: Path | str) -> ExperimentVariant:
    """Parse a single experiment variant YAML."""
    data = _load_yaml(Path(path))
    return ExperimentVariant.model_validate(data)


def load_dataset(path: Path | str) -> EvalDataset:
    """Parse a golden eval dataset JSON/YAML file."""
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"dataset not found: {file_path}")

    with file_path.open(encoding="utf-8") as handle:
        if file_path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            import json

            data = json.load(handle)

    return EvalDataset.model_validate(data)


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from start until pyproject.toml is found."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def resolve_path(base: Path, relative: str) -> Path:
    """Resolve a path relative to project root, then matrix directory."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    root = find_project_root(base)
    for anchor in (root, base):
        resolved = (anchor / candidate).resolve()
        if resolved.exists():
            return resolved
    return (root / candidate).resolve()
