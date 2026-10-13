"""Project paths from `configs/paths.yaml`, resolved against the repository root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(config: Path = REPO_ROOT / "configs" / "paths.yaml") -> dict[str, Path]:
    with open(config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {key: (REPO_ROOT / value).resolve() for key, value in raw.items()}
