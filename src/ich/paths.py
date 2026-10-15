"""Project paths from `configs/paths.yaml`, resolved against the repository root.

Machine-specific paths go in `configs/paths.local.yaml` (same keys, not tracked by git) and override the defaults.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(config: Path = REPO_ROOT / "configs" / "paths.yaml") -> dict[str, Path]:
    with open(config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    local = config.with_name(f"{config.stem}.local.yaml")
    if local.exists():
        with open(local, encoding="utf-8") as f:
            raw.update(yaml.safe_load(f) or {})
    return {key: (REPO_ROOT / value).resolve() for key, value in raw.items()}
