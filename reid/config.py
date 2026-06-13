# src/config.py

from pathlib import Path
import yaml


def load_yaml_config(path: str) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, path: str, default=None):
    """
    Access nested config values using dot notation.

    Example:
        get_config_value(config, "model.backbone")
    """
    current = config

    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current
