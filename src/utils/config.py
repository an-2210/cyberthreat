"""
Configuration loader and manager for Cyber Threat Detection framework.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot notation access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[key] = value


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> ConfigDict:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ConfigDict containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return ConfigDict(config_data)


def get_absolute_path(relative_path: Union[str, Path], base_dir: Union[str, Path] = ".") -> Path:
    """Resolves a relative path against a base directory into an absolute Path object.

    Args:
        relative_path: Path string or Path object relative to project root.
        base_dir: Base directory for resolution.

    Returns:
        Absolute Path object.
    """
    base = Path(base_dir).resolve()
    target = Path(relative_path)
    if target.is_absolute():
        return target
    return (base / target).resolve()
