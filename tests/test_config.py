"""
Unit tests for configuration loader and path utility functions.
"""

from pathlib import Path
import pytest
from src.utils.config import get_absolute_path, load_config


def test_load_config():
    config = load_config("configs/config.yaml")
    assert config.system.name == "AI-Based Cyber Threat Detection and Intelligence Generation"
    assert config.system.seed == 42
    assert config.data.test_size == 0.20
    assert config.fusion.default_alpha == 0.6


def test_config_dot_notation():
    config = load_config("configs/config.yaml")
    assert config.models.advanced.xgboost.max_depth == 8
    assert config.models.advanced.xgboost.n_estimators == 200


def test_get_absolute_path():
    abs_path = get_absolute_path("data/raw")
    assert isinstance(abs_path, Path)
    assert abs_path.is_absolute()
