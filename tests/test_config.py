"""Tests for loading and validating project configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from teleassist.config import AppConfig, load_config


def test_load_default_config() -> None:
    """The checked-in YAML supplies project defaults."""
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")

    assert config.seed == 42
    assert config.max_doc_share_for_sampling == 0.15
    assert config.chunking.chunk_size == 512
    assert config.models.generator == "Qwen/Qwen2.5-1.5B-Instruct"


def test_chunk_size_must_be_positive() -> None:
    """Invalid chunk sizes fail at config validation time."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"chunking": {"chunk_size": 0}})


def test_max_doc_share_for_sampling_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"max_doc_share_for_sampling": 0})
