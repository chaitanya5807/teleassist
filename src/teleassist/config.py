"""Typed configuration loading for TeleAssist YAML files."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Project data and artifact locations, relative to the project root."""

    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    train_data: Path = Path("data/train")
    eval_data: Path = Path("data/eval")
    models: Path = Path("models")
    results: Path = Path("results")


class ChunkingConfig(BaseModel):
    """Text chunking parameters."""

    chunk_size: int = Field(default=512, gt=0)
    overlap: int = Field(default=64, ge=0)


class RetrievalConfig(BaseModel):
    """Retrieval candidate sizes and fusion weights."""

    top_k: int = Field(default=5, gt=0)
    candidate_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_weight: float = Field(default=1.0, ge=0)


class ModelConfig(BaseModel):
    """Hugging Face model identifiers."""

    embedding: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator: str = "Qwen/Qwen2.5-1.5B-Instruct"


class AppConfig(BaseModel):
    """Validated application configuration."""

    seed: int = 42
    include_drafts: bool = False
    max_doc_share_for_sampling: float = Field(default=0.15, gt=0, le=1)
    paths: PathConfig = Field(default_factory=PathConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an application configuration from YAML."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file) or {}
    return AppConfig.model_validate(raw_config)
