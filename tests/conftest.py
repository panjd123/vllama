"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from vllama.config import ModelConfig, VllamaConfig
from vllama.models import ModelInfo
from vllama.state import InstanceState, InstanceStatus


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_dir(temp_dir: Path) -> Path:
    """Provide a temporary config directory."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Provide a temporary transformers cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def vllama_config(temp_cache_dir: Path) -> VllamaConfig:
    """Provide a test VllamaConfig instance."""
    return VllamaConfig(
        host="127.0.0.1",
        port=33258,
        vllm_port_start=33300,
        vllm_port_end=34300,
        transformers_cache=str(temp_cache_dir),
        unload_timeout=300,
        unload_mode=2,
    )


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Provide a sample ModelConfig."""
    return ModelConfig(
        model_name="test/model",
        port=8080,
        gpu_memory_utilization=0.8,
        devices=[0],
        tensor_parallel_size=1,
        max_model_len=2048,
        trust_remote_code=False,
        dtype="auto",
        extra_args={},
    )


@pytest.fixture
def sample_models() -> list[ModelInfo]:
    """Provide sample model list for testing."""
    return [
        ModelInfo(
            model_id="Qwen/Qwen3-0.6B",
            local_path="/fake/path/qwen",
            model_type="chat",
        ),
        ModelInfo(
            model_id="BAAI/bge-m3",
            local_path="/fake/path/bge",
            model_type="embedding",
        ),
        ModelInfo(
            model_id="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
            local_path="/fake/path/reranker",
            model_type="rerank",
        ),
    ]


@pytest.fixture
def sample_instance_state() -> InstanceState:
    """Provide a sample InstanceState."""
    return InstanceState(
        model_id="test/model",
        port=8000,
        status=InstanceStatus.RUNNING,
        devices=[0],
        pid=12345,
        start_time=1234567890.0,
        last_request_time=1234567890.0,
    )


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env


@pytest.fixture
def state_file(temp_config_dir: Path) -> Path:
    """Provide a temporary state file path."""
    return temp_config_dir / "state.json"


@pytest.fixture
def yaml_config_file(temp_config_dir: Path) -> Path:
    """Provide a temporary YAML config file path."""
    return temp_config_dir / "models.yaml"


@pytest.fixture
def pid_file(temp_config_dir: Path) -> Path:
    """Provide a temporary PID file path."""
    return temp_config_dir / "server.pid"


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration for each test."""
    import logging
    # Clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Reset level
    logging.root.setLevel(logging.WARNING)
    yield
    # Cleanup after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
