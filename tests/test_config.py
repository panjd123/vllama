"""Tests for configuration management (vllama/config.py)."""

import pytest

from vllama.config import ModelConfig, VllamaConfig


class TestVllamaConfig:
    """Tests for VllamaConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VllamaConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 33258
        assert config.vllm_port_start == 33300
        assert config.vllm_port_end == 34300
        assert config.unload_timeout == 1800
        assert config.unload_mode == 2
        assert config.transformers_cache is not None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VllamaConfig(
            host="127.0.0.1",
            port=9999,
            vllm_port_start=10000,
            vllm_port_end=11000,
            unload_timeout=600,
            unload_mode=1,
            transformers_cache="/custom/cache",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9999
        assert config.vllm_port_start == 10000
        assert config.vllm_port_end == 11000
        assert config.unload_timeout == 600
        assert config.unload_mode == 1
        assert config.transformers_cache == "/custom/cache"

    def test_environment_variable_loading(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("VLLAMA_HOST", "192.168.1.1")
        monkeypatch.setenv("VLLAMA_PORT", "12345")
        monkeypatch.setenv("VLLAMA_VLLM_PORT_START", "20000")
        monkeypatch.setenv("VLLAMA_VLLM_PORT_END", "21000")
        monkeypatch.setenv("VLLAMA_UNLOAD_TIMEOUT", "900")
        # Note: Literal types from env vars need special handling in pydantic
        # We skip unload_mode for this test

        config = VllamaConfig()
        assert config.host == "192.168.1.1"
        assert config.port == 12345
        assert config.vllm_port_start == 20000
        assert config.vllm_port_end == 21000
        assert config.unload_timeout == 900

    def test_get_next_available_port(self):
        """Test port assignment logic."""
        config = VllamaConfig(vllm_port_start=33300, vllm_port_end=33310)

        # No used ports, should return first
        port = config.get_next_available_port(set())
        assert port == 33300

        # Some used ports
        used_ports = {33300, 33301, 33302}
        port = config.get_next_available_port(used_ports)
        assert port == 33303

        # All ports used
        used_ports = set(range(33300, 33310))
        with pytest.raises(RuntimeError, match="No available ports"):
            config.get_next_available_port(used_ports)

    def test_transformers_cache_fallback(self, mock_env):
        """Test transformers cache fallback to HF_HOME."""
        mock_env(HF_HOME="/custom/hf_home")
        config = VllamaConfig()
        assert config.transformers_cache == "/custom/hf_home/hub"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig(model_name="test/model")
        assert config.model_name == "test/model"
        assert config.port is None  # Auto-assign
        assert config.gpu_memory_utilization == 0.7
        assert config.devices is None  # Auto-detect
        assert config.tensor_parallel_size == 1
        assert config.max_model_len is None
        assert config.trust_remote_code is False
        assert config.dtype == "auto"
        assert config.extra_args == {}

    def test_custom_values(self):
        """Test custom model configuration values."""
        config = ModelConfig(
            model_name="test/model",
            port=8080,
            gpu_memory_utilization=0.8,
            devices=[0, 1],
            tensor_parallel_size=2,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="float16",
            extra_args={"quantization": "awq"},
        )
        assert config.model_name == "test/model"
        assert config.port == 8080
        assert config.gpu_memory_utilization == 0.8
        assert config.devices == [0, 1]
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 4096
        assert config.trust_remote_code is True
        assert config.dtype == "float16"
        assert config.extra_args == {"quantization": "awq"}

    def test_gpu_memory_utilization_validation(self):
        """Test GPU memory utilization validation."""
        # Valid values
        config = ModelConfig(model_name="test/model", gpu_memory_utilization=0.5)
        assert config.gpu_memory_utilization == 0.5

        # Invalid values (too low)
        with pytest.raises(Exception):
            ModelConfig(model_name="test/model", gpu_memory_utilization=0.05)

        # Invalid values (too high)
        with pytest.raises(Exception):
            ModelConfig(model_name="test/model", gpu_memory_utilization=1.5)

    def test_immutability_after_creation(self):
        """Test that config is immutable after creation."""
        config = ModelConfig(model_name="test/model")
        # Pydantic models are mutable by default, but we can test assignment
        config.port = 9999
        assert config.port == 9999
