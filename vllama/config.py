"""Configuration models for vllama.

This module contains runtime server configuration only.
Fixed paths are defined in vllama.constants.VllamaPaths.
"""

import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for a single VLLM model instance."""

    model_name: str
    port: Optional[int] = None  # Auto-assign if not specified
    gpu_memory_utilization: float = Field(default=0.7, ge=0.1, le=1.0)
    devices: Optional[list[int]] = None  # GPU device IDs, None for auto
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = 2048
    trust_remote_code: bool = False
    dtype: str = "auto"
    extra_args: dict[str, str] = Field(default_factory=dict)


class VllamaConfig(BaseSettings):
    """Runtime configuration for vllama server.

    This class contains ONLY runtime server settings.
    Fixed paths (config_dir, PID file, etc.) are in vllama.constants.VllamaPaths.

    Configurable via environment variables:
        VLLAMA_HOST: Server host (default: 0.0.0.0)
        VLLAMA_PORT: Server port (default: 33258)
        VLLAMA_VLLM_PORT_START: VLLM instance port range start (default: 33300)
        VLLAMA_VLLM_PORT_END: VLLM instance port range end (default: 34300)
        VLLAMA_TRANSFORMERS_CACHE: Model cache directory
        VLLAMA_UNLOAD_TIMEOUT: Seconds of inactivity before unloading (default: 1800)
        VLLAMA_UNLOAD_MODE: Unload mode 1/2/3 (default: 2)
    """

    model_config = SettingsConfigDict(
        env_prefix="VLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings (configurable via environment variables)
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=33258, description="Server port")

    # VLLM instance settings (configurable via environment variables)
    vllm_port_start: int = Field(default=33300, description="VLLM instance port range start")
    vllm_port_end: int = Field(default=34300, description="VLLM instance port range end")

    # Model cache directory (configurable via environment variables)
    transformers_cache: Optional[str] = Field(default=None, description="Model cache directory")

    # Unload settings (configurable via environment variables)
    unload_timeout: int = Field(default=1800, description="Seconds of inactivity before unloading")
    unload_mode: Literal[1, 2, 3] = Field(default=2, description="1=sleep level1, 2=sleep level2, 3=stop instance")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Use TRANSFORMERS_CACHE as fallback if VLLAMA_TRANSFORMERS_CACHE not set
        if self.transformers_cache is None:
            self.transformers_cache = os.environ.get(
                "TRANSFORMERS_CACHE",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            )

    def get_next_available_port(self, used_ports: set[int]) -> int:
        """Get next available port for VLLM instance."""
        for port in range(self.vllm_port_start, self.vllm_port_end):
            if port not in used_ports:
                return port
        raise RuntimeError(f"No available ports in range {self.vllm_port_start}-{self.vllm_port_end}")
