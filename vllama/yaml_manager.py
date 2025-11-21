"""YAML configuration file management for model deployment."""

import logging
from pathlib import Path
from typing import Optional

import yaml

from vllama.config import ModelConfig

logger = logging.getLogger(__name__)


class YAMLConfigManager:
    """Manages YAML configuration files for model deployments."""

    def __init__(self, config_file: Path):
        """Initialize YAML config manager.

        Args:
            config_file: Path to models.yaml file
        """
        self.config_file = config_file
        self.configs: dict[str, ModelConfig] = self._load_configs()

    def _load_configs(self) -> dict[str, ModelConfig]:
        """Load model configurations from YAML file.

        Returns:
            Dictionary mapping model_id to ModelConfig
        """
        if not self.config_file.exists():
            logger.info(f"Config file does not exist: {self.config_file}")
            return {}

        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f) or {}

            configs = {}
            for model_id, config_data in data.items():
                try:
                    configs[model_id] = ModelConfig(model_name=model_id, **config_data)
                except Exception as e:
                    logger.error(f"Failed to parse config for {model_id}: {e}")

            logger.info(f"Loaded {len(configs)} model configurations")
            return configs

        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            return {}

    def save_configs(self):
        """Save current configurations to YAML file."""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert configs to dict format
            data = {}
            for model_id, config in self.configs.items():
                # Convert ModelConfig to dict, excluding model_name (it's the key)
                config_dict = config.model_dump(exclude={"model_name"})
                # Remove None values for cleaner YAML
                config_dict = {k: v for k, v in config_dict.items() if v is not None}
                data[model_id] = config_dict

            with open(self.config_file, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved configurations to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save YAML config: {e}")

    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig if exists, None otherwise
        """
        return self.configs.get(model_id)

    def set_config(self, model_id: str, config: ModelConfig):
        """Set configuration for a model.

        Args:
            model_id: Model identifier
            config: Model configuration
        """
        config.model_name = model_id
        self.configs[model_id] = config
        self.save_configs()

    def remove_config(self, model_id: str):
        """Remove configuration for a model.

        Args:
            model_id: Model identifier
        """
        if model_id in self.configs:
            del self.configs[model_id]
            self.save_configs()

    def has_config(self, model_id: str) -> bool:
        """Check if configuration exists for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if config exists
        """
        return model_id in self.configs

    def get_all_configs(self) -> dict[str, ModelConfig]:
        """Get all model configurations.

        Returns:
            Dictionary of model_id -> ModelConfig
        """
        return self.configs

    def update_config(
        self,
        model_id: str,
        **kwargs
    ) -> ModelConfig:
        """Update or create configuration for a model.

        Args:
            model_id: Model identifier
            **kwargs: Configuration parameters to update

        Returns:
            Updated ModelConfig
        """
        if model_id in self.configs:
            # Update existing config
            config = self.configs[model_id]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            # Create new config
            config = ModelConfig(model_name=model_id, **kwargs)

        self.set_config(model_id, config)
        return config
