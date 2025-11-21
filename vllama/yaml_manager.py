"""YAML configuration file management for model deployment."""

import logging
from pathlib import Path
from typing import Optional

import yaml

from vllama.config import ModelConfig

logger = logging.getLogger(__name__)


class YAMLConfigManager:
    """Manages YAML configuration files for model deployments.

    This class reads directly from the config file on every operation
    to ensure consistency with manual file edits.
    """

    def __init__(self, config_file: Path):
        """Initialize YAML config manager.

        Args:
            config_file: Path to models.yaml file
        """
        self.config_file = config_file

    def _read_yaml(self) -> dict:
        """Read and parse YAML config file.

        Returns:
            Dictionary of raw YAML data, empty dict if file doesn't exist
        """
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f) or {}
            return data
        except Exception as e:
            logger.error(f"Failed to read YAML config: {e}")
            return {}

    def _write_yaml(self, data: dict):
        """Write data to YAML config file.

        Args:
            data: Dictionary to write to YAML
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            logger.debug(f"Saved configurations to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to write YAML config: {e}")
            raise

    def get_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig if exists, None otherwise
        """
        data = self._read_yaml()
        config_data = data.get(model_id)

        if config_data is None:
            return None

        try:
            return ModelConfig(model_name=model_id, **config_data)
        except Exception as e:
            logger.error(f"Failed to parse config for {model_id}: {e}")
            return None

    def set_config(self, model_id: str, config: ModelConfig):
        """Set configuration for a model.

        Args:
            model_id: Model identifier
            config: Model configuration
        """
        # Read current configs
        data = self._read_yaml()

        # Update config for this model
        config.model_name = model_id
        config_dict = config.model_dump(exclude={"model_name"})
        # Remove None values for cleaner YAML
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        data[model_id] = config_dict

        # Write back
        self._write_yaml(data)
        logger.info(f"Updated configuration for {model_id}")

    def remove_config(self, model_id: str):
        """Remove configuration for a model.

        Args:
            model_id: Model identifier
        """
        # Read current configs
        data = self._read_yaml()

        if model_id in data:
            del data[model_id]
            self._write_yaml(data)
            logger.info(f"Removed configuration for {model_id}")

    def has_config(self, model_id: str) -> bool:
        """Check if configuration exists for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if config exists
        """
        data = self._read_yaml()
        return model_id in data

    def get_all_configs(self) -> dict[str, ModelConfig]:
        """Get all model configurations.

        Returns:
            Dictionary of model_id -> ModelConfig
        """
        data = self._read_yaml()
        configs = {}

        for model_id, config_data in data.items():
            try:
                configs[model_id] = ModelConfig(model_name=model_id, **config_data)
            except Exception as e:
                logger.error(f"Failed to parse config for {model_id}: {e}")

        return configs

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
        # Get existing config or create new one
        config = self.get_config(model_id)

        if config is None:
            # Create new config
            config = ModelConfig(model_name=model_id, **kwargs)
        else:
            # Update existing config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Save updated config
        self.set_config(model_id, config)
        return config
