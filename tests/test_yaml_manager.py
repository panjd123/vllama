"""Tests for YAML configuration management (vllama/yaml_manager.py)."""

import yaml
from pathlib import Path

import pytest

from vllama.config import ModelConfig
from vllama.yaml_manager import YAMLConfigManager


class TestYAMLConfigManager:
    """Tests for YAMLConfigManager."""

    def test_initialization_nonexistent_file(self, yaml_config_file):
        """Test initialization with non-existent config file."""
        manager = YAMLConfigManager(yaml_config_file)
        assert manager.config_file == yaml_config_file
        assert manager.configs == {}

    def test_initialization_with_existing_file(self, yaml_config_file):
        """Test initialization with existing config file."""
        # Create a config file
        config_data = {
            "test/model": {
                "port": 8080,
                "gpu_memory_utilization": 0.8,
                "devices": [0, 1],
                "tensor_parallel_size": 2,
            }
        }
        with open(yaml_config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        # Load it
        manager = YAMLConfigManager(yaml_config_file)
        assert len(manager.configs) == 1
        assert "test/model" in manager.configs

        config = manager.get_config("test/model")
        assert config.model_name == "test/model"
        assert config.port == 8080
        assert config.gpu_memory_utilization == 0.8

    def test_get_config_existing(self, yaml_config_file):
        """Test getting existing config."""
        manager = YAMLConfigManager(yaml_config_file)
        config = ModelConfig(model_name="test/model", port=8080)
        manager.set_config("test/model", config)

        retrieved = manager.get_config("test/model")
        assert retrieved is not None
        assert retrieved.model_name == "test/model"
        assert retrieved.port == 8080

    def test_get_config_nonexistent(self, yaml_config_file):
        """Test getting non-existent config."""
        manager = YAMLConfigManager(yaml_config_file)
        result = manager.get_config("nonexistent")
        assert result is None

    def test_set_config(self, yaml_config_file):
        """Test setting config."""
        manager = YAMLConfigManager(yaml_config_file)
        config = ModelConfig(
            model_name="test/model",
            port=8080,
            gpu_memory_utilization=0.8,
            devices=[0],
        )
        manager.set_config("test/model", config)

        # Verify it was set
        assert manager.has_config("test/model")
        retrieved = manager.get_config("test/model")
        assert retrieved.port == 8080

        # Verify file was written
        assert yaml_config_file.exists()

    def test_save_and_load_configs(self, yaml_config_file):
        """Test saving and loading configurations."""
        manager1 = YAMLConfigManager(yaml_config_file)

        # Add multiple configs
        manager1.set_config("model1", ModelConfig(model_name="model1", port=8000))
        manager1.set_config("model2", ModelConfig(model_name="model2", port=8001))

        # Load in new manager
        manager2 = YAMLConfigManager(yaml_config_file)
        assert len(manager2.configs) == 2
        assert manager2.has_config("model1")
        assert manager2.has_config("model2")

    def test_remove_config(self, yaml_config_file):
        """Test removing config."""
        manager = YAMLConfigManager(yaml_config_file)

        # Add a config
        manager.set_config("test/model", ModelConfig(model_name="test/model", port=8080))
        assert manager.has_config("test/model")

        # Remove it
        manager.remove_config("test/model")
        assert not manager.has_config("test/model")

        # Verify file was updated
        manager2 = YAMLConfigManager(yaml_config_file)
        assert not manager2.has_config("test/model")

    def test_remove_nonexistent_config(self, yaml_config_file):
        """Test removing non-existent config (should not error)."""
        manager = YAMLConfigManager(yaml_config_file)
        manager.remove_config("nonexistent")  # Should not raise

    def test_has_config(self, yaml_config_file):
        """Test checking if config exists."""
        manager = YAMLConfigManager(yaml_config_file)

        assert not manager.has_config("test/model")

        manager.set_config("test/model", ModelConfig(model_name="test/model"))
        assert manager.has_config("test/model")

    def test_get_all_configs(self, yaml_config_file):
        """Test getting all configs."""
        manager = YAMLConfigManager(yaml_config_file)

        # Add multiple configs
        manager.set_config("model1", ModelConfig(model_name="model1", port=8000))
        manager.set_config("model2", ModelConfig(model_name="model2", port=8001))
        manager.set_config("model3", ModelConfig(model_name="model3", port=8002))

        all_configs = manager.get_all_configs()
        assert len(all_configs) == 3
        assert "model1" in all_configs
        assert "model2" in all_configs
        assert "model3" in all_configs

    def test_update_config_existing(self, yaml_config_file):
        """Test updating existing config."""
        manager = YAMLConfigManager(yaml_config_file)

        # Create initial config
        manager.set_config("test/model", ModelConfig(
            model_name="test/model",
            port=8080,
            gpu_memory_utilization=0.7,
        ))

        # Update it
        updated = manager.update_config("test/model", port=9090, devices=[0, 1])

        assert updated.port == 9090
        assert updated.devices == [0, 1]
        assert updated.gpu_memory_utilization == 0.7  # Preserved

    def test_update_config_create_new(self, yaml_config_file):
        """Test updating non-existent config (should create new)."""
        manager = YAMLConfigManager(yaml_config_file)

        config = manager.update_config("new/model", port=8080, gpu_memory_utilization=0.8)

        assert config.model_name == "new/model"
        assert config.port == 8080
        assert config.gpu_memory_utilization == 0.8
        assert manager.has_config("new/model")

    def test_yaml_file_format(self, yaml_config_file):
        """Test that saved YAML file has correct format."""
        manager = YAMLConfigManager(yaml_config_file)
        manager.set_config("test/model", ModelConfig(
            model_name="test/model",
            port=8080,
            gpu_memory_utilization=0.8,
            devices=[0, 1],
            max_model_len=2048,
        ))

        # Load raw YAML
        with open(yaml_config_file, "r") as f:
            raw_data = yaml.safe_load(f)

        # Verify structure
        assert "test/model" in raw_data
        model_config = raw_data["test/model"]
        assert model_config["port"] == 8080
        assert model_config["gpu_memory_utilization"] == 0.8
        assert model_config["devices"] == [0, 1]
        # model_name should not be in the YAML (it's the key)
        assert "model_name" not in model_config

    def test_invalid_yaml_handling(self, yaml_config_file):
        """Test handling of invalid YAML file."""
        # Write invalid YAML
        with open(yaml_config_file, "w") as f:
            f.write("invalid: yaml: content: {")

        # Should not crash, just return empty configs
        manager = YAMLConfigManager(yaml_config_file)
        assert manager.configs == {}

    def test_partial_invalid_configs(self, yaml_config_file):
        """Test handling of partially invalid configs."""
        # Write YAML with one valid and one invalid config
        config_data = {
            "valid/model": {
                "port": 8080,
                "gpu_memory_utilization": 0.8,
            },
            "invalid/model": {
                "port": "not_a_number",  # Invalid port type
                "gpu_memory_utilization": 5.0,  # Out of range
            }
        }
        with open(yaml_config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        manager = YAMLConfigManager(yaml_config_file)
        # Valid config should be loaded
        assert manager.has_config("valid/model")
        # Invalid config should be skipped
        # (actual behavior depends on validation)

    def test_none_values_excluded(self, yaml_config_file):
        """Test that None values are excluded from saved YAML."""
        manager = YAMLConfigManager(yaml_config_file)
        config = ModelConfig(
            model_name="test/model",
            port=None,  # None value
            gpu_memory_utilization=0.8,
            devices=None,  # None value
        )
        manager.set_config("test/model", config)

        # Load raw YAML
        with open(yaml_config_file, "r") as f:
            raw_data = yaml.safe_load(f)

        model_config = raw_data["test/model"]
        # None values should not be present
        assert "port" not in model_config
        assert "devices" not in model_config
        # Non-None values should be present
        assert "gpu_memory_utilization" in model_config

    def test_directory_creation(self, temp_dir):
        """Test that parent directory is created if it doesn't exist."""
        nested_config_file = temp_dir / "nested" / "dir" / "models.yaml"

        manager = YAMLConfigManager(nested_config_file)
        manager.set_config("test/model", ModelConfig(model_name="test/model"))

        # Verify directory was created
        assert nested_config_file.parent.exists()
        assert nested_config_file.exists()
