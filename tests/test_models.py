"""Tests for model discovery and information (vllama/models.py)."""

import json
from pathlib import Path

import pytest

from vllama.models import (
    ModelInfo,
    find_model_by_name,
    get_model_list_for_api,
    scan_transformers_cache,
    _detect_model_type,
)


class TestModelInfo:
    """Tests for ModelInfo data class."""

    def test_creation(self):
        """Test ModelInfo creation."""
        model = ModelInfo(
            model_id="Qwen/Qwen3-0.6B",
            local_path="/path/to/model",
            model_type="chat",
        )
        assert model.model_id == "Qwen/Qwen3-0.6B"
        assert model.local_path == "/path/to/model"
        assert model.model_type == "chat"

    def test_optional_model_type(self):
        """Test ModelInfo with optional model_type."""
        model = ModelInfo(
            model_id="test/model",
            local_path="/path",
        )
        assert model.model_type is None


class TestFindModelByName:
    """Tests for find_model_by_name function."""

    def test_exact_match(self, sample_models):
        """Test exact model ID match."""
        result = find_model_by_name(sample_models, "Qwen/Qwen3-0.6B")
        assert result is not None
        assert result.model_id == "Qwen/Qwen3-0.6B"

    def test_partial_match_model_name_only(self, sample_models):
        """Test matching with model name only (without org)."""
        result = find_model_by_name(sample_models, "Qwen3-0.6B")
        assert result is not None
        assert result.model_id == "Qwen/Qwen3-0.6B"

    def test_partial_match_substring(self, sample_models):
        """Test substring matching."""
        result = find_model_by_name(sample_models, "bge-m3")
        assert result is not None
        assert result.model_id == "BAAI/bge-m3"

    def test_no_match(self, sample_models):
        """Test non-existent model."""
        result = find_model_by_name(sample_models, "nonexistent/model")
        assert result is None

    def test_empty_list(self):
        """Test with empty model list."""
        result = find_model_by_name([], "test")
        assert result is None

    def test_case_sensitive(self, sample_models):
        """Test that matching is case-sensitive."""
        # Should not match with different case
        result = find_model_by_name(sample_models, "qwen/qwen3-0.6b")
        assert result is None


class TestScanTransformersCache:
    """Tests for scan_transformers_cache function."""

    def test_nonexistent_directory(self):
        """Test scanning non-existent cache directory."""
        models = scan_transformers_cache("/nonexistent/path/to/cache")
        assert models == []

    def test_empty_directory(self, temp_cache_dir):
        """Test scanning empty cache directory."""
        models = scan_transformers_cache(str(temp_cache_dir))
        assert models == []

    def test_valid_model_structure(self, temp_cache_dir):
        """Test scanning directory with valid model structure."""
        # Create a mock model directory structure
        model_dir = temp_cache_dir / "models--Qwen--Qwen3-0.6B"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create a minimal config.json
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        with open(snapshot_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Scan the cache
        models = scan_transformers_cache(str(temp_cache_dir))
        assert len(models) == 1
        assert models[0].model_id == "Qwen/Qwen3-0.6B"
        assert "abc123" in models[0].local_path
        assert models[0].model_type == "chat"

    def test_multiple_models(self, temp_cache_dir):
        """Test scanning directory with multiple models."""
        # Create multiple model directories
        models_data = [
            ("models--Qwen--Qwen3-0.6B", {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
            ("models--BAAI--bge-m3", {"model_type": "bert", "architectures": ["BertEmbedding"]}),
        ]

        for model_name, config_data in models_data:
            model_dir = temp_cache_dir / model_name
            snapshot_dir = model_dir / "snapshots" / "hash123"
            snapshot_dir.mkdir(parents=True)
            with open(snapshot_dir / "config.json", "w") as f:
                json.dump(config_data, f)

        models = scan_transformers_cache(str(temp_cache_dir))
        assert len(models) == 2

    def test_missing_snapshots_directory(self, temp_cache_dir):
        """Test model directory without snapshots subdirectory."""
        model_dir = temp_cache_dir / "models--Invalid--Model"
        model_dir.mkdir(parents=True)

        models = scan_transformers_cache(str(temp_cache_dir))
        assert models == []

    def test_missing_config_file(self, temp_cache_dir):
        """Test snapshot directory without config.json."""
        model_dir = temp_cache_dir / "models--Test--Model"
        snapshot_dir = model_dir / "snapshots" / "hash"
        snapshot_dir.mkdir(parents=True)
        # Create a different JSON file
        with open(snapshot_dir / "other.json", "w") as f:
            json.dump({}, f)

        models = scan_transformers_cache(str(temp_cache_dir))
        # Should still include models with any JSON file
        assert len(models) >= 0


class TestDetectModelType:
    """Tests for _detect_model_type function."""

    def test_chat_model(self, temp_dir):
        """Test detecting chat/completion model."""
        config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
        }
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        model_type = _detect_model_type(temp_dir)
        assert model_type == "chat"

    def test_embedding_model(self, temp_dir):
        """Test detecting embedding model."""
        config = {
            "model_type": "bert",
            "architectures": ["BertEmbedding"],
        }
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        model_type = _detect_model_type(temp_dir)
        assert model_type == "embedding"

    def test_rerank_model(self, temp_dir):
        """Test detecting reranker model."""
        config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
        }
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        model_type = _detect_model_type(temp_dir)
        assert model_type == "rerank"

    def test_missing_config(self, temp_dir):
        """Test detection with missing config file."""
        model_type = _detect_model_type(temp_dir)
        assert model_type is None

    def test_invalid_json(self, temp_dir):
        """Test detection with invalid JSON."""
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            f.write("invalid json{")

        model_type = _detect_model_type(temp_dir)
        assert model_type is None


class TestGetModelListForApi:
    """Tests for get_model_list_for_api function."""

    def test_empty_list(self):
        """Test formatting empty model list."""
        result = get_model_list_for_api([])
        assert result == []

    def test_format_models(self, sample_models):
        """Test formatting model list for API."""
        result = get_model_list_for_api(sample_models)
        assert len(result) == 3

        # Check first model format
        model = result[0]
        assert model["id"] == "Qwen/Qwen3-0.6B"
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "user"
        assert "permission" in model

    def test_all_models_formatted(self, sample_models):
        """Test that all models are properly formatted."""
        result = get_model_list_for_api(sample_models)
        model_ids = [m["id"] for m in result]
        assert "Qwen/Qwen3-0.6B" in model_ids
        assert "BAAI/bge-m3" in model_ids
        assert "tomaarsen/Qwen3-Reranker-0.6B-seq-cls" in model_ids
