"""Model discovery and information management."""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """Information about a cached model."""

    model_id: str  # e.g., "Qwen/Qwen3-0.6B"
    local_path: str  # Full path to model directory
    model_type: Optional[str] = None  # "chat", "embedding", "rerank", etc.


def scan_transformers_cache(cache_dir: str) -> list[ModelInfo]:
    """Scan HF_HOME/hub directory for available models.

    The HuggingFace cache structure is:
    {HF_HOME}/hub/models--{org}--{model}/snapshots/{hash}/

    Args:
        cache_dir: Path to HF_HOME/hub directory

    Returns:
        List of ModelInfo objects
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []

    models = []

    # Look for directories matching the pattern models--*
    for model_dir in cache_path.glob("models--*"):
        if not model_dir.is_dir():
            continue

        # Parse model ID from directory name
        # Format: models--{org}--{model} or models--{model}
        parts = model_dir.name.split("--")[1:]  # Remove "models" prefix
        if not parts:
            continue

        model_id = "/".join(parts)

        # Find the actual model files in snapshots
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        # Get the latest snapshot (there should typically be one)
        snapshot_dirs = list(snapshots_dir.iterdir())
        if not snapshot_dirs:
            continue

        # Use the first (or only) snapshot directory
        latest_snapshot = snapshot_dirs[0]

        # Verify it's a valid model directory by checking for config files
        if not (latest_snapshot / "config.json").exists():
            # Try to check if it's a valid model dir in other ways
            if not any(latest_snapshot.glob("*.json")):
                continue

        # Determine model type
        if "embedding" in model_id.lower():
            model_type = "embedding"
        elif "rerank" in model_id.lower() or "ranker" in model_id.lower():
            model_type = "rerank"
        elif "instruct" in model_id.lower() or "chat" in model_id.lower():
            model_type = "chat"
        else:
            model_type = _detect_model_type(latest_snapshot)

        models.append(ModelInfo(
            model_id=model_id,
            local_path=str(latest_snapshot),
            model_type=model_type
        ))

        logger.debug(f"Found model: {model_id} at {latest_snapshot}")

    logger.info(f"Scanned {len(models)} models from cache")
    return models


def _detect_model_type(model_path: Path) -> Optional[str]:
    """Detect the type of model based on its configuration.

    Args:
        model_path: Path to model snapshot directory

    Returns:
        Model type string: "chat", "embedding", "rerank", or None
    """
    config_file = model_path / "config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            config = json.load(f)

        architectures = config.get("architectures", [])
        model_type_str = config.get("model_type", "").lower()

        # Check for embedding models
        if any("Embedding" in arch for arch in architectures):
            return "embedding"

        # Check for reranker models
        if any("Rerank" in arch or "SequenceClassification" in arch for arch in architectures):
            return "rerank"

        # Check for encoder-only models (typically embedding)
        if "bert" in model_type_str and "ForMaskedLM" not in str(architectures):
            if any("SequenceClassification" in arch for arch in architectures):
                return "rerank"
            return "embedding"
        
        # Default to chat/completion model
        return "chat"

    except Exception as e:
        logger.warning(f"Failed to detect model type for {model_path}: {e}")
        return None


def find_model_by_name(
    models: list[ModelInfo],
    name: str
) -> Optional[ModelInfo]:
    """Find a model by its name or ID.

    Supports partial matching:
    - Full ID: "Qwen/Qwen3-0.6B"
    - Org/model: "Qwen/Qwen3-0.6B"
    - Model name only: "Qwen3-0.6B"

    Args:
        models: List of available models
        name: Model name to search for

    Returns:
        ModelInfo if found, None otherwise
    """
    # Exact match first
    for model in models:
        if model.model_id == name:
            return model

    # Try matching just the model name (without org)
    for model in models:
        if model.model_id.split("/")[-1] == name:
            return model

    # Try partial match
    for model in models:
        if name in model.model_id:
            return model

    return None


def get_model_list_for_api(models: list[ModelInfo]) -> list[dict]:
    """Format model list for OpenAI-compatible API response.

    Args:
        models: List of ModelInfo objects

    Returns:
        List of model objects in OpenAI format
    """
    result = []
    for model in models:
        result.append({
            "id": model.model_id,
            "object": "model",
            "created": 0,  # Could use actual creation time if needed
            "owned_by": "user",
            "permission": [],
            "root": model.model_id,
            "parent": None,
        })
    return result
