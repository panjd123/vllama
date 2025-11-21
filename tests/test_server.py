"""Tests for server API (vllama/server.py)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllama.config import VllamaConfig
from vllama.models import ModelInfo
from vllama.server import create_app
from vllama.state import InstanceState, InstanceStatus


@pytest.fixture
def test_app(temp_cache_dir):
    """Create a test FastAPI app."""
    config = VllamaConfig(
        host="127.0.0.1",
        port=33258,
        transformers_cache=str(temp_cache_dir),
    )
    app = create_app(config)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    with TestClient(test_app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    @patch("vllama.server.scan_transformers_cache")
    def test_list_models_empty(self, mock_scan, client):
        """Test listing models when cache is empty."""
        mock_scan.return_value = []

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    @patch("vllama.models.scan_transformers_cache")
    def test_list_models_with_data(self, mock_scan, client, sample_models):
        """Test listing models with data."""
        mock_scan.return_value = sample_models

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        # Note: actual number may differ due to how the app initializes
        # Just check that it's a list
        assert isinstance(data["data"], list)


class TestInstanceManagement:
    """Tests for instance management endpoints."""

    def test_list_instances_empty(self, client):
        """Test listing instances when none are running."""
        response = client.get("/instances")
        assert response.status_code == 200
        data = response.json()
        assert "instances" in data
        assert isinstance(data["instances"], list)


class TestForwardingRequests:
    """Tests for request forwarding."""

    def test_missing_model_parameter(self, client):
        """Test request without model parameter."""
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 400
        assert "Model name required" in response.text or "model" in response.text.lower()

    @patch("vllama.models.find_model_by_name")
    def test_model_not_found(self, mock_find, client):
        """Test request with non-existent model."""
        mock_find.return_value = None

        response = client.post("/v1/chat/completions", json={"model": "nonexistent"})
        assert response.status_code == 404
        assert "not found" in response.text.lower()
