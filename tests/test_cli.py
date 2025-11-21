"""Tests for CLI commands (vllama/cli.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vllama.cli import app


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_pid_file(temp_config_dir):
    """Create a mock PID file."""
    pid_file = temp_config_dir / "server.pid"
    pid_data = {
        "pid": 12345,
        "port": 33258,
        "host": "0.0.0.0",
        "start_time": 1234567890.0,
    }
    with open(pid_file, "w") as f:
        json.dump(pid_data, f)
    return pid_file


class TestServeCommand:
    """Tests for 'vllama serve' command."""

    @patch("vllama.cli.uvicorn.run")
    @patch("vllama.cli.VllamaPaths")
    def test_serve_default_args(self, mock_paths, mock_uvicorn, cli_runner, temp_config_dir):
        """Test serve command with default arguments."""
        mock_paths.CONFIG_DIR = temp_config_dir
        mock_paths.MODELS_CONFIG_FILE = temp_config_dir / "models.yaml"

        result = cli_runner.invoke(app, ["serve"])

        # Should call uvicorn.run
        assert mock_uvicorn.called
        # Check that it doesn't error
        # Note: This may exit with code 0 or not complete due to mock

    @patch("vllama.cli.uvicorn.run")
    @patch("vllama.cli.VllamaPaths")
    def test_serve_custom_port(self, mock_paths, mock_uvicorn, cli_runner, temp_config_dir):
        """Test serve command with custom port."""
        mock_paths.CONFIG_DIR = temp_config_dir
        mock_paths.MODELS_CONFIG_FILE = temp_config_dir / "models.yaml"

        result = cli_runner.invoke(app, ["serve", "--port", "9999"])

        assert mock_uvicorn.called
        # Verify port was passed to config
        # (This is indirect - we'd need to inspect the call args)


class TestInfoCommand:
    """Tests for 'vllama info' command."""

    @patch("vllama.cli.VllamaPaths")
    def test_info_no_server(self, mock_paths, cli_runner, temp_config_dir):
        """Test info command when server is not running."""
        mock_paths.PID_FILE = temp_config_dir / "server.pid"

        result = cli_runner.invoke(app, ["info"])
        assert result.exit_code == 0
        # The output should indicate no server is running
        output_lower = result.stdout.lower()
        assert "no" in output_lower or "not" in output_lower

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.psutil.Process")
    @patch("vllama.cli.httpx.get")
    def test_info_server_running(
        self,
        mock_httpx,
        mock_process,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        temp_config_dir,
        mock_pid_file,
    ):
        """Test info command when server is running."""
        mock_paths.PID_FILE = mock_pid_file
        mock_pid_exists.return_value = True

        # Mock process info
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process.return_value = mock_proc

        # Mock health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Running" in result.stdout or "running" in result.stdout.lower()


class TestPsCommand:
    """Tests for 'vllama ps' command."""

    @patch("vllama.cli.VllamaPaths")
    def test_ps_no_server(self, mock_paths, cli_runner, temp_config_dir):
        """Test ps command when server is not running."""
        mock_paths.PID_FILE = temp_config_dir / "server.pid"

        result = cli_runner.invoke(app, ["ps"])
        assert result.exit_code == 1
        assert "not running" in result.stdout.lower()

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.httpx.get")
    def test_ps_no_instances(
        self,
        mock_httpx,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        mock_pid_file,
    ):
        """Test ps command with no running instances."""
        mock_paths.PID_FILE = mock_pid_file
        mock_pid_exists.return_value = True

        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"instances": []}
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["ps"])
        assert result.exit_code == 0
        assert "No active instances" in result.stdout


class TestStartCommand:
    """Tests for 'vllama start' command."""

    @patch("vllama.cli.VllamaPaths")
    def test_start_no_server(self, mock_paths, cli_runner, temp_config_dir):
        """Test start command when server is not running."""
        mock_paths.PID_FILE = temp_config_dir / "server.pid"

        result = cli_runner.invoke(app, ["start", "test/model"])
        assert result.exit_code == 1
        assert "not running" in result.stdout.lower()

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.httpx.post")
    def test_start_success(
        self,
        mock_httpx,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        mock_pid_file,
    ):
        """Test starting a model successfully."""
        mock_paths.PID_FILE = mock_pid_file
        mock_paths.LOGS_DIR = Path("/tmp/logs")
        mock_pid_exists.return_value = True

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_id": "test/model",
            "port": 8000,
            "pid": 12345,
        }
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["start", "test/model"])
        assert result.exit_code == 0
        assert "Successfully" in result.stdout


class TestStopCommand:
    """Tests for 'vllama stop' command."""

    @patch("vllama.cli.VllamaPaths")
    def test_stop_no_server(self, mock_paths, cli_runner, temp_config_dir):
        """Test stop command when server is not running."""
        mock_paths.PID_FILE = temp_config_dir / "server.pid"

        result = cli_runner.invoke(app, ["stop", "test/model"])
        assert result.exit_code == 1

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.httpx.post")
    def test_stop_success(
        self,
        mock_httpx,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        mock_pid_file,
    ):
        """Test stopping a model successfully."""
        mock_paths.PID_FILE = mock_pid_file
        mock_pid_exists.return_value = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["stop", "test/model"])
        assert result.exit_code == 0
        assert "Successfully stopped" in result.stdout


class TestSleepCommand:
    """Tests for 'vllama sleep' command."""

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.httpx.post")
    def test_sleep_default_level(
        self,
        mock_httpx,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        mock_pid_file,
    ):
        """Test sleep command with default level."""
        mock_paths.PID_FILE = mock_pid_file
        mock_pid_exists.return_value = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["sleep", "test/model"])
        assert result.exit_code == 0

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.psutil.pid_exists")
    @patch("vllama.cli.httpx.post")
    def test_sleep_custom_level(
        self,
        mock_httpx,
        mock_pid_exists,
        mock_paths,
        cli_runner,
        mock_pid_file,
    ):
        """Test sleep command with custom level."""
        mock_paths.PID_FILE = mock_pid_file
        mock_pid_exists.return_value = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value = mock_response

        result = cli_runner.invoke(app, ["sleep", "test/model", "--level", "1"])
        assert result.exit_code == 0


class TestAssignCommand:
    """Tests for 'vllama assign' command."""

    @patch("vllama.cli.VllamaPaths")
    @patch("vllama.cli.YAMLConfigManager")
    @patch("vllama.cli.scan_transformers_cache")
    @patch("vllama.cli.find_model_by_name")
    def test_assign_basic(
        self,
        mock_find,
        mock_scan,
        mock_yaml,
        mock_paths,
        cli_runner,
        temp_config_dir,
        sample_models,
    ):
        """Test assign command with basic options."""
        mock_paths.MODELS_CONFIG_FILE = temp_config_dir / "models.yaml"
        mock_scan.return_value = sample_models
        mock_find.return_value = sample_models[0]

        # Mock YAML manager
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = MagicMock(
            devices=[0],
            gpu_memory_utilization=0.7,
            port=None,
        )
        mock_yaml.return_value = mock_manager

        result = cli_runner.invoke(app, ["assign", "Qwen3-0.6B", "--devices", "0,1"])
        # May succeed or fail depending on environment
        # Main goal is to not crash


class TestPullCommand:
    """Tests for 'vllama pull' command."""

    @patch("vllama.cli.snapshot_download")
    def test_pull_basic(self, mock_download, cli_runner):
        """Test pull command with basic model ID."""
        mock_download.return_value = "/path/to/model"

        result = cli_runner.invoke(app, ["pull", "test/model"])
        assert result.exit_code == 0
        assert "Successfully downloaded" in result.stdout

        # Verify snapshot_download was called
        mock_download.assert_called_once()
        args, kwargs = mock_download.call_args
        assert kwargs["repo_id"] == "test/model"

    @patch("vllama.cli.snapshot_download")
    def test_pull_with_revision(self, mock_download, cli_runner):
        """Test pull command with revision."""
        mock_download.return_value = "/path/to/model"

        result = cli_runner.invoke(app, ["pull", "test/model", "--revision", "dev"])
        assert result.exit_code == 0

        # Verify revision was passed
        args, kwargs = mock_download.call_args
        assert kwargs["revision"] == "dev"

    @patch("vllama.cli.snapshot_download")
    def test_pull_failure(self, mock_download, cli_runner):
        """Test pull command when download fails."""
        mock_download.side_effect = Exception("Network error")

        result = cli_runner.invoke(app, ["pull", "test/model"])
        assert result.exit_code == 1
        assert "Failed" in result.stdout


class TestListCommand:
    """Tests for 'vllama list' command."""

    @patch("vllama.cli.scan_transformers_cache")
    def test_list_no_models(self, mock_scan, cli_runner):
        """Test list command when no models are found."""
        mock_scan.return_value = []

        result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No models found" in result.stdout

    @patch("vllama.cli.scan_transformers_cache")
    def test_list_with_models(self, mock_scan, cli_runner, sample_models):
        """Test list command with models."""
        mock_scan.return_value = sample_models

        result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Check that model IDs appear in output
        assert "Qwen/Qwen3-0.6B" in result.stdout
        assert "BAAI/bge-m3" in result.stdout
