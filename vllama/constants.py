"""Global constants for vllama.

These paths are fixed and cannot be configured via environment variables.
They ensure consistency between CLI and server.
"""

from pathlib import Path


class VllamaPaths:
    """Fixed paths used by vllama CLI and server.

    These paths are NOT configurable to ensure CLI and server
    always use the same configuration directory.
    """

    # Base configuration directory
    CONFIG_DIR = Path.home() / ".vllama"

    # Server PID file - stores server port and metadata
    PID_FILE = CONFIG_DIR / "server.pid"

    # Models configuration file
    MODELS_CONFIG_FILE = CONFIG_DIR / "models.yaml"

    # State file (currently unused, kept for future use)
    STATE_FILE = CONFIG_DIR / "state.json"

    # Logs directory
    LOGS_DIR = CONFIG_DIR / "logs"

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
