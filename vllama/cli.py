"""Command-line interface for vllama."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import psutil
import typer
import uvicorn
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.table import Table

from vllama.config import ModelConfig, VllamaConfig
from vllama.constants import VllamaPaths
from vllama.models import find_model_by_name, scan_transformers_cache
from vllama.server import create_app
from vllama.yaml_manager import YAMLConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(name="vllama", help="A vllm management tool with Ollama-like interface")
console = Console()


def get_transformers_cache() -> str:
    """Get the transformers cache directory.

    Returns the path following this priority:
    1. HF_HOME/hub if HF_HOME is set
    2. ~/.cache/huggingface/hub as default

    Returns:
        Path to transformers cache directory
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def get_server_port() -> Optional[int]:
    """Get server port from PID file.

    Returns:
        Server port if server is running, None otherwise
    """
    if not VllamaPaths.PID_FILE.exists():
        return None

    try:
        with open(VllamaPaths.PID_FILE, "r") as f:
            pid_data = json.load(f)

        pid = pid_data.get("pid")
        if not pid or not psutil.pid_exists(pid):
            return None

        return pid_data.get("port")
    except Exception:
        return None


def check_server_running() -> tuple[bool, Optional[int]]:
    """Check if server is running and return its port.

    Returns:
        Tuple of (is_running, port)
    """
    port = get_server_port()
    return (port is not None, port)


def ensure_server_running() -> int:
    """Ensure server is running and return its port.

    Raises:
        typer.Exit: If server is not running
    """
    is_running, port = check_server_running()
    if not is_running:
        console.print("[red]Error: vllama server is not running[/red]")
        console.print("\nStart the server with:")
        console.print("  vllama serve")
        raise typer.Exit(1)
    return port


def format_time_ago(timestamp: float) -> str:
    """Format a timestamp as human-readable time ago.

    Args:
        timestamp: Unix timestamp

    Returns:
        Human-readable string like "2m 30s" or "1h 5m"
    """
    now = datetime.now().timestamp()
    delta = now - timestamp

    if delta < 0:
        return "just now"

    # Calculate time units
    days = int(delta // 86400)
    hours = int((delta % 86400) // 3600)
    minutes = int((delta % 3600) // 60)
    seconds = int(delta % 60)

    # Format output
    if days > 0:
        if hours > 0:
            return f"{days}d {hours}h"
        return f"{days}d"
    elif hours > 0:
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
    elif minutes > 0:
        if seconds > 0:
            return f"{minutes}m {seconds}s"
        return f"{minutes}m"
    else:
        return f"{seconds}s"


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Host to bind to (default: 0.0.0.0)"),
    port: Optional[int] = typer.Option(None, "--port", help="Port to bind to (default: 33258)"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
):
    """Start vllama server.

    Configuration priority (highest to lowest):
    1. Command line arguments (--host, --port)
    2. Environment variables (VLLAMA_HOST, VLLAMA_PORT, etc.)
    3. Default values
    """
    # Build config kwargs only from explicitly provided arguments
    config_kwargs = {}
    if host is not None:
        config_kwargs["host"] = host
    if port is not None:
        config_kwargs["port"] = port

    # Create config - will use env vars for unspecified values
    config = VllamaConfig(**config_kwargs)

    console.print(f"[green]Starting vllama server on {config.host}:{config.port}[/green]")
    console.print(f"[cyan]Config directory: {VllamaPaths.CONFIG_DIR}[/cyan]")
    console.print(f"[cyan]Models cache: {config.transformers_cache}[/cyan]")

    # Create FastAPI app
    fastapi_app = create_app(config)

    # Run with uvicorn
    uvicorn.run(
        fastapi_app,
        host=config.host,
        port=config.port,
        log_level=log_level,
    )


@app.command()
def sleep(
    model: str = typer.Argument(..., help="Model name to sleep"),
    level: int = typer.Option(2, "--level", "-l", help="Sleep level (1, 2, or 3)"),
):
    """Put a model instance to sleep.

    Sleep levels:
      1: Light sleep - quick wake up
      2: Deep sleep - releases more memory
      3: Full stop - completely terminate instance
    """
    server_port = ensure_server_running()

    console.print(f"Sleeping model {model} (level {level})...")

    try:
        response = httpx.post(
            f"http://localhost:{server_port}/instances/{model}/sleep?level={level}",
            timeout=30.0
        )

        if response.status_code == 200:
            console.print(f"[green]Successfully put {model} to sleep (level {level})[/green]")
        elif response.status_code == 404:
            data = response.json()
            console.print(f"[red]Error: {data.get('detail', 'Model not found')}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Failed to sleep model: {response.text}[/red]")
            raise typer.Exit(1)

    except httpx.TimeoutException:
        console.print("[red]Failed to sleep model: timed out[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Failed to connect to vllama server[/red]")
        raise typer.Exit(1)


@app.command(name="wake-up")
@app.command(name="wakeup")
@app.command(name="wake_up")
@app.command(name="start")
def start(model: str = typer.Argument(..., help="Model name to start/wake up")):
    """Start or wake up a model instance."""
    server_port = ensure_server_running()

    console.print(f"Starting/waking up model {model}...")

    try:
        response = httpx.post(
            f"http://localhost:{server_port}/instances/{model}/start",
            timeout=300.0
        )

        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]Successfully started/woke up {data['model_id']}[/green]")
            console.print(f"Port: {data['port']}")
            console.print(f"PID: {data['pid']}")
            console.print(f"Logs: {VllamaPaths.LOGS_DIR}/{data['model_id'].replace('/', '_')}_{data['port']}.log")
        elif response.status_code == 404:
            data = response.json()
            console.print(f"[red]Error: {data.get('detail', 'Model not found')}[/red]")
            raise typer.Exit(1)
        else:
            data = response.json()
            console.print(f"[red]Failed to start/wake up model: {data.get('detail', response.text)}[/red]")
            raise typer.Exit(1)

    except httpx.TimeoutException:
        console.print("[red]Failed to start/wake up model: timed out[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Failed to connect to vllama server[/red]")
        raise typer.Exit(1)


@app.command()
def stop(model: str = typer.Argument(..., help="Model name to stop")):
    """Stop a running model instance."""
    server_port = ensure_server_running()

    console.print(f"Stopping model {model}...")

    try:
        response = httpx.post(
            f"http://localhost:{server_port}/instances/{model}/stop",
            timeout=60.0
        )

        if response.status_code == 200:
            console.print(f"[green]Successfully stopped {model}[/green]")
        elif response.status_code == 404:
            data = response.json()
            console.print(f"[red]Error: {data.get('detail', 'Model not found')}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Failed to stop model: {response.text}[/red]")
            raise typer.Exit(1)

    except httpx.TimeoutException:
        console.print("[red]Failed to stop model: timed out[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Failed to connect to vllama server[/red]")
        raise typer.Exit(1)


@app.command()
def restart(model: str = typer.Argument(..., help="Model name to restart")):
    """Restart a model instance to apply configuration changes.

    This command stops the model and starts it again with the latest configuration.
    Useful after manually editing ~/.vllama/models.yaml.

    Examples:
        vllama restart Qwen/Qwen3-0.6B
    """
    server_port = ensure_server_running()

    console.print(f"Restarting model {model}...")

    try:
        # First stop the model
        console.print("  Stopping...")
        response = httpx.post(
            f"http://localhost:{server_port}/instances/{model}/stop",
            timeout=60.0
        )

        if response.status_code == 200:
            console.print("  [green]Stopped[/green]")
        elif response.status_code == 404:
            # Model not running, just start it
            console.print("  [yellow]Model was not running[/yellow]")
        else:
            console.print(f"[red]Failed to stop model: {response.text}[/red]")
            raise typer.Exit(1)

        # Then start the model
        console.print("  Starting with new configuration...")
        response = httpx.post(
            f"http://localhost:{server_port}/instances/{model}/start",
            timeout=300.0
        )

        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]Successfully restarted {data['model_id']}[/green]")
            console.print(f"Port: {data['port']}")
            console.print(f"PID: {data['pid']}")
            console.print(f"Logs: {VllamaPaths.LOGS_DIR}/{data['model_id'].replace('/', '_')}_{data['port']}.log")
        elif response.status_code == 404:
            data = response.json()
            console.print(f"[red]Error: {data.get('detail', 'Model not found')}[/red]")
            raise typer.Exit(1)
        else:
            data = response.json()
            console.print(f"[red]Failed to start model: {data.get('detail', response.text)}[/red]")
            raise typer.Exit(1)

    except httpx.TimeoutException:
        console.print("[red]Failed to restart model: timed out[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Failed to connect to vllama server[/red]")
        raise typer.Exit(1)


@app.command()
def ps():
    """List all running VLLM instances."""
    server_port = ensure_server_running()

    try:
        response = httpx.get(f"http://localhost:{server_port}/instances", timeout=5.0)
        instances = response.json().get("instances", [])

        if not instances:
            console.print("No active instances")
            return

        # Create table
        table = Table(title="VLLM Instances")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Port", style="yellow")
        table.add_column("PID", style="blue")
        table.add_column("Devices", style="magenta")
        table.add_column("GPU Memory", style="white")
        table.add_column("Last Access", style="white")

        for instance in instances:
            status = instance["status"]
            # Add sleep level info to status
            if "sleeping" in status.lower():
                sleep_level = instance.get("sleep_level")
                if sleep_level:
                    status = f"{status} (L{sleep_level})"

            devices_str = ",".join(str(d) for d in instance["devices"]) if instance["devices"] else "-"
            last_access = format_time_ago(instance["last_request_time"]) if instance["last_request_time"] else "never"

            # Format GPU memory
            gpu_memory_str = "-"
            gpu_used = instance.get("gpu_memory_used")
            gpu_total = instance.get("gpu_memory_total")
            if gpu_used is not None and gpu_total is not None:
                gpu_used_gb = gpu_used / (1024 ** 3)
                gpu_total_gb = gpu_total / (1024 ** 3)
                gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
                gpu_memory_str = f"{gpu_used_gb:.1f}/{gpu_total_gb:.1f}GB ({gpu_percent:.0f}%)"

            table.add_row(
                instance["model_id"],
                status,
                str(instance["port"]),
                str(instance["pid"]) if instance["pid"] else "-",
                devices_str,
                gpu_memory_str,
                last_access,
            )

        console.print(table)

    except httpx.ConnectError:
        console.print("[red]Failed to connect to vllama server[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show vllama server information."""
    if not VllamaPaths.PID_FILE.exists():
        console.print("[yellow]No vllama server is currently running[/yellow]")
        console.print("\nTo start the server:")
        console.print("  vllama serve --port <port>")
        return

    try:
        with open(VllamaPaths.PID_FILE, "r") as f:
            pid_data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading PID file: {e}[/red]")
        return

    pid = pid_data.get("pid")
    if not pid or not psutil.pid_exists(pid):
        console.print("[yellow]Server PID file exists but process is not running[/yellow]")
        console.print(f"Stale PID file: {VllamaPaths.PID_FILE}")
        console.print("\nTo start the server:")
        console.print("  vllama serve --port <port>")
        return

    # Server is running, show info
    server_port = pid_data.get("port")

    # Try to check if server is healthy
    try:
        response = httpx.get(f"http://localhost:{server_port}/health", timeout=5.0)
        is_healthy = response.status_code == 200
    except Exception:
        is_healthy = False

    # Get process info
    try:
        proc = psutil.Process(pid)
        memory_mb = proc.memory_info().rss / 1024 / 1024
        uptime = time.time() - pid_data.get("start_time", time.time())
    except Exception:
        memory_mb = 0
        uptime = 0

    console.print("Vllama Server Information")
    console.print("=" * 60)
    console.print()
    console.print("[bold]Server Status:[/bold]")
    console.print(f"  Status:        [green]Running[/green]" if is_healthy else f"  Status:        [yellow]Not Responding[/yellow]")
    console.print(f"  PID:           {pid}")
    console.print(f"  Host:          {pid_data.get('host', 'unknown')}")
    console.print(f"  Port:          {server_port}")
    console.print(f"  Uptime:        {format_time_ago(pid_data.get('start_time', time.time()))}")
    console.print(f"  Memory Usage:  {memory_mb:.1f} MB")
    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Config Dir:         {pid_data.get('config_dir', VllamaPaths.CONFIG_DIR)}")
    console.print(f"  Models Config:      {pid_data.get('models_config', VllamaPaths.MODELS_CONFIG_FILE)}")
    console.print(f"  Transformers Cache: {pid_data.get('transformers_cache', 'unknown')}")
    console.print(f"  VLLM Port Range:    {pid_data.get('vllm_port_range', [0, 0])[0]} - {pid_data.get('vllm_port_range', [0, 0])[1]}")
    console.print(f"  Unload Mode:        Level {pid_data.get('unload_mode', 'unknown')}")
    console.print(f"  Unload Timeout:     {pid_data.get('unload_timeout', 'unknown')}s")
    console.print()
    console.print("[bold]API Endpoints:[/bold]")
    console.print(f"  Health:       http://localhost:{server_port}/health")
    console.print(f"  Models:       http://localhost:{server_port}/v1/models")
    console.print(f"  Chat:         http://localhost:{server_port}/v1/chat/completions")
    console.print(f"  Completions:  http://localhost:{server_port}/v1/completions")
    console.print(f"  Embeddings:   http://localhost:{server_port}/v1/embeddings")
    console.print()

    if is_healthy:
        console.print("[green]✓[/green] Server is healthy and responding")
    else:
        console.print("[yellow]⚠[/yellow] Server is not responding to health checks")

    console.print()
    console.print("=" * 60)
    console.print()
    console.print("Run [cyan]vllama ps[/cyan] to see active model instances")


@app.command()
def assign(
    model: str = typer.Argument(..., help="Model name to configure"),
    devices: Optional[str] = typer.Option(None, "--devices", "-d", help="GPU devices (comma-separated, e.g., '0,1')"),
    gpu_memory: Optional[float] = typer.Option(None, "--gpu-memory", "-m", help="GPU memory utilization (0.1-1.0)"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Fixed port to use"),
    restart: bool = typer.Option(False, "--restart", "-r", help="Restart instance if running"),
):
    """Configure model settings.

    This command updates the model configuration in ~/.vllama/models.yaml.
    Use --restart to apply changes immediately if the model is running.
    """
    # Load config directly from file (no server needed for this)
    yaml_manager = YAMLConfigManager(VllamaPaths.MODELS_CONFIG_FILE)

    # Scan models to find the requested one
    transformers_cache = get_transformers_cache()
    models = scan_transformers_cache(transformers_cache)
    model_info = find_model_by_name(models, model)

    if not model_info:
        console.print(f"[red]Model '{model}' not found in cache[/red]")
        raise typer.Exit(1)

    # Get existing config or create new one
    model_config = yaml_manager.get_config(model_info.model_id)

    # Update config
    if devices is not None:
        device_list = [int(d.strip()) for d in devices.split(",")]
        model_config.devices = device_list
    if gpu_memory is not None:
        model_config.gpu_memory_utilization = gpu_memory
    if port is not None:
        model_config.port = port

    # Save config
    yaml_manager.set_config(model_info.model_id, model_config)
    yaml_manager.save_configs()

    console.print(f"[green]Updated configuration for {model_info.model_id}[/green]")
    console.print(f"  Devices: {model_config.devices}")
    console.print(f"  GPU Memory: {model_config.gpu_memory_utilization}")
    console.print(f"  Port: {model_config.port}")

    if restart:
        # Check if server is running
        is_running, server_port = check_server_running()
        if not is_running:
            console.print("[yellow]Server not running, cannot restart instance[/yellow]")
            console.print("Use --restart to apply changes immediately if the model is running.")
            return

        console.print("Restarting instance to apply new configuration...")

        try:
            # Check if instance is running
            response = httpx.get(f"http://localhost:{server_port}/instances", timeout=5.0)
            instances = response.json().get("instances", [])
            instance_running = any(inst.get("model_id") == model_info.model_id for inst in instances)

            if not instance_running:
                console.print("[yellow]Instance not running, no restart needed[/yellow]")
                return

            # Stop instance
            httpx.post(
                f"http://localhost:{server_port}/instances/{model_info.model_id}/stop",
                timeout=60.0
            )
            time.sleep(2)

            # Start instance
            response = httpx.post(
                f"http://localhost:{server_port}/instances/{model_info.model_id}/start",
                timeout=300.0
            )

            if response.status_code == 200:
                console.print(f"[green]Restarted {model_info.model_id} with new configuration[/green]")
            else:
                console.print(f"[red]Failed to restart: {response.text}[/red]")
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]Error during restart: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print("Use [cyan]--restart[/cyan] to apply changes immediately if instance is running")


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model ID from Hugging Face (e.g., Qwen/Qwen3-0.6B)"),
    revision: Optional[str] = typer.Option(None, "--revision", "-r", help="Model revision/branch (default: main)"),
):
    """Download a model from Hugging Face Hub.

    Examples:
        vllama pull Qwen/Qwen3-0.6B
        vllama pull BAAI/bge-m3 --revision main
    """
    # Get HF_HOME directory for cache
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    transformers_cache = os.path.join(hf_home, "hub")

    console.print(f"[cyan]Downloading model: {model}[/cyan]")
    if revision:
        console.print(f"[cyan]Revision: {revision}[/cyan]")
    console.print(f"[cyan]HF_HOME: {hf_home}[/cyan]")
    console.print(f"[cyan]Cache directory: {transformers_cache}[/cyan]")
    console.print()

    try:
        # Download model with progress bar
        with console.status(f"[bold green]Downloading {model}..."):
            local_path = snapshot_download(
                repo_id=model,
                revision=revision,
                cache_dir=hf_home,  # snapshot_download expects HF_HOME, not HF_HOME/hub
                resume_download=True,
                local_files_only=False,
            )

        console.print(f"[green]✓[/green] Successfully downloaded {model}")
        console.print(f"[cyan]Local path: {local_path}[/cyan]")
        console.print()
        console.print("You can now use this model with:")
        console.print(f"  vllama start {model}")

    except Exception as e:
        console.print(f"[red]Failed to download model: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="list")
def list_models():
    """List all available models in cache.

    This command scans the HF_HOME/hub directory and displays
    all downloaded models with their types and locations.
    """
    # Get transformers cache directory
    transformers_cache = get_transformers_cache()

    console.print(f"[cyan]Scanning models in: {transformers_cache}[/cyan]")
    console.print()

    # Scan for models
    models = scan_transformers_cache(transformers_cache)

    if not models:
        console.print("[yellow]No models found in cache[/yellow]")
        console.print()
        console.print("Download models with:")
        console.print("  vllama pull <model-id>")
        return

    # Create table
    table = Table(title=f"Available Models ({len(models)} total)")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Local Path", style="white", overflow="fold")

    for model in models:
        model_type = model.model_type or "unknown"
        table.add_row(
            model.model_id,
            model_type,
            model.local_path,
        )

    console.print(table)
    console.print()
    console.print("Start a model with:")
    console.print("  vllama start <model-id>")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
