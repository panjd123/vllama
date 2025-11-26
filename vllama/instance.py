"""VLLM instance manager - core module for managing VLLM processes."""

import asyncio
import logging
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import psutil

from vllama.config import ModelConfig, VllamaConfig
from vllama.constants import VllamaPaths
from vllama.gpu import get_gpu_monitor
from vllama.models import ModelInfo
from vllama.state import InstanceState, InstanceStatus, StateManager
from vllama.utils import calculate_llm_inference_vram
from vllama.yaml_manager import YAMLConfigManager

logger = logging.getLogger(__name__)


class VLLMInstanceManager:
    """Manages VLLM server instances."""

    def __init__(
        self,
        config: VllamaConfig,
        state_manager: StateManager,
        yaml_manager: YAMLConfigManager,
    ):
        """Initialize instance manager.

        Args:
            config: Global vllama configuration
            state_manager: State persistence manager
            yaml_manager: YAML configuration manager
        """
        self.config = config
        self.state_manager = state_manager
        self.yaml_manager = yaml_manager
        self.gpu_monitor = get_gpu_monitor()
        self._http_client: Optional[httpx.AsyncClient] = None
        self.start_lock = asyncio.Lock()

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Cleanup resources."""
        if self._http_client is not None:
            await self._http_client.aclose()

    def _get_model_config(self, model_info: ModelInfo) -> ModelConfig:
        """Get configuration for a model, creating default if not exists.

        Args:
            model_info: Model information

        Returns:
            Model configuration
        """
        config = self.yaml_manager.get_config(model_info.model_id)
        if config is None:
            logger.info(f"No YAML config for {model_info.model_id}, using default settings")
            config = ModelConfig(model_name=model_info.model_id)
        return config

    def _get_default_devices(self, model_config: ModelConfig) -> list[int]:
        """Get default devices for a model.

        Priority:
        1. model_config.devices if specified
        2. config.default_device if specified
        3. Auto-select device with most free memory

        Args:
            model_config: Model configuration

        Returns:
            List of device IDs
        """
        # Use model-specific devices if specified
        if model_config.devices:
            return model_config.devices

        # Use global default device if specified
        if self.config.default_device is not None:
            return [self.config.default_device]

        # Auto-select device with most total memory
        if self.gpu_monitor.get_device_count() > 0:
            best_device = self.gpu_monitor.get_device_with_most_total_memory()
            logger.info(f"Auto-selected GPU {best_device} (most total memory)")
            return [best_device]

        # Fallback to device 0
        return [0]

    def _calculate_optimal_memory_utilization(self, devices: list[int], request_memory_utilization: float=None) -> float:
        """Calculate optimal GPU memory utilization based on available memory.

        Args:
            devices: GPU device IDs

        Returns:
            Optimal memory utilization ratio
        """
        if not devices or self.gpu_monitor.get_device_count() == 0:
            return 0.85  # Default fallback

        # Get memory info for the first device (assume all devices have similar free memory)
        mem_info = self.gpu_monitor.get_memory_info(devices[0])
        total_memory = mem_info.get("total", 0)
        free_memory = mem_info.get("free", 0)

        if total_memory == 0:
            return 0.85  # Default fallback

        # Calculate what percentage of total memory is free
        free_ratio = free_memory / total_memory

        optimal = min(request_memory_utilization or 0.85, free_ratio * 0.9, (free_memory - 0.5 * 1024**3) / total_memory)

        logger.info(
            f"GPU memory: {free_memory / (1024**3):.2f}GB free / "
            f"{total_memory / (1024**3):.2f}GB total ({free_ratio:.1%}). "
            f"Using {optimal:.2f} utilization"
        )

        return optimal

    def _check_memory_requirements(
        self,
        model_info: ModelInfo,
        model_config: ModelConfig,
        devices: list[int],
        gpu_memory_util: float
    ) -> None:
        """Check if GPU has enough memory for the model.

        Args:
            model_info: Model information
            model_config: Model configuration
            devices: GPU devices to use
            gpu_memory_util: GPU memory utilization ratio

        Raises:
            RuntimeError: If insufficient memory
        """
        if not devices or self.gpu_monitor.get_device_count() == 0:
            # No GPU monitoring available, skip check
            return

        try:
            context_length = model_config.max_model_len

            # Map dtype to supported values for calculate_llm_inference_vram
            # "auto" will be mapped to "bf16" as a reasonable default for modern models
            dtype_map = {
                "auto": "bf16",
                "half": "bf16",
                "float16": "bf16",
                "bfloat16": "bf16",
                "float32": "fp32",
                "float": "fp32",
            }
            dtype = dtype_map.get(model_config.dtype.lower(), model_config.dtype.lower())

            # Calculate required VRAM
            vram_info = calculate_llm_inference_vram(
                model_name_or_path=model_info.local_path,
                context_length=context_length,
                batch_size=1,  # Default batch size for inference
                dtype=dtype
            )

            # Get minimum required VRAM (excluding KV cache as it's dynamic)
            min_required_gb = vram_info["total_vram_excluding_kv_cache_GB"]

            # Get GPU total memory
            mem_info = self.gpu_monitor.get_memory_info(devices[0])
            total_memory_gb = mem_info.get("total", 0) / (1024 ** 3)

            if total_memory_gb == 0:
                # Cannot determine GPU memory, skip check
                logger.warning("Cannot determine GPU memory, skipping pre-check")
                return

            # Calculate available memory with configured utilization
            available_memory_gb = total_memory_gb * gpu_memory_util

            logger.info(
                f"Memory pre-check: Model requires at least {min_required_gb:.2f}GB, "
                f"GPU has {total_memory_gb:.2f}GB total, "
                f"configured utilization {gpu_memory_util:.2f} provides {available_memory_gb:.2f}GB"
            )

            # Check if we have enough memory
            if min_required_gb > available_memory_gb:
                raise RuntimeError(
                    f"Insufficient GPU memory for model {model_info.model_id}:\n"
                    f"  - Model requires at least: {min_required_gb:.2f}GB\n"
                    f"  - GPU total memory: {total_memory_gb:.2f}GB\n"
                    f"  - Configured gpu_memory_utilization: {gpu_memory_util:.2f}\n"
                    f"  - Available memory: {available_memory_gb:.2f}GB\n"
                    f"  - Memory shortage: {min_required_gb - available_memory_gb:.2f}GB\n"
                    f"\nSuggestions:\n"
                    f"  1. Reduce max_model_len (current: {context_length})\n"
                    f"  2. Use a smaller model\n"
                    f"  3. Free up GPU memory by stopping other models"
                )

        except RuntimeError:
            # Re-raise RuntimeError (our custom error)
            raise
        except Exception as e:
            # For other errors (like model config loading failure), log warning and continue
            logger.warning(f"Failed to pre-check memory requirements: {e}")
            logger.warning("Proceeding with model start, vLLM will report if memory is insufficient")

    def _build_vllm_command(
        self,
        model_info: ModelInfo,
        model_config: ModelConfig,
        port: int,
        devices: list[int],
        gpu_memory_util: float
    ) -> list[str]:
        """Build vllm serve command.

        Args:
            model_info: Model information
            model_config: Model configuration
            port: Port to serve on
            devices: GPU devices to use
            gpu_memory_util: GPU memory utilization ratio

        Returns:
            Command list for subprocess
        """
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_info.local_path,
            "--served-model-name", model_info.model_id,  # Register friendly model name
            "--port", str(port),
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", str(gpu_memory_util),
            "--tensor-parallel-size", str(model_config.tensor_parallel_size),
            "--dtype", model_config.dtype,
            "--enable-sleep-mode",  # Enable sleep/wake functionality
        ]

        if model_config.max_model_len:
            cmd.extend(["--max-model-len", str(model_config.max_model_len)])

        if model_config.trust_remote_code:
            cmd.append("--trust-remote-code")

        # Add extra arguments
        for key, value in model_config.extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        return cmd

    async def start_instance(
        self,
        model_info: ModelInfo,
        port: Optional[int] = None
    ) -> InstanceState:
        """Start a new VLLM instance.

        Args:
            model_info: Model information
            port: Optional port to use, auto-assign if None

        Returns:
            Instance state
        """
        model_id = model_info.model_id

        # Check if already running
        existing = self.state_manager.get_instance(model_id)
        if existing and existing.status == InstanceStatus.RUNNING:
            logger.info(f"Instance {model_id} already running on port {existing.port}")
            return existing

        # Get configuration
        model_config = self._get_model_config(model_info)

        # Assign port if not specified
        if port is None:
            port = model_config.port or self.config.get_next_available_port(
                self.state_manager.get_used_ports()
            )

        # Prepare devices - use model-specific, default, or auto-select
        devices = self._get_default_devices(model_config)

        # Calculate GPU memory utilization (same logic as in _build_vllm_command)
        gpu_memory_util = model_config.gpu_memory_utilization
        gpu_memory_util = self._calculate_optimal_memory_utilization(devices, gpu_memory_util)
        logger.info(f"Auto-calculated GPU memory utilization: {gpu_memory_util:.2f}")

        # Pre-check memory requirements before starting
        self._check_memory_requirements(model_info, model_config, devices, gpu_memory_util)

        # Create initial state
        instance_state = InstanceState(
            model_id=model_id,
            port=port,
            status=InstanceStatus.STARTING,
            devices=devices,
            start_time=datetime.now().timestamp(),
            last_request_time=datetime.now().timestamp(),
        )
        self.state_manager.set_instance(model_id, instance_state)

        # Build command
        cmd = self._build_vllm_command(model_info, model_config, port, devices, gpu_memory_util)

        # Set environment for GPU devices
        env = os.environ.copy()
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        # Enable vLLM development mode for sleep functionality
        env["VLLM_SERVER_DEV_MODE"] = "1"

        # Start process
        try:
            # Create logs directory
            log_dir = VllamaPaths.LOGS_DIR
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create log file path
            # Sanitize model_id for filename (replace / with _)
            safe_model_id = model_id.replace("/", "_")
            log_file = log_dir / f"{safe_model_id}_{port}.log"

            # Log model configuration
            logger.info(f"Starting VLLM instance for {model_id} on port {port}")
            logger.info(f"Model configuration:")
            logger.info(f"  - devices: {devices}")
            logger.info(f"  - gpu_memory_utilization: {gpu_memory_util}")
            logger.info(f"  - max_model_len: {model_config.max_model_len}")
            logger.info(f"  - tensor_parallel_size: {model_config.tensor_parallel_size}")
            logger.info(f"  - dtype: {model_config.dtype}")
            logger.info(f"  - trust_remote_code: {model_config.trust_remote_code}")
            if model_config.extra_args:
                logger.info(f"  - extra_args: {model_config.extra_args}")
            logger.info(f"Logs will be written to: {log_file}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Open log file
            log_fd = open(log_file, "w", buffering=1)  # Line buffered

            # Start process with start_new_session to create a new process group
            # This allows us to kill the vLLM instance without affecting the parent server
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fd,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                start_new_session=True,
            )

            instance_state.pid = process.pid
            self.state_manager.set_instance(model_id, instance_state)

            # Wait for server to be ready (pass log_file for better error reporting)
            ready = await self._wait_for_server(port, pid=process.pid, timeout=300, log_file=log_file)
            if not ready:
                # Kill the process if it didn't start properly
                try:
                    if psutil.pid_exists(process.pid):
                        process.terminate()
                        process.wait(timeout=5)
                except Exception:
                    try:
                        if psutil.pid_exists(process.pid):
                            process.kill()
                    except Exception:
                        pass
                log_fd.close()
                instance_state.status = InstanceStatus.ERROR
                self.state_manager.set_instance(model_id, instance_state)

                # Read last lines of log for error message
                error_details = ""
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get last 10 lines for error message
                            last_lines = ''.join(lines[-10:])
                            error_details = f"\n\nLast log lines:\n{last_lines}"
                except Exception:
                    pass

                raise RuntimeError(
                    f"VLLM server failed to start. "
                    f"Check logs at: {log_file}{error_details}"
                )

            instance_state.status = InstanceStatus.RUNNING
            self.state_manager.set_instance(model_id, instance_state)
            logger.info(f"VLLM instance {model_id} started successfully on port {port}")

            # Note: log_fd is intentionally kept open as the process is still writing to it
            return instance_state

        except Exception as e:
            logger.error(f"Failed to start VLLM instance: {e}")
            instance_state.status = InstanceStatus.ERROR
            self.state_manager.set_instance(model_id, instance_state)
            raise

    async def _wait_for_server(
        self, port: int, pid: Optional[int] = None, timeout: int = 300, log_file: Optional[Path] = None
    ) -> bool:
        """Wait for VLLM server to be ready.

        Args:
            port: Server port
            pid: Process ID to monitor (optional)
            timeout: Maximum wait time in seconds
            log_file: Path to log file for error reporting

        Returns:
            True if server is ready, False otherwise
        """
        client = await self.get_http_client()
        start_time = datetime.now().timestamp()
        last_check_time = start_time

        while (datetime.now().timestamp() - start_time) < timeout:
            current_time = datetime.now().timestamp()

            # Check if process is still alive (check more frequently)
            if pid is not None:
                if not psutil.pid_exists(pid):
                    logger.error(f"VLLM process {pid} died during startup")

                    # Try to get last lines from log file for better error message
                    if log_file and log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                # Get last 20 lines
                                last_lines = ''.join(lines[-20:]) if lines else "Empty log file"
                                logger.error(f"Last log lines:\n{last_lines}")
                        except Exception as e:
                            logger.warning(f"Could not read log file: {e}")

                    return False

                # Also check if process is in zombie state
                try:
                    proc = psutil.Process(pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        logger.error(f"VLLM process {pid} is in zombie state")
                        return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.error(f"Cannot access VLLM process {pid}")
                    return False

            try:
                response = await client.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            # Sleep for shorter interval for faster failure detection
            await asyncio.sleep(0.5)

        return False

    async def stop_instance(self, model_id: str, force: bool = False) -> bool:
        """Stop a VLLM instance.

        Args:
            model_id: Model identifier
            force: Force kill the process

        Returns:
            True if stopped successfully
        """
        instance = self.state_manager.get_instance(model_id)
        if not instance:
            logger.warning(f"Instance {model_id} not found")
            return False

        if instance.status == InstanceStatus.STOPPED:
            logger.info(f"Instance {model_id} already stopped")
            return True

        if not instance.pid:
            logger.warning(f"Instance {model_id} has no PID")
            self.state_manager.remove_instance(model_id)
            return True

        try:
            # Update status
            instance.status = InstanceStatus.STOPPING
            self.state_manager.set_instance(model_id, instance)

            # Try to terminate gracefully
            if psutil.pid_exists(instance.pid):
                if force:
                    os.killpg(os.getpgid(instance.pid), signal.SIGKILL)
                else:
                    os.killpg(os.getpgid(instance.pid), signal.SIGTERM)

                # Wait for process to terminate
                for _ in range(30):  # Wait up to 30 seconds
                    if not psutil.pid_exists(instance.pid):
                        break
                    await asyncio.sleep(1)
                else:
                    # Force kill if still running
                    os.killpg(os.getpgid(instance.pid), signal.SIGKILL)

            # Remove from state
            self.state_manager.remove_instance(model_id)
            logger.info(f"Instance {model_id} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop instance {model_id}: {e}")
            # Still remove from state
            self.state_manager.remove_instance(model_id)
            return False

    async def sleep_instance(self, model_id: str, level: int = 2) -> bool:
        """Put VLLM instance to sleep.

        Args:
            model_id: Model identifier
            level: Sleep level (1, 2, or 3)

        Returns:
            True if successful
        """
        instance = self.state_manager.get_instance(model_id)
        if not instance:
            logger.warning(f"Instance {model_id} not found")
            return False

        if instance.status != InstanceStatus.RUNNING:
            logger.warning(f"Instance {model_id} is not running (status: {instance.status})")
            return False

        if level == 3:
            # Level 3: completely stop the instance
            return await self.stop_instance(model_id)

        # Record memory usage before sleeping for each device
        memory_before: dict[int, int] = {}
        if instance.devices and self.gpu_monitor.get_device_count() > 0:
            for device_id in instance.devices:
                if device_id < self.gpu_monitor.get_device_count():
                    mem_info = self.gpu_monitor.get_memory_info(device_id)
                    memory_before[device_id] = mem_info.get("used", 0)

        try:
            client = await self.get_http_client()
            url = f"http://localhost:{instance.port}/sleep?level={level}"
            response = await client.post(url)

            if response.status_code == 200:
                # Calculate memory delta after sleeping for each device
                memory_delta: dict[int, int] = {}
                if instance.devices and self.gpu_monitor.get_device_count() > 0:
                    total_freed = 0
                    for device_id in instance.devices:
                        if device_id < self.gpu_monitor.get_device_count() and device_id in memory_before:
                            mem_info = self.gpu_monitor.get_memory_info(device_id)
                            memory_after = mem_info.get("used", 0)
                            freed = memory_before[device_id] - memory_after  # How much was freed on this device
                            memory_delta[device_id] = freed
                            total_freed += freed
                            logger.info(f"Device {device_id}: freed {freed / (1024**3):.2f} GB")

                    logger.info(f"Total memory freed across all devices: {total_freed / (1024**3):.2f} GB")
                    instance.memory_delta = memory_delta

                instance.status = InstanceStatus.SLEEPING_L1 if level == 1 else InstanceStatus.SLEEPING_L2
                instance.sleep_level = level
                self.state_manager.set_instance(model_id, instance)
                logger.info(f"Instance {model_id} put to sleep (level {level})")
                return True
            elif response.status_code == 404:
                # Sleep API not available, fall back to stop
                logger.warning(f"Sleep API not available for {model_id}, using stop instead")
                return await self.stop_instance(model_id)
            else:
                logger.error(f"Failed to sleep instance: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to sleep instance {model_id}: {e}")
            return False

    async def wake_instance(self, model_id: str) -> bool:
        """Wake up a sleeping VLLM instance.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        instance = self.state_manager.get_instance(model_id)
        if not instance:
            logger.warning(f"Instance {model_id} not found")
            return False

        if instance.status not in (InstanceStatus.SLEEPING_L1, InstanceStatus.SLEEPING_L2):
            logger.warning(f"Instance {model_id} is not sleeping (status: {instance.status})")
            return False

        # Check if enough memory is available for wake up on each device
        if instance.memory_delta and instance.devices and self.gpu_monitor.get_device_count() > 0:
            for device_id in instance.devices:
                if device_id not in instance.memory_delta:
                    continue

                required_memory = instance.memory_delta[device_id]
                if required_memory <= 0:
                    continue

                # Check if this device has enough free memory
                has_memory, error_msg = self.gpu_monitor.has_enough_memory(
                    [device_id], required_memory
                )
                if not has_memory:
                    raise RuntimeError(
                        f"Cannot wake instance on device {device_id}: {error_msg}. "
                        f"Need {required_memory / (1024**3):.2f}GB"
                    )

        try:
            client = await self.get_http_client()
            port = instance.port

            if instance.sleep_level == 1:
                # Simple wake up for level 1
                response = await client.post(f"http://localhost:{port}/wake_up")
                if response.status_code != 200:
                    raise RuntimeError(f"Wake up failed: {response.text}")

            elif instance.sleep_level == 2:
                # Level 2 requires multi-step wake up
                # 1. Reallocate weights memory
                response = await client.post(f"http://localhost:{port}/wake_up?tags=weights")
                if response.status_code != 200:
                    raise RuntimeError(f"Wake up weights failed: {response.text}")

                # 2. Load weights in-place
                response = await client.post(
                    f"http://localhost:{port}/collective_rpc",
                    json={"method": "reload_weights"}
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Reload weights failed: {response.text}")

                # 3. Reallocate KV cache
                response = await client.post(f"http://localhost:{port}/wake_up?tags=kv_cache")
                if response.status_code != 200:
                    raise RuntimeError(f"Wake up KV cache failed: {response.text}")

            instance.status = InstanceStatus.RUNNING
            instance.sleep_level = None
            instance.last_request_time = datetime.now().timestamp()
            self.state_manager.set_instance(model_id, instance)

            logger.info(f"Instance {model_id} woken up successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to wake instance {model_id}: {e}")
            return False

    async def evict_models_for_memory(self, required_memory: dict[int, int], devices: list[int]) -> bool:
        """Evict running models to free up memory.

        Evicts models in LRU order (least recently used first) until enough memory is available.

        Args:
            required_memory: Dictionary mapping device_id to required memory in bytes
            devices: GPU device IDs that need memory

        Returns:
            True if enough memory was freed, False otherwise
        """
        if not devices or self.gpu_monitor.get_device_count() == 0:
            return False

        # Check current free memory for each device
        device_free_memory: dict[int, int] = {}
        all_satisfied = True

        for device_id in devices:
            if device_id >= self.gpu_monitor.get_device_count():
                logger.warning(f"Device {device_id} does not exist")
                continue

            mem_info = self.gpu_monitor.get_memory_info(device_id)
            free_memory = mem_info.get("free", 0)
            device_free_memory[device_id] = free_memory

            required = required_memory.get(device_id, 0)
            logger.info(
                f"Device {device_id}: need {required / (1024**3):.2f}GB, "
                f"have {free_memory / (1024**3):.2f}GB free"
            )

            if free_memory < required:
                all_satisfied = False

        if all_satisfied:
            return True  # Already have enough memory on all devices

        # Get all running instances that use any of the target devices
        all_instances = self.state_manager.get_all_instances()
        candidate_instances = []

        for model_id, inst in all_instances.items():
            if inst.status != InstanceStatus.RUNNING:
                continue
            if not inst.devices:
                continue

            # Check if this instance uses any of the target devices
            if any(dev in devices for dev in inst.devices):
                candidate_instances.append((model_id, inst))

        if not candidate_instances:
            logger.warning("No running instances to evict")
            return False

        # Sort by last_request_time (oldest first)
        candidate_instances.sort(key=lambda x: x[1].last_request_time)

        logger.info(f"Found {len(candidate_instances)} running instances to potentially evict")

        # Evict models one by one until we have enough memory on all devices
        for model_id, instance in candidate_instances:
            # Check if all devices now have enough memory
            all_satisfied = True
            for device_id in devices:
                if device_id not in device_free_memory:
                    continue
                required = required_memory.get(device_id, 0)
                if device_free_memory[device_id] < required:
                    all_satisfied = False
                    break

            if all_satisfied:
                break

            logger.info(
                f"Evicting model {model_id} (last accessed "
                f"{(datetime.now().timestamp() - instance.last_request_time):.0f}s ago)"
            )

            # Sleep the model (level 2 to free maximum memory while keeping it quickly recoverable)
            success = await self.sleep_instance(model_id, level=2)

            if success:
                # Update free memory for all devices used by this instance
                await asyncio.sleep(1)  # Give time for memory to be freed

                for device_id in devices:
                    if device_id >= self.gpu_monitor.get_device_count():
                        continue
                    mem_info = self.gpu_monitor.get_memory_info(device_id)
                    device_free_memory[device_id] = mem_info.get("free", 0)
                    logger.info(
                        f"Device {device_id} after evicting {model_id}: "
                        f"{device_free_memory[device_id] / (1024**3):.2f}GB free"
                    )
            else:
                logger.warning(f"Failed to evict model {model_id}")

        # Final check - verify all devices have enough memory
        all_satisfied = True
        for device_id in devices:
            if device_id not in device_free_memory:
                continue
            required = required_memory.get(device_id, 0)
            if device_free_memory[device_id] < required:
                all_satisfied = False
                logger.warning(
                    f"Device {device_id} still insufficient: "
                    f"need {required / (1024**3):.2f}GB, "
                    f"have {device_free_memory[device_id] / (1024**3):.2f}GB"
                )

        return all_satisfied

    async def ensure_instance_running(self, model_info: ModelInfo, auto_evict: bool = True) -> InstanceState:
        """Ensure an instance is running, starting or waking it if necessary.

        Args:
            model_info: Model information
            auto_evict: Whether to automatically evict other models if memory is insufficient

        Returns:
            Instance state
        """
        async with self.start_lock:
            model_id = model_info.model_id
            instance = self.state_manager.get_instance(model_id)

            if instance is None or instance.status == InstanceStatus.STOPPED:
                # Starting new instance - estimate memory needed (conservative estimate)
                model_config = self._get_model_config(model_info)
                devices = self._get_default_devices(model_config)

                if auto_evict and self.gpu_monitor.get_device_count() > 0:
                    # Calculate required memory for each device
                    required_memory: dict[int, int] = {}

                    for device_id in devices:
                        if device_id >= self.gpu_monitor.get_device_count():
                            continue

                        mem_info = self.gpu_monitor.get_memory_info(device_id)
                        total_memory = mem_info.get("total", 0)

                        # Estimate needed memory: use gpu_memory_utilization setting
                        # For tensor parallel, memory is distributed across GPUs
                        gpu_util = model_config.gpu_memory_utilization
                        estimated_needed = int(total_memory * gpu_util / model_config.tensor_parallel_size)
                        required_memory[device_id] = estimated_needed

                    # Check if eviction is needed
                    needs_eviction = False
                    for device_id in devices:
                        if device_id not in required_memory:
                            continue
                        mem_info = self.gpu_monitor.get_memory_info(device_id)
                        free_memory = mem_info.get("free", 0)
                        required = required_memory[device_id]

                        logger.info(
                            f"Device {device_id}: estimated need {required / (1024**3):.2f}GB, "
                            f"free: {free_memory / (1024**3):.2f}GB"
                        )

                        if free_memory < required:
                            needs_eviction = True

                    if needs_eviction:
                        logger.info(f"Insufficient memory, attempting to evict models")
                        evicted = await self.evict_models_for_memory(required_memory, devices)
                        if not evicted:
                            required_str = ", ".join(
                                f"Device {dev}: {mem / (1024**3):.2f}GB"
                                for dev, mem in required_memory.items()
                            )
                            raise RuntimeError(
                                f"Cannot start {model_id}: insufficient GPU memory even after eviction. "
                                f"Required: {required_str}"
                            )

                # Start new instance
                return await self.start_instance(model_info)

            elif instance.status in (InstanceStatus.SLEEPING_L1, InstanceStatus.SLEEPING_L2):
                # Waking up sleeping instance - check memory_delta for each device
                if auto_evict and instance.memory_delta and instance.devices and self.gpu_monitor.get_device_count() > 0:
                    # Check each device
                    needs_eviction = False
                    for device_id in instance.devices:
                        if device_id not in instance.memory_delta:
                            continue

                        mem_info = self.gpu_monitor.get_memory_info(device_id)
                        free_memory = mem_info.get("free", 0)
                        required = instance.memory_delta[device_id]

                        logger.info(
                            f"Device {device_id}: memory delta {required / (1024**3):.2f}GB, "
                            f"free: {free_memory / (1024**3):.2f}GB"
                        )

                        if free_memory < required:
                            needs_eviction = True

                    if needs_eviction:
                        logger.info(f"Insufficient memory to wake {model_id}, attempting to evict models")
                        evicted = await self.evict_models_for_memory(instance.memory_delta, instance.devices)
                        if not evicted:
                            required_str = ", ".join(
                                f"Device {dev}: {mem / (1024**3):.2f}GB"
                                for dev, mem in instance.memory_delta.items()
                            )
                            raise RuntimeError(
                                f"Cannot wake {model_id}: insufficient GPU memory even after eviction. "
                                f"Required: {required_str}"
                            )

                # Wake up sleeping instance
                success = await self.wake_instance(model_id)
                if not success:
                    raise RuntimeError(f"Failed to wake instance {model_id}")
                return self.state_manager.get_instance(model_id)

            elif instance.status == InstanceStatus.ERROR:
                # Error state - clean up and restart (with same auto_evict logic as new instance)
                logger.info(f"Instance {model_id} in ERROR state, cleaning up and restarting")
                await self.stop_instance(model_id, force=True)
                return await self.ensure_instance_running(model_info, auto_evict=auto_evict)

            elif instance.status == InstanceStatus.RUNNING:
                # Already running, update last request time
                self.state_manager.update_last_request_time(model_id)
                return instance

            elif instance.status == InstanceStatus.STARTING:
                # Still starting, wait for it
                raise RuntimeError(f"Instance {model_id} is still starting, please wait")

            else:
                raise RuntimeError(f"Instance {model_id} in invalid state: {instance.status}")

    def get_instance_status(self, model_id: str) -> Optional[InstanceState]:
        """Get the status of an instance.

        Args:
            model_id: Model identifier

        Returns:
            Instance state if exists
        """
        return self.state_manager.get_instance(model_id)

    def get_all_instances(self) -> dict[str, InstanceState]:
        """Get all instance states.

        Returns:
            Dictionary of model_id -> InstanceState
        """
        return self.state_manager.get_all_instances()

    async def cleanup_all_instances(self):
        """Cleanup all running instances.

        This method should be called when shutting down vllama to ensure
        all child vLLM processes are properly terminated.
        """
        instances = self.state_manager.get_all_instances()

        if not instances:
            logger.info("No instances to cleanup")
            return

        logger.info(f"Cleaning up {len(instances)} instances...")

        for model_id, instance in instances.items():
            if instance.pid and psutil.pid_exists(instance.pid):
                try:
                    logger.info(f"Terminating instance {model_id} (PID: {instance.pid})")

                    # Try to get process group and terminate all processes in the group
                    try:
                        pgid = os.getpgid(instance.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        logger.debug(f"Sent SIGTERM to process group {pgid}")

                        # Wait up to 5 seconds for graceful termination
                        await asyncio.sleep(1)
                        if psutil.pid_exists(instance.pid):
                            for _ in range(4):
                                await asyncio.sleep(1)
                                if not psutil.pid_exists(instance.pid):
                                    break
                            else:
                                # Force kill if still running
                                logger.warning(f"Force killing instance {model_id}")
                                os.killpg(pgid, signal.SIGKILL)

                        logger.info(f"Instance {model_id} terminated gracefully")
                    except ProcessLookupError:
                        # Process group doesn't exist, try individual process
                        proc = psutil.Process(instance.pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            proc.wait()

                except (psutil.NoSuchProcess, ProcessLookupError):
                    logger.debug(f"Instance {model_id} already terminated")
                except Exception as e:
                    logger.error(f"Error terminating instance {model_id}: {e}")

        # Clear all state
        self.state_manager.clear_state()
        logger.info("All instances cleaned up")
