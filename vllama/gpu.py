"""GPU and memory monitoring utilities."""

import logging
from typing import Optional
import torch

try:
    import pynvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU memory usage."""

    def __init__(self):
        """Initialize GPU monitor."""
        self._initialized = False
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self._initialized = True
                self.device_count = nvml.nvmlDeviceGetCount()
                logger.info(f"GPU monitor initialized with {self.device_count} devices")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self._initialized = False
        else:
            logger.warning("pynvml not available, GPU monitoring disabled")

    def __del__(self):
        """Cleanup NVML."""
        if self._initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    def get_device_count(self) -> int:
        """Get number of available GPU devices."""
        if not self._initialized:
            return 0
        return self.device_count

    def get_memory_info(self, device_id: int = 0) -> dict[str, int]:
        """Get memory information for a specific device.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with 'total', 'used', and 'free' memory in bytes
        """
        if not self._initialized:
            return {"total": 0, "used": 0, "free": 0}

        try:
            # handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
            # mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            # return {
            #     "total": mem_info.total,
            #     "used": mem_info.used,
            #     "free": mem_info.free,
            # }
            free, total = torch.cuda.mem_get_info(device_id)
            return {
                "total": total,
                "used": total - free,
                "free": free,
            }
        except Exception as e:
            logger.error(f"Failed to get memory info for device {device_id}: {e}")
            return {"total": 0, "used": 0, "free": 0}

    def get_all_memory_info(self) -> dict[int, dict[str, int]]:
        """Get memory information for all devices.

        Returns:
            Dictionary mapping device_id to memory info
        """
        if not self._initialized:
            return {}

        result = {}
        for i in range(self.device_count):
            result[i] = self.get_memory_info(i)
        return result

    def get_device_with_most_free_memory(self) -> int:
        """Get the device ID with the most free memory.

        Returns:
            Device ID with most free memory, or 0 if no devices available
        """
        if not self._initialized or self.device_count == 0:
            return 0

        max_free = -1
        best_device = 0

        for i in range(self.device_count):
            mem_info = self.get_memory_info(i)
            if mem_info["free"] > max_free:
                max_free = mem_info["free"]
                best_device = i

        logger.debug(f"Device {best_device} has most free memory: {max_free / (1024**3):.2f}GB")
        return best_device

    def get_device_with_most_total_memory(self) -> int:
        """Get the device ID with the most total memory.

        Returns:
            Device ID with most total memory, or 0 if no devices available
        """
        if not self._initialized or self.device_count == 0:
            return 0

        max_total = -1
        best_device = 0

        for i in range(self.device_count):
            mem_info = self.get_memory_info(i)
            if mem_info["total"] > max_total:
                max_total = mem_info["total"]
                best_device = i

        logger.debug(f"Device {best_device} has most total memory: {max_total / (1024**3):.2f}GB")
        return best_device

    def has_enough_memory(
        self,
        device_ids: list[int],
        required_bytes: int
    ) -> tuple[bool, Optional[str]]:
        """Check if devices have enough free memory.

        Args:
            device_ids: List of GPU device IDs to check
            required_bytes: Required memory in bytes

        Returns:
            Tuple of (has_enough, error_message)
        """
        if not self._initialized:
            return True, None  # Cannot verify, assume ok

        for device_id in device_ids:
            if device_id >= self.device_count:
                return False, f"Device {device_id} does not exist"

            mem_info = self.get_memory_info(device_id)
            if mem_info["free"] < required_bytes:
                free_gb = mem_info["free"] / (1024 ** 3)
                required_gb = required_bytes / (1024 ** 3)
                return False, (
                    f"Device {device_id} has insufficient memory: "
                    f"{free_gb:.2f}GB free, {required_gb:.2f}GB required"
                )

        return True, None

    def estimate_model_memory(
        self,
        _model_path: str,
        _tensor_parallel_size: int = 1,
        _gpu_memory_utilization: float = 0.9
    ) -> int:
        """Estimate memory requirements for a model.

        This is a rough estimate. For more accurate results, you should
        profile the actual model loading.

        Note: This method is deprecated and not used anymore since we let
        vLLM handle memory allocation directly.

        Args:
            _model_path: Path to the model (unused)
            _tensor_parallel_size: Number of GPUs for tensor parallelism (unused)
            _gpu_memory_utilization: GPU memory utilization ratio (unused)

        Returns:
            Estimated memory in bytes
        """
        # This is a placeholder - in reality, you'd want to:
        # 1. Read model config to get parameter count
        # 2. Estimate based on dtype (fp16, bf16, etc.)
        # 3. Add overhead for KV cache and activations

        # For now, return a conservative estimate
        # This should be improved with actual model inspection
        return int(8 * (1024 ** 3))  # 8GB default estimate


# Global instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Get global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor
