"""Scheduler for automatically unloading idle instances."""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from vllama.config import VllamaConfig
from vllama.state import InstanceStatus

if TYPE_CHECKING:
    from vllama.instance import VLLMInstanceManager

logger = logging.getLogger(__name__)


class UnloadScheduler:
    """Scheduler for automatically unloading idle VLLM instances."""

    def __init__(
        self,
        config: VllamaConfig,
        instance_manager: "VLLMInstanceManager",
    ):
        """Initialize scheduler.

        Args:
            config: Global vllama configuration
            instance_manager: VLLM instance manager
        """
        self.config = config
        self.instance_manager = instance_manager
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        logger.info("Unload scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Unload scheduler stopped")

    async def _run_scheduler(self):
        """Main scheduler loop."""
        check_interval = 60  # Check every 60 seconds

        while self._running:
            try:
                await self._check_and_unload_idle_instances()
            except Exception as e:
                logger.error(f"Error in scheduler: {e}", exc_info=True)

            try:
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break

    async def _check_and_unload_idle_instances(self):
        """Check all instances and unload idle ones."""
        current_time = datetime.now().timestamp()
        timeout = self.config.unload_timeout
        unload_mode = self.config.unload_mode

        instances = self.instance_manager.get_all_instances()

        for model_id, instance in instances.items():
            # Only check running instances
            if instance.status != InstanceStatus.RUNNING:
                continue

            # Calculate idle time
            idle_time = current_time - instance.last_request_time

            if idle_time >= timeout:
                logger.info(
                    f"Instance {model_id} has been idle for {idle_time:.0f}s, "
                    f"unloading with mode {unload_mode}"
                )

                try:
                    if unload_mode in (1, 2):
                        # Sleep the instance
                        success = await self.instance_manager.sleep_instance(
                            model_id, level=unload_mode
                        )
                        if success:
                            logger.info(f"Instance {model_id} put to sleep (level {unload_mode})")
                        else:
                            logger.error(f"Failed to sleep instance {model_id}")

                    elif unload_mode == 3:
                        # Stop the instance
                        success = await self.instance_manager.stop_instance(model_id)
                        if success:
                            logger.info(f"Instance {model_id} stopped")
                        else:
                            logger.error(f"Failed to stop instance {model_id}")

                except Exception as e:
                    logger.error(f"Failed to unload instance {model_id}: {e}", exc_info=True)

    def get_status(self) -> dict:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler status information
        """
        return {
            "running": self._running,
            "timeout": self.config.unload_timeout,
            "mode": self.config.unload_mode,
        }
