"""State management for vllama instances."""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InstanceStatus(str, Enum):
    """Status of a VLLM instance."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SLEEPING_L1 = "sleeping_l1"
    SLEEPING_L2 = "sleeping_l2"
    STOPPING = "stopping"
    ERROR = "error"


class InstanceState(BaseModel):
    """State of a single VLLM instance."""

    model_id: str
    port: int
    pid: Optional[int] = None
    status: InstanceStatus = InstanceStatus.STOPPED
    last_request_time: float = Field(default_factory=lambda: datetime.now().timestamp())
    start_time: Optional[float] = None
    memory_delta: dict[int, int] = Field(default_factory=dict)  # device_id -> memory_bytes (freed when sleeping)
    devices: list[int] = Field(default_factory=list)
    sleep_level: Optional[int] = None


class GlobalState(BaseModel):
    """Global state of vllama daemon."""

    instances: dict[str, InstanceState] = Field(default_factory=dict)  # model_id -> state
    used_ports: set[int] = Field(default_factory=set)
    last_updated: float = Field(default_factory=lambda: datetime.now().timestamp())


class StateManager:
    """Manages in-memory state for vllama.

    All state is stored in memory and will be lost when the process exits.
    This ensures that when vllama is closed, all state is cleared.
    """

    def __init__(self):
        """Initialize state manager with empty state."""
        self.state = GlobalState()
        logger.info("StateManager initialized with in-memory storage")

    def get_instance(self, model_id: str) -> Optional[InstanceState]:
        """Get instance state by model ID.

        Args:
            model_id: Model identifier

        Returns:
            InstanceState if exists, None otherwise
        """
        return self.state.instances.get(model_id)

    def set_instance(self, model_id: str, instance: InstanceState):
        """Set instance state.

        Args:
            model_id: Model identifier
            instance: Instance state to set
        """
        self.state.instances[model_id] = instance
        if instance.port:
            self.state.used_ports.add(instance.port)
        self.state.last_updated = datetime.now().timestamp()

    def remove_instance(self, model_id: str):
        """Remove instance from state.

        Args:
            model_id: Model identifier
        """
        if model_id in self.state.instances:
            instance = self.state.instances[model_id]
            if instance.port in self.state.used_ports:
                self.state.used_ports.remove(instance.port)
            del self.state.instances[model_id]
            self.state.last_updated = datetime.now().timestamp()

    def update_last_request_time(self, model_id: str):
        """Update the last request time for an instance.

        Args:
            model_id: Model identifier
        """
        if model_id in self.state.instances:
            self.state.instances[model_id].last_request_time = datetime.now().timestamp()
            self.state.last_updated = datetime.now().timestamp()

    def get_all_instances(self) -> dict[str, InstanceState]:
        """Get all instance states.

        Returns:
            Dictionary of model_id -> InstanceState
        """
        return self.state.instances

    def get_used_ports(self) -> set[int]:
        """Get set of currently used ports.

        Returns:
            Set of used port numbers
        """
        return self.state.used_ports

    def clear_state(self):
        """Clear all state."""
        self.state = GlobalState()
        logger.info("State cleared")

    def update_instance_status(self, model_id: str, status: InstanceStatus):
        """Update the status of an instance.

        Args:
            model_id: Model identifier
            status: New status
        """
        if model_id in self.state.instances:
            self.state.instances[model_id].status = status
            self.state.last_updated = datetime.now().timestamp()

    def update_instance_memory_delta(self, model_id: str, memory_delta: dict[int, int]):
        """Update the memory delta of an instance.

        Args:
            model_id: Model identifier
            memory_delta: Dictionary mapping device_id to memory bytes (freed when sleeping)
        """
        if model_id in self.state.instances:
            self.state.instances[model_id].memory_delta = memory_delta
            self.state.last_updated = datetime.now().timestamp()
