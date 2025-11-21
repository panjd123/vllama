"""Tests for state management (vllama/state.py)."""

import time

import pytest

from vllama.state import InstanceState, InstanceStatus, StateManager


class TestInstanceStatus:
    """Tests for InstanceStatus enum."""

    def test_all_statuses(self):
        """Test all status values."""
        assert InstanceStatus.STOPPED == "stopped"
        assert InstanceStatus.STARTING == "starting"
        assert InstanceStatus.RUNNING == "running"
        assert InstanceStatus.SLEEPING_L1 == "sleeping_l1"
        assert InstanceStatus.SLEEPING_L2 == "sleeping_l2"
        assert InstanceStatus.STOPPING == "stopping"
        assert InstanceStatus.ERROR == "error"


class TestInstanceState:
    """Tests for InstanceState model."""

    def test_minimal_creation(self):
        """Test creating instance state with minimal fields."""
        state = InstanceState(model_id="test/model", port=8000)
        assert state.model_id == "test/model"
        assert state.port == 8000
        assert state.pid is None
        assert state.status == InstanceStatus.STOPPED
        assert state.last_request_time > 0  # Should be auto-set
        assert state.start_time is None
        assert state.memory_delta is None
        assert state.devices == []
        assert state.sleep_level is None

    def test_full_creation(self):
        """Test creating instance state with all fields."""
        state = InstanceState(
            model_id="test/model",
            port=8000,
            pid=12345,
            status=InstanceStatus.RUNNING,
            last_request_time=1234567890.0,
            start_time=1234567800.0,
            memory_delta=1024 * 1024 * 1024,  # 1GB
            devices=[0, 1],
            sleep_level=2,
        )
        assert state.model_id == "test/model"
        assert state.port == 8000
        assert state.pid == 12345
        assert state.status == InstanceStatus.RUNNING
        assert state.last_request_time == 1234567890.0
        assert state.start_time == 1234567800.0
        assert state.memory_delta == 1024 * 1024 * 1024
        assert state.devices == [0, 1]
        assert state.sleep_level == 2

    def test_last_request_time_default(self):
        """Test that last_request_time is set to current time by default."""
        before = time.time()
        state = InstanceState(model_id="test", port=8000)
        after = time.time()
        assert before <= state.last_request_time <= after


class TestStateManager:
    """Tests for StateManager."""

    def test_initialization(self):
        """Test state manager initialization."""
        manager = StateManager()
        assert manager.state.instances == {}
        assert manager.state.used_ports == set()
        assert manager.state.last_updated > 0

    def test_set_and_get_instance(self):
        """Test setting and getting instance state."""
        manager = StateManager()
        state = InstanceState(
            model_id="test/model",
            port=8000,
            status=InstanceStatus.RUNNING,
        )
        manager.set_instance("test/model", state)

        # Retrieve instance
        retrieved = manager.get_instance("test/model")
        assert retrieved is not None
        assert retrieved.model_id == "test/model"
        assert retrieved.port == 8000
        assert retrieved.status == InstanceStatus.RUNNING

    def test_get_nonexistent_instance(self):
        """Test getting non-existent instance."""
        manager = StateManager()
        result = manager.get_instance("nonexistent")
        assert result is None

    def test_remove_instance(self):
        """Test removing instance."""
        manager = StateManager()
        state = InstanceState(model_id="test/model", port=8000)
        manager.set_instance("test/model", state)

        # Verify it exists
        assert manager.get_instance("test/model") is not None
        assert 8000 in manager.get_used_ports()

        # Remove it
        manager.remove_instance("test/model")

        # Verify it's gone
        assert manager.get_instance("test/model") is None
        assert 8000 not in manager.get_used_ports()

    def test_remove_nonexistent_instance(self):
        """Test removing non-existent instance (should not error)."""
        manager = StateManager()
        manager.remove_instance("nonexistent")  # Should not raise

    def test_update_last_request_time(self):
        """Test updating last request time."""
        manager = StateManager()
        state = InstanceState(model_id="test/model", port=8000, last_request_time=1000.0)
        manager.set_instance("test/model", state)

        # Update time
        before = time.time()
        manager.update_last_request_time("test/model")
        after = time.time()

        # Verify update
        updated = manager.get_instance("test/model")
        assert before <= updated.last_request_time <= after

    def test_update_last_request_time_nonexistent(self):
        """Test updating last request time for non-existent instance."""
        manager = StateManager()
        manager.update_last_request_time("nonexistent")  # Should not raise

    def test_get_all_instances(self):
        """Test getting all instances."""
        manager = StateManager()

        # Add multiple instances
        manager.set_instance("model1", InstanceState(model_id="model1", port=8000))
        manager.set_instance("model2", InstanceState(model_id="model2", port=8001))
        manager.set_instance("model3", InstanceState(model_id="model3", port=8002))

        # Get all
        all_instances = manager.get_all_instances()
        assert len(all_instances) == 3
        assert "model1" in all_instances
        assert "model2" in all_instances
        assert "model3" in all_instances

    def test_get_all_instances_empty(self):
        """Test getting all instances when empty."""
        manager = StateManager()
        all_instances = manager.get_all_instances()
        assert all_instances == {}

    def test_get_used_ports(self):
        """Test getting used ports."""
        manager = StateManager()

        # Add instances with different ports
        manager.set_instance("model1", InstanceState(model_id="model1", port=8000))
        manager.set_instance("model2", InstanceState(model_id="model2", port=8001))

        used_ports = manager.get_used_ports()
        assert 8000 in used_ports
        assert 8001 in used_ports
        assert len(used_ports) == 2

    def test_clear_state(self):
        """Test clearing all state."""
        manager = StateManager()

        # Add some instances
        manager.set_instance("model1", InstanceState(model_id="model1", port=8000))
        manager.set_instance("model2", InstanceState(model_id="model2", port=8001))

        # Clear state
        manager.clear_state()

        # Verify everything is cleared
        assert manager.get_all_instances() == {}
        assert manager.get_used_ports() == set()

    def test_update_instance_status(self):
        """Test updating instance status."""
        manager = StateManager()
        state = InstanceState(model_id="test/model", port=8000, status=InstanceStatus.STARTING)
        manager.set_instance("test/model", state)

        # Update status
        manager.update_instance_status("test/model", InstanceStatus.RUNNING)

        # Verify update
        updated = manager.get_instance("test/model")
        assert updated.status == InstanceStatus.RUNNING

    def test_update_instance_status_nonexistent(self):
        """Test updating status for non-existent instance."""
        manager = StateManager()
        manager.update_instance_status("nonexistent", InstanceStatus.RUNNING)  # Should not raise

    def test_update_instance_memory_delta(self):
        """Test updating instance memory delta."""
        manager = StateManager()
        state = InstanceState(model_id="test/model", port=8000, memory_delta=None)
        manager.set_instance("test/model", state)

        # Update memory delta
        memory_delta = 1024 * 1024 * 1024  # 1GB
        manager.update_instance_memory_delta("test/model", memory_delta)

        # Verify update
        updated = manager.get_instance("test/model")
        assert updated.memory_delta == memory_delta

    def test_update_instance_memory_delta_nonexistent(self):
        """Test updating memory delta for non-existent instance."""
        manager = StateManager()
        manager.update_instance_memory_delta("nonexistent", 1024)  # Should not raise

    def test_state_isolation(self):
        """Test that different managers have isolated state."""
        manager1 = StateManager()
        manager2 = StateManager()

        manager1.set_instance("model1", InstanceState(model_id="model1", port=8000))

        # Manager2 should not see manager1's state
        assert manager2.get_instance("model1") is None

    def test_port_tracking_with_multiple_instances(self):
        """Test that ports are correctly tracked across multiple instances."""
        manager = StateManager()

        # Add instances
        manager.set_instance("model1", InstanceState(model_id="model1", port=8000))
        manager.set_instance("model2", InstanceState(model_id="model2", port=8001))

        # Check ports
        assert 8000 in manager.get_used_ports()
        assert 8001 in manager.get_used_ports()

        # Remove one
        manager.remove_instance("model1")

        # Check ports again
        assert 8000 not in manager.get_used_ports()
        assert 8001 in manager.get_used_ports()
