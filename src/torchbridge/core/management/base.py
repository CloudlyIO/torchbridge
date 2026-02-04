"""
Base classes and types for TorchBridge Management System.

This module provides the foundation for all managers:
- ManagerType, ManagerState enums
- ManagerContext dataclass
- BaseManager abstract class

Version: 0.3.11
"""

import gc
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from ..config import TorchBridgeConfig


class ManagerType(Enum):
    """Types of management domains."""
    HARDWARE = "hardware"
    OPTIMIZATION = "optimization"
    INFRASTRUCTURE = "infrastructure"


class ManagerState(Enum):
    """Manager lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ManagerContext:
    """Management context for coordination."""
    manager_id: str
    manager_type: ManagerType
    state: ManagerState
    device: torch.device
    config: TorchBridgeConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class BaseManager(ABC):
    """
    Unified base class for all managers.

    Consolidates common functionality from all manager classes.
    Provides lifecycle management, thread-safety, and status reporting.
    """

    def __init__(self, config: TorchBridgeConfig, context: ManagerContext | None = None):
        self.config = config
        self.context = context or ManagerContext(
            manager_id=self._generate_id(),
            manager_type=self._get_manager_type(),
            state=ManagerState.INITIALIZING,
            device=config.device,
            config=config
        )

        self._lock = threading.RLock()
        self._initialized = False
        self._active_operations: dict[str, Any] = {}

        self._initialize()

    @abstractmethod
    def _get_manager_type(self) -> ManagerType:
        """Get the manager type classification."""
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize manager-specific resources."""
        pass

    @abstractmethod
    def optimize(self, target: Any, **kwargs) -> Any:
        """Primary optimization/management operation."""
        pass

    def _generate_id(self) -> str:
        """Generate unique manager ID."""
        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    def get_status(self) -> dict[str, Any]:
        """Get current manager status."""
        return {
            "manager_id": self.context.manager_id,
            "type": self.context.manager_type.value,
            "state": self.context.state.value,
            "device": str(self.context.device),
            "active_operations": len(self._active_operations),
            "uptime": time.time() - self.context.created_at
        }

    def suspend(self) -> None:
        """Suspend manager operations."""
        with self._lock:
            self.context.state = ManagerState.SUSPENDED

    def resume(self) -> None:
        """Resume manager operations."""
        with self._lock:
            self.context.state = ManagerState.ACTIVE

    def shutdown(self) -> None:
        """Shutdown manager and cleanup resources."""
        with self._lock:
            self.context.state = ManagerState.SHUTDOWN
            self._cleanup()

    def _cleanup(self) -> None:
        """Cleanup manager resources."""
        self._active_operations.clear()
        gc.collect()
