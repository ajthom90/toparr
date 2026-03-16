from abc import ABC, abstractmethod
from typing import Optional


class GpuBackend(ABC):
    """Abstract base class for GPU monitoring backends."""

    @abstractmethod
    def discover_devices(self) -> list[dict]:
        """Discover available GPU devices.

        Returns list of dicts with keys: device, name, driver.
        The driver field is internal metadata.
        """

    @abstractmethod
    def read_sample(self, device: str) -> dict:
        """Read current GPU metrics for the given device.

        Returns a normalized sample dict with keys:
        period, frequency, gpu_busy, power, engines, clients.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clear internal state (delta counters, caches)."""
