import asyncio
import logging
import time
from collections import deque
from typing import Optional

from app.backends.base import GpuBackend

logger = logging.getLogger(__name__)


class GpuMonitor:
    def __init__(
        self,
        buffer_size: int = 300,
        device: Optional[str] = None,
        backend: Optional[GpuBackend] = None,
    ):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._current: Optional[dict] = None
        self._subscribers: list = []
        self._error: Optional[str] = None
        self._device: Optional[str] = device
        self._available_gpus: list[dict] = []
        self._backend = backend
        self.gpu_name: str = "GPU"
        self._start_time: float = time.time()

    @staticmethod
    def _read_cmdline(pid: str) -> Optional[str]:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
            if not raw:
                return None
            return raw.replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None

    def _enrich_clients(self, data: dict) -> None:
        clients = data.get("clients")
        if not clients:
            return
        for client in clients.values():
            pid = client.get("pid")
            if pid:
                client["cmdline"] = self._read_cmdline(pid)

    def discover_gpus(self) -> list[dict]:
        if self._backend:
            self._available_gpus = self._backend.discover_devices()
            return [
                {"device": g["device"], "name": g["name"]}
                for g in self._available_gpus
            ]
        return []

    @property
    def current_device(self) -> Optional[str]:
        return self._device

    @property
    def available_gpus(self) -> list[dict]:
        return [
            {"device": g["device"], "name": g["name"]}
            for g in self._available_gpus
        ]

    async def select_device(self, device: Optional[str]) -> None:
        self._device = device
        self._buffer.clear()
        self._current = None
        self._error = None
        if self._backend:
            self._backend.cleanup()
        for gpu in self._available_gpus:
            if gpu["device"] == device:
                self.gpu_name = gpu["name"]
                break

    def add_sample(self, data: dict) -> None:
        data["timestamp"] = time.time()
        self._enrich_clients(data)
        self._current = data
        self._buffer.append(data)
        for queue in self._subscribers:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def get_current(self) -> Optional[dict]:
        return self._current

    def get_history(self) -> list:
        return list(self._buffer)

    def get_error(self) -> Optional[str]:
        return self._error

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    async def run(self) -> None:
        if not self._backend:
            self._error = "No GPU backend available"
            logger.error(self._error)
            return

        while True:
            try:
                device = self._device
                if not device and self._available_gpus:
                    device = self._available_gpus[0]["device"]
                if not device:
                    self._error = "No GPU device found"
                    await self._broadcast_status("waiting", self._error)
                    await asyncio.sleep(5)
                    continue

                sample = await asyncio.to_thread(
                    self._backend.read_sample, device
                )
                self._error = None
                self.add_sample(sample)
            except Exception as e:
                self._error = str(e)
                logger.error("GPU read error: %s. Retrying in 5s...", e)
                await self._broadcast_status("waiting", str(e))
            await asyncio.sleep(1)

    async def _broadcast_status(self, status: str, error: str) -> None:
        msg = {"status": status, "error": error}
        for queue in self._subscribers:
            try:
                queue.put_nowait(("status", msg))
            except asyncio.QueueFull:
                pass
