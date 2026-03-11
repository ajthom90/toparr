import asyncio
import json
import logging
import subprocess
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

SYSFS_GPU_NAME = "/sys/class/drm/card0/device/product_name"


class GpuMonitor:
    def __init__(self, buffer_size: int = 300):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._current: Optional[dict] = None
        self._subscribers: list = []
        self._process: Optional[asyncio.subprocess.Process] = None
        self._error: Optional[str] = None
        self.gpu_name: str = self.detect_gpu_name()
        self._start_time: float = time.time()

    @staticmethod
    def detect_gpu_name() -> str:
        try:
            with open(SYSFS_GPU_NAME) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            pass
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "VGA" in line and "Intel" in line:
                    parts = line.split(": ", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "Intel GPU"

    def parse_line(self, line: str) -> Optional[dict]:
        line = line.strip()
        if not line:
            return None
        line = line.lstrip("[,").rstrip("],")
        if not line:
            return None
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "period" in data:
                return data
            return None
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSON line: %s", line[:100])
            return None

    def add_sample(self, data: dict) -> None:
        data["timestamp"] = time.time()
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
        while True:
            try:
                await self._run_gpu_top()
            except Exception as e:
                self._error = str(e)
                logger.error("intel_gpu_top error: %s. Retrying in 5s...", e)
                await self._broadcast_status("waiting", str(e))
                await asyncio.sleep(5)

    async def _run_gpu_top(self) -> None:
        logger.info("Starting intel_gpu_top...")
        self._process = await asyncio.create_subprocess_exec(
            "intel_gpu_top", "-J", "-s", "1000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._error = None
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                parsed = self.parse_line(line.decode("utf-8", errors="replace"))
                if parsed:
                    self.add_sample(parsed)
        finally:
            await self._process.wait()
            stderr = ""
            if self._process.stderr:
                stderr_bytes = await self._process.stderr.read()
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = self._process.returncode
            self._error = (
                f"intel_gpu_top exited (code={returncode}): {stderr.strip()}"
            )
            logger.warning(self._error)
            await self._broadcast_status("waiting", self._error)
            await asyncio.sleep(2)

    async def _broadcast_status(self, status: str, error: str) -> None:
        msg = {"status": status, "error": error}
        for queue in self._subscribers:
            try:
                queue.put_nowait(("status", msg))
            except asyncio.QueueFull:
                pass
