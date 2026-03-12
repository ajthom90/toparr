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
        self._raw_lines: deque = deque(maxlen=50)
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

    def parse_json(self, text: str) -> Optional[dict]:
        text = text.strip().lstrip("[,").rstrip("],")
        if not text:
            return None
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "period" in data:
                return data
            return None
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON chunk: %s", text[:100])
            return None

    # Keep for test compatibility
    def parse_line(self, line: str) -> Optional[dict]:
        return self.parse_json(line)

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
        self._stderr_lines: list[str] = []
        self._process = await asyncio.create_subprocess_exec(
            "intel_gpu_top", "-J", "-l", "-s", "1000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def drain_stderr():
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.warning("intel_gpu_top stderr: %s", text)
                    self._stderr_lines.append(text)
                    # Update error immediately so SSE handler can see it
                    self._error = f"intel_gpu_top stderr: {text}"

        stderr_task = asyncio.create_task(drain_stderr())
        json_buf = []
        brace_depth = 0
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    stderr = "\n".join(self._stderr_lines[-10:])
                    msg = (
                        f"intel_gpu_top produced no output for 10s. "
                        f"stderr: {stderr.strip()}" if stderr.strip()
                        else "intel_gpu_top produced no output for 10s"
                    )
                    logger.error(msg)
                    self._error = msg
                    await self._broadcast_status("waiting", msg)
                    self._process.kill()
                    break
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                self._raw_lines.append(decoded.strip()[:500])
                # Track brace depth to detect complete JSON objects
                for ch in decoded:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                if json_buf or '{' in decoded:
                    json_buf.append(decoded)
                if brace_depth == 0 and json_buf:
                    chunk = "".join(json_buf)
                    json_buf = []
                    parsed = self.parse_json(chunk)
                    if parsed:
                        self._error = None
                        self.add_sample(parsed)
        finally:
            await self._process.wait()
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
            stderr = "\n".join(self._stderr_lines[-10:])
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
