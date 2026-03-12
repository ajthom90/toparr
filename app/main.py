import asyncio
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.gpu_monitor import GpuMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tdp = int(os.environ.get("GPU_TDP_WATTS", "60"))
monitor = GpuMonitor(buffer_size=300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitor.run())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def status():
    return {
        "gpu_name": monitor.gpu_name,
        "uptime_seconds": monitor.uptime_seconds,
        "connected_clients": len(monitor._subscribers),
        "current": monitor.get_current(),
        "history": monitor.get_history(),
        "tdp_watts": tdp,
    }


@app.get("/api/debug")
async def debug():
    current = monitor.get_current()

    # Check container permissions relevant to per-client tracking
    checks = {}
    checks["pid_namespace"] = "host" if os.path.exists("/proc/1/cmdline") else "container"
    try:
        with open("/proc/1/cmdline", "rb") as f:
            checks["pid1_cmdline"] = f.read().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
    except (FileNotFoundError, PermissionError) as e:
        checks["pid1_cmdline"] = str(e)

    # Check if we can read fdinfo for any process
    fdinfo_sample = None
    try:
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            fdinfo_path = f"/proc/{pid_dir}/fdinfo"
            if os.path.isdir(fdinfo_path):
                try:
                    entries = os.listdir(fdinfo_path)
                    fdinfo_sample = {
                        "pid": pid_dir,
                        "fdinfo_count": len(entries),
                        "readable": True,
                    }
                    break
                except PermissionError:
                    fdinfo_sample = {
                        "pid": pid_dir,
                        "readable": False,
                        "error": "PermissionError",
                    }
                    break
    except Exception as e:
        fdinfo_sample = {"error": str(e)}
    checks["fdinfo_access"] = fdinfo_sample

    # Check capabilities
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cap"):
                    key = line.split(":")[0].strip()
                    checks[key] = line.split(":")[1].strip()
    except (FileNotFoundError, PermissionError):
        checks["capabilities"] = "unreadable"

    # Check intel_gpu_top version
    try:
        result = subprocess.run(
            ["intel_gpu_top", "--help"],
            capture_output=True, text=True, timeout=5,
        )
        checks["intel_gpu_top_help"] = (
            result.stdout[:500] + result.stderr[:500]
        ).strip()
    except Exception as e:
        checks["intel_gpu_top_help"] = str(e)

    return {
        "gpu_name": monitor.gpu_name,
        "uptime_seconds": monitor.uptime_seconds,
        "error": monitor.get_error(),
        "raw_lines": list(monitor._raw_lines)[-5:],
        "has_clients": bool(
            current and current.get("clients")
            and len(current["clients"]) > 0
        ),
        "current_clients": current.get("clients", {}) if current else {},
        "current_sample_keys": list(current.keys()) if current else [],
        "buffer_size": len(monitor._buffer),
        "subscriber_count": len(monitor._subscribers),
        "environment_checks": checks,
    }


@app.get("/api/stream")
async def stream(request: Request):
    queue = monitor.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    error = monitor.get_error()
                    if error:
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {"status": "waiting", "error": error}
                            ),
                        }
                    else:
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {"status": "starting"}
                            ),
                        }
                    continue

                if isinstance(data, tuple) and data[0] == "status":
                    yield {"event": "status", "data": json.dumps(data[1])}
                else:
                    yield {"event": "gpu_data", "data": json.dumps(data)}
        finally:
            monitor.unsubscribe(queue)

    return EventSourceResponse(event_generator())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
