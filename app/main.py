import asyncio
import json
import logging
import os
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
