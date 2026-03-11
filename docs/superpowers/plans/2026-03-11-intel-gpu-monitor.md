# Intel GPU Monitor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker container that runs `intel_gpu_top` and serves a live web dashboard showing GPU metrics via SSE.

**Architecture:** Single Docker container with three layers: `intel_gpu_top -J` subprocess streaming JSON, FastAPI backend parsing and broadcasting via SSE, vanilla HTML/JS/CSS dashboard consuming the stream. Ring buffer holds 300 samples (~5 min) in memory.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, sse-starlette, pytest, vanilla HTML/CSS/JS, Docker

**Spec:** `docs/superpowers/specs/2026-03-11-intel-gpu-monitor-design.md`

---

## File Structure

```
intel-gpu-top-docker/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py            # FastAPI app, lifespan, routes, SSE endpoint
│   ├── gpu_monitor.py     # GpuMonitor class: subprocess, parser, ring buffer, GPU name detection
│   └── static/
│       ├── index.html      # Dashboard HTML
│       ├── style.css       # Dashboard styles
│       └── app.js          # SSE client, gauge/bar/sparkline rendering
└── tests/
    ├── conftest.py         # Shared fixtures (sample JSON data)
    ├── test_gpu_monitor.py # Unit tests for parser, ring buffer, GPU name
    └── test_api.py         # Integration tests for FastAPI endpoints
```

---

## Chunk 1: Backend Core

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `app/__init__.py`

- [ ] **Step 1: Create requirements.txt and pytest.ini**

`requirements.txt`:
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sse-starlette==2.2.1
pytest==8.3.4
httpx==0.28.1
pytest-asyncio==0.25.0
```

`pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 2: Create test scaffolding**

Create `tests/__init__.py` (empty) and `tests/conftest.py` with sample `intel_gpu_top` JSON data as a fixture:

```python
import pytest


SAMPLE_GPU_JSON = {
    "period": {"unit": "ms", "duration": 1000.0},
    "frequency": {"unit": "MHz", "requested": 1350.0, "actual": 1300.0},
    "interrupts": {"unit": "irq/s", "count": 1842.0},
    "rc6": {"unit": "%", "value": 32.5},
    "power": {"unit": "W", "GPU": 8.2, "Package": 45.2},
    "imc-bandwidth": {"unit": "MB/s", "reads": 1024.0, "writes": 512.0},
    "engines": {
        "Render/3D/0": {"unit": "%", "busy": 42.0, "sema": 0.0, "wait": 2.3},
        "Video/0": {"unit": "%", "busy": 87.0, "sema": 0.0, "wait": 0.5},
        "VideoEnhance/0": {"unit": "%", "busy": 65.0, "sema": 0.0, "wait": 0.0},
        "Blitter/0": {"unit": "%", "busy": 12.0, "sema": 0.0, "wait": 0.0},
    },
    "clients": {
        "1234": {
            "pid": "4821",
            "name": "Plex Transcoder",
            "engine-classes": {
                "Render/3D": {"busy": "38.0", "unit": "%"},
                "Video": {"busy": "72.0", "unit": "%"},
            },
        }
    },
}

SAMPLE_GPU_JSON_MINIMAL = {
    "period": {"unit": "ms", "duration": 1000.0},
    "frequency": {"unit": "MHz", "requested": 300.0, "actual": 300.0},
    "interrupts": {"unit": "irq/s", "count": 0.0},
    "rc6": {"unit": "%", "value": 98.0},
    "engines": {
        "Render/3D/0": {"unit": "%", "busy": 0.0, "sema": 0.0, "wait": 0.0},
    },
    "clients": {},
}


@pytest.fixture
def sample_gpu_json():
    return SAMPLE_GPU_JSON.copy()


@pytest.fixture
def sample_gpu_json_minimal():
    return SAMPLE_GPU_JSON_MINIMAL.copy()
```

- [ ] **Step 3: Create app/__init__.py**

Empty file.

- [ ] **Step 4: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Run: `pytest tests/ -v`
Expected: no tests collected, no errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini tests/ app/__init__.py
git commit -m "chore: project scaffolding with dependencies and test fixtures"
```

---

### Task 2: GPU monitor — ring buffer and parser

**Files:**
- Create: `app/gpu_monitor.py`
- Create: `tests/test_gpu_monitor.py`

- [ ] **Step 1: Write failing tests for ring buffer and JSON parsing**

```python
import json
import time
from unittest.mock import patch

from tests.conftest import SAMPLE_GPU_JSON, SAMPLE_GPU_JSON_MINIMAL

from app.gpu_monitor import GpuMonitor


class TestRingBuffer:
    def test_add_sample_and_get_current(self):
        monitor = GpuMonitor(buffer_size=300)
        monitor.add_sample(SAMPLE_GPU_JSON)
        current = monitor.get_current()
        assert current is not None
        assert "timestamp" in current
        assert current["frequency"]["requested"] == 1350.0

    def test_buffer_respects_max_size(self):
        monitor = GpuMonitor(buffer_size=5)
        for i in range(10):
            sample = SAMPLE_GPU_JSON.copy()
            sample["interrupts"] = {"unit": "irq/s", "count": float(i)}
            monitor.add_sample(sample)
        history = monitor.get_history()
        assert len(history) == 5
        # Oldest should be sample 5 (0-4 evicted)
        assert history[0]["interrupts"]["count"] == 5.0

    def test_get_current_when_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        assert monitor.get_current() is None

    def test_get_history_when_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        assert monitor.get_history() == []


class TestJsonParsing:
    def test_parse_full_sample(self):
        monitor = GpuMonitor(buffer_size=300)
        line = json.dumps(SAMPLE_GPU_JSON)
        result = monitor.parse_line(line)
        assert result is not None
        assert result["engines"]["Video/0"]["busy"] == 87.0

    def test_parse_minimal_sample(self):
        monitor = GpuMonitor(buffer_size=300)
        line = json.dumps(SAMPLE_GPU_JSON_MINIMAL)
        result = monitor.parse_line(line)
        assert result is not None
        assert "power" not in result

    def test_parse_malformed_json_returns_none(self):
        monitor = GpuMonitor(buffer_size=300)
        assert monitor.parse_line("not json at all") is None
        assert monitor.parse_line("{incomplete") is None

    def test_parse_strips_leading_comma(self):
        """intel_gpu_top v1.18+ may prefix lines with commas."""
        monitor = GpuMonitor(buffer_size=300)
        line = "," + json.dumps(SAMPLE_GPU_JSON)
        result = monitor.parse_line(line)
        assert result is not None
        assert result["frequency"]["requested"] == 1350.0

    def test_parse_strips_brackets(self):
        """First/last lines may have [ or ] characters."""
        monitor = GpuMonitor(buffer_size=300)
        line = "[" + json.dumps(SAMPLE_GPU_JSON)
        result = monitor.parse_line(line)
        assert result is not None


class TestGpuName:
    @patch(
        "app.gpu_monitor.open",
        side_effect=lambda *a, **kw: __import__("io").StringIO(
            "Intel UHD Graphics 730\n"
        ),
    )
    def test_detect_gpu_name_from_sysfs(self, mock_open):
        name = GpuMonitor.detect_gpu_name()
        assert "730" in name

    @patch("app.gpu_monitor.open", side_effect=FileNotFoundError)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_fallback_gpu_name(self, mock_run, mock_open):
        name = GpuMonitor.detect_gpu_name()
        assert name == "Intel GPU"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpu_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.gpu_monitor'`

- [ ] **Step 3: Implement GpuMonitor class**

Create `app/gpu_monitor.py`:

```python
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
        self._buffer: deque[dict] = deque(maxlen=buffer_size)
        self._current: Optional[dict] = None
        self._subscribers: list[asyncio.Queue] = []
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

    def get_history(self) -> list[dict]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_monitor.py -v`
Expected: all 11 tests PASS (4 ring buffer + 5 parsing + 2 GPU name).

- [ ] **Step 5: Commit**

```bash
git add app/gpu_monitor.py tests/test_gpu_monitor.py
git commit -m "feat: GpuMonitor with JSON parser, ring buffer, and GPU name detection"
```

---

### Task 3: FastAPI app with SSE and status endpoints

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for API endpoints**

```python
import pytest
from httpx import AsyncClient, ASGITransport

from tests.conftest import SAMPLE_GPU_JSON


@pytest.fixture
def monitor():
    """Create a GpuMonitor with sample data, without starting subprocess."""
    from app.gpu_monitor import GpuMonitor
    import app.main

    m = GpuMonitor(buffer_size=300)
    m.gpu_name = "Intel UHD Graphics 730"
    m.add_sample(SAMPLE_GPU_JSON.copy())
    app.main.monitor = m
    yield m


@pytest.mark.asyncio
async def test_status_endpoint(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gpu_name"] == "Intel UHD Graphics 730"
    assert data["current"] is not None
    assert data["current"]["frequency"]["requested"] == 1350.0
    assert "uptime_seconds" in data
    assert "history" in data
    assert len(data["history"]) == 1
    assert "tdp_watts" in data


@pytest.mark.asyncio
async def test_status_endpoint_empty():
    from app.gpu_monitor import GpuMonitor
    import app.main

    app.main.monitor = GpuMonitor(buffer_size=300)
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current"] is None
    assert data["history"] == []


@pytest.mark.asyncio
async def test_index_serves_html():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement FastAPI app**

Create `app/main.py`:

```python
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
                    continue

                if isinstance(data, tuple) and data[0] == "status":
                    yield {"event": "status", "data": json.dumps(data[1])}
                else:
                    yield {"event": "gpu_data", "data": json.dumps(data)}
        finally:
            monitor.unsubscribe(queue)

    return EventSourceResponse(event_generator())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

- [ ] **Step 4: Create a minimal index.html so tests pass**

Create `app/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Intel GPU Monitor</title></head>
<body><h1>Intel GPU Monitor</h1></body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: all 14 tests PASS (11 gpu_monitor + 3 api).

- [ ] **Step 7: Commit**

```bash
git add app/main.py app/static/index.html tests/test_api.py
git commit -m "feat: FastAPI app with SSE streaming, status endpoint, and static file serving"
```

---

## Chunk 2: Frontend Dashboard

### Task 4: Dashboard HTML and CSS

**Files:**
- Modify: `app/static/index.html`
- Create: `app/static/style.css`

- [ ] **Step 1: Write the full dashboard HTML**

Replace `app/static/index.html` with the complete dashboard markup. This follows the mockup from the design spec — header, 3 gauge cards, engine utilization bars, sparkline history, clients table, footer. All elements have `id` attributes for JS to target.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Intel GPU Monitor</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <!-- Header -->
  <div class="header">
    <div>
      <h1>Intel GPU Monitor</h1>
      <div class="gpu-name" id="gpu-name">Detecting GPU...</div>
    </div>
    <div class="status" id="connection-status">
      <div class="status-dot"></div>
      <span>Connecting...</span>
    </div>
  </div>

  <!-- Error banner (hidden by default) -->
  <div class="error-banner" id="error-banner" style="display:none;">
    <span id="error-message"></span>
  </div>

  <!-- Top row: gauges -->
  <div class="grid-3">
    <div class="card">
      <div class="card-title">GPU Busy (RC6 inverse)</div>
      <div class="gauge-container">
        <div class="gauge">
          <svg viewBox="0 0 80 80">
            <circle class="gauge-bg" cx="40" cy="40" r="34"/>
            <circle class="gauge-fill" id="gpu-busy-arc" cx="40" cy="40" r="34"
              stroke-dasharray="213.6" stroke-dashoffset="213.6"/>
          </svg>
          <div class="gauge-text" id="gpu-busy-pct">--%</div>
        </div>
        <div class="gauge-details">
          <div class="label">Interrupts/s</div>
          <div class="value" id="interrupts-val">--</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Frequency</div>
      <div class="freq-display">
        <span class="freq-current" id="freq-actual">--</span>
        <span class="freq-unit">MHz</span>
        <div class="freq-sub">
          <div class="item">
            <div class="val" id="freq-requested">--</div>
            <div class="lbl">Requested</div>
          </div>
          <div class="item">
            <div class="val" id="freq-actual-sub">--</div>
            <div class="lbl">Actual</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Power</div>
      <div class="power-display">
        <span class="power-current" id="power-gpu">--</span>
        <span class="power-unit">W</span>
        <div class="power-bar-container">
          <div class="power-bar-track">
            <div class="power-bar-fill" id="power-bar"></div>
          </div>
          <div class="power-bar-labels">
            <span>0W</span>
            <span id="power-tdp-label">--W TDP</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Middle row: engines + sparklines -->
  <div class="grid-2">
    <div class="card">
      <div class="section-label">Engine Utilization</div>
      <div id="engine-bars">
        <div class="placeholder-text">Waiting for data...</div>
      </div>
    </div>

    <div class="card">
      <div class="section-label">History (5 min)</div>
      <div id="sparklines">
        <div class="placeholder-text">Waiting for data...</div>
      </div>
    </div>
  </div>

  <!-- Bottom: clients table -->
  <div class="card">
    <div class="section-label">Active Clients</div>
    <table class="clients-table">
      <thead id="clients-head">
        <tr>
          <th>PID</th>
          <th>Name</th>
        </tr>
      </thead>
      <tbody id="clients-body">
        <tr><td colspan="2" class="placeholder-text">No active clients</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Footer -->
  <div class="footer">
    Intel GPU Monitor — <span id="footer-status">Connecting...</span> — Uptime: <span id="uptime">--</span>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the CSS**

Create `app/static/style.css` with the dark theme from the design mockup:

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: #0f1419;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  padding: 16px;
  min-width: 320px;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: #1a2332;
  border-radius: 10px;
  margin-bottom: 16px;
}
.header h1 { font-size: 18px; font-weight: 600; color: #60a5fa; }
.gpu-name { font-size: 13px; color: #888; }
.status { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #34d399; }
.status-dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; }
.status.disconnected { color: #ef4444; }
.status.disconnected .status-dot { background: #ef4444; }
.status.warning { color: #fbbf24; }
.status.warning .status-dot { background: #fbbf24; }

.error-banner {
  background: rgba(251, 191, 36, 0.15);
  border: 1px solid #fbbf24;
  color: #fbbf24;
  padding: 10px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 13px;
}

.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }

.card { background: #1a2332; border-radius: 10px; padding: 16px; }
.card-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 12px; }
.section-label { font-size: 13px; font-weight: 600; color: #ccc; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #2a3544; }

.gauge-container { display: flex; align-items: center; gap: 16px; }
.gauge { position: relative; width: 80px; height: 80px; }
.gauge svg { transform: rotate(-90deg); }
.gauge-bg { fill: none; stroke: #2a3544; stroke-width: 8; }
.gauge-fill { fill: none; stroke: #60a5fa; stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 0.5s; }
.gauge-text { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; color: #60a5fa; }
.gauge-details { flex: 1; }
.gauge-details .label { font-size: 11px; color: #666; margin-bottom: 2px; }
.gauge-details .value { font-size: 14px; font-weight: 600; }

.freq-display { text-align: center; }
.freq-current { font-size: 36px; font-weight: 700; color: #60a5fa; }
.freq-unit { font-size: 14px; color: #666; }
.freq-sub { display: flex; justify-content: center; gap: 24px; margin-top: 8px; }
.freq-sub .item { text-align: center; }
.freq-sub .val { font-size: 14px; font-weight: 600; }
.freq-sub .lbl { font-size: 10px; color: #666; text-transform: uppercase; }

.power-display { text-align: center; }
.power-current { font-size: 36px; font-weight: 700; color: #fbbf24; }
.power-unit { font-size: 14px; color: #666; }
.power-bar-container { margin-top: 8px; }
.power-bar-track { height: 6px; background: #2a3544; border-radius: 3px; overflow: hidden; }
.power-bar-fill { height: 100%; background: #fbbf24; border-radius: 3px; transition: width 0.5s; width: 0%; }
.power-bar-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 10px; color: #666; }

.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-row:last-child { margin-bottom: 0; }
.bar-label { width: 90px; font-size: 12px; color: #888; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; height: 20px; background: #2a3544; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
.bar-value { width: 45px; font-size: 13px; font-weight: 600; text-align: right; }

.sparkline-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.sparkline-row:last-child { margin-bottom: 0; }
.sparkline-label { width: 60px; font-size: 11px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sparkline { flex: 1; height: 30px; }
.sparkline-val { width: 40px; font-size: 12px; font-weight: 600; text-align: right; }

.clients-table { width: 100%; font-size: 12px; border-collapse: collapse; }
.clients-table th { text-align: left; color: #666; font-weight: 500; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; padding: 6px 8px; border-bottom: 1px solid #2a3544; }
.clients-table td { padding: 8px; border-bottom: 1px solid #1e2d3d; }
.clients-table tr:last-child td { border-bottom: none; }

.placeholder-text { color: #555; font-size: 13px; font-style: italic; padding: 8px 0; }

.footer { text-align: center; font-size: 11px; color: #444; margin-top: 12px; }

@media (max-width: 900px) {
  .grid-3 { grid-template-columns: 1fr; }
  .grid-2 { grid-template-columns: 1fr; }
}
```

- [ ] **Step 3: Verify HTML test still passes**

Run: `pytest tests/test_api.py::test_index_serves_html -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html app/static/style.css
git commit -m "feat: dashboard HTML and CSS with dark theme, responsive layout"
```

---

### Task 5: Dashboard JavaScript (SSE client and rendering)

**Files:**
- Create: `app/static/app.js`

- [ ] **Step 1: Write the SSE client and rendering logic**

Create `app/static/app.js`:

```javascript
(function () {
  "use strict";

  // --- Config ---
  var HISTORY_MAX = 300;
  var ENGINE_COLORS = {
    "Render/3D": "#60a5fa",
    "Video": "#34d399",
    "VideoEnhance": "#a855f7",
    "Blitter": "#fbbf24",
  };

  // --- State ---
  var history = [];
  var engineNames = [];
  var tdpWatts = 60;
  var gpuName = "";
  var startTime = Date.now();
  var connected = false;

  // --- DOM refs ---
  function $(id) { return document.getElementById(id); }

  // --- Init ---
  async function init() {
    await fetchStatus();
    connectSSE();
  }

  // --- Fetch initial status ---
  async function fetchStatus() {
    try {
      var resp = await fetch("/api/status");
      var data = await resp.json();
      gpuName = data.gpu_name || "Intel GPU";
      $("gpu-name").textContent = gpuName;
      startTime = Date.now() - data.uptime_seconds * 1000;
      if (data.tdp_watts) tdpWatts = data.tdp_watts;

      if (data.history && data.history.length > 0) {
        history = data.history.slice(-HISTORY_MAX);
        discoverEngines(history[history.length - 1]);
        render(history[history.length - 1]);
      }
    } catch (e) {
      console.error("Failed to fetch status:", e);
    }
  }

  // --- SSE ---
  function connectSSE() {
    var eventSource = new EventSource("/api/stream");
    var hasConnectedBefore = false;

    eventSource.addEventListener("gpu_data", function (e) {
      connected = true;
      setConnectionStatus("live");
      hideError();
      var data = JSON.parse(e.data);
      history.push(data);
      if (history.length > HISTORY_MAX) history.shift();
      discoverEngines(data);
      render(data);
    });

    eventSource.addEventListener("status", function (e) {
      var data = JSON.parse(e.data);
      if (data.status === "waiting") {
        setConnectionStatus("warning", "GPU Unavailable");
        showError(data.error || "Waiting for GPU data...");
      }
    });

    eventSource.addEventListener("open", function () {
      connected = true;
      setConnectionStatus("live");
      // On reconnect (not first connect), backfill missed history
      if (hasConnectedBefore) {
        fetchStatus();
      }
      hasConnectedBefore = true;
    });

    eventSource.addEventListener("error", function () {
      connected = false;
      setConnectionStatus("disconnected", "Reconnecting...");
      // EventSource auto-reconnects; backfill happens in the open handler
    });
  }

  // --- Connection status ---
  function setConnectionStatus(state, text) {
    var el = $("connection-status");
    el.className = "status";
    if (state === "live") {
      el.querySelector("span").textContent = "Live \u2014 updating every 1s";
    } else if (state === "warning") {
      el.classList.add("warning");
      el.querySelector("span").textContent = text || "Warning";
    } else if (state === "disconnected") {
      el.classList.add("disconnected");
      el.querySelector("span").textContent = text || "Disconnected";
    }
  }

  function showError(msg) {
    $("error-banner").style.display = "block";
    $("error-message").textContent = msg;
  }

  function hideError() {
    $("error-banner").style.display = "none";
  }

  // --- Discover engines ---
  function discoverEngines(sample) {
    if (!sample || !sample.engines) return;
    var names = Object.keys(sample.engines).map(function (k) {
      return k.replace(/\/\d+$/, "");
    });
    if (JSON.stringify(names) !== JSON.stringify(engineNames)) {
      engineNames = names;
    }
  }

  function engineColor(name) {
    if (ENGINE_COLORS[name]) return ENGINE_COLORS[name];
    for (var key in ENGINE_COLORS) {
      if (name.startsWith(key)) return ENGINE_COLORS[key];
    }
    return "#888";
  }

  // --- Render ---
  function render(sample) {
    if (!sample) return;
    renderGpuBusy(sample);
    renderFrequency(sample);
    renderPower(sample);
    renderEngineBars(sample);
    renderSparklines();
    renderClients(sample);
    renderFooter();
  }

  function renderGpuBusy(sample) {
    var rc6 = sample.rc6 ? sample.rc6.value : 0;
    var busy = Math.max(0, Math.min(100, 100 - rc6));
    var circumference = 213.6;
    var offset = circumference - (busy / 100) * circumference;
    $("gpu-busy-arc").style.strokeDashoffset = offset;
    $("gpu-busy-pct").textContent = busy.toFixed(0) + "%";

    var irq = sample.interrupts ? sample.interrupts.count : 0;
    $("interrupts-val").textContent = Math.round(irq).toLocaleString();
  }

  function renderFrequency(sample) {
    var freq = sample.frequency;
    if (!freq) return;
    $("freq-actual").textContent = Math.round(freq.actual);
    $("freq-requested").textContent = Math.round(freq.requested);
    $("freq-actual-sub").textContent = Math.round(freq.actual);
  }

  function renderPower(sample) {
    var power = sample.power;
    if (!power) {
      $("power-gpu").textContent = "N/A";
      $("power-bar").style.width = "0%";
      return;
    }
    var gpu = power.GPU || 0;
    $("power-gpu").textContent = gpu.toFixed(1);
    var pct = Math.min(100, (gpu / tdpWatts) * 100);
    $("power-bar").style.width = pct + "%";
    $("power-tdp-label").textContent = tdpWatts + "W TDP";
  }

  function renderEngineBars(sample) {
    if (!sample.engines) return;
    var container = $("engine-bars");
    var html = "";
    for (var i = 0; i < engineNames.length; i++) {
      var name = engineNames[i];
      var key = null;
      var keys = Object.keys(sample.engines);
      for (var j = 0; j < keys.length; j++) {
        if (keys[j].startsWith(name)) { key = keys[j]; break; }
      }
      var busy = key ? sample.engines[key].busy : 0;
      var color = engineColor(name);
      var shortName = name.replace("/3D", "");
      html += '<div class="bar-row">' +
        '<div class="bar-label">' + shortName + '</div>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + busy + '%;background:' + color + '"></div></div>' +
        '<div class="bar-value" style="color:' + color + '">' + busy.toFixed(0) + '%</div>' +
        '</div>';
    }
    container.innerHTML = html;
  }

  function renderSparklines() {
    if (engineNames.length === 0) return;
    var container = $("sparklines");
    var html = "";
    for (var i = 0; i < engineNames.length; i++) {
      var name = engineNames[i];
      var color = engineColor(name);
      var shortName = name.replace("/3D", "");
      var values = history.map(function (s) {
        if (!s.engines) return 0;
        var keys = Object.keys(s.engines);
        for (var j = 0; j < keys.length; j++) {
          if (keys[j].startsWith(name)) return s.engines[keys[j]].busy;
        }
        return 0;
      });
      var current = values.length > 0 ? values[values.length - 1] : 0;
      var points = sparklinePoints(values, 300, 30);
      html += '<div class="sparkline-row">' +
        '<div class="sparkline-label" style="color:' + color + '">' + shortName + '</div>' +
        '<div class="sparkline"><svg width="100%" height="30" preserveAspectRatio="none" viewBox="0 0 300 30">' +
        '<polyline fill="none" stroke="' + color + '" stroke-width="1.5" points="' + points + '"/>' +
        '</svg></div>' +
        '<div class="sparkline-val" style="color:' + color + '">' + current.toFixed(0) + '%</div>' +
        '</div>';
    }
    container.innerHTML = html;
  }

  function sparklinePoints(values, width, height) {
    if (values.length === 0) return "";
    var step = width / Math.max(values.length - 1, 1);
    var parts = [];
    for (var i = 0; i < values.length; i++) {
      var x = i * step;
      var y = height - (values[i] / 100) * height;
      parts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    return parts.join(" ");
  }

  function renderClients(sample) {
    var tbody = $("clients-body");
    var thead = $("clients-head");
    if (!sample.clients || Object.keys(sample.clients).length === 0) {
      thead.innerHTML = '<tr><th>PID</th><th>Name</th></tr>';
      tbody.innerHTML = '<tr><td colspan="2" class="placeholder-text">No active clients</td></tr>';
      return;
    }

    var engineClasses = {};
    var clients = Object.values(sample.clients);
    for (var i = 0; i < clients.length; i++) {
      var ec = clients[i]["engine-classes"];
      if (ec) {
        var ecKeys = Object.keys(ec);
        for (var j = 0; j < ecKeys.length; j++) {
          engineClasses[ecKeys[j]] = true;
        }
      }
    }
    var ecList = Object.keys(engineClasses).sort();

    var headerHtml = '<tr><th>PID</th><th>Name</th>';
    for (var i = 0; i < ecList.length; i++) {
      headerHtml += '<th>' + ecList[i].replace("/3D", "") + '</th>';
    }
    headerHtml += '</tr>';
    thead.innerHTML = headerHtml;

    var rows = "";
    for (var i = 0; i < clients.length; i++) {
      var client = clients[i];
      var name = client.name || "unknown";
      var pid = client.pid || "?";
      rows += '<tr><td>' + pid + '</td><td style="color:#60a5fa">' + name + '</td>';
      for (var j = 0; j < ecList.length; j++) {
        var data = client["engine-classes"] ? client["engine-classes"][ecList[j]] : null;
        var busy = data ? parseFloat(data.busy) : 0;
        rows += '<td>' + busy.toFixed(0) + '%</td>';
      }
      rows += '</tr>';
    }
    tbody.innerHTML = rows;
  }

  function renderFooter() {
    var elapsed = (Date.now() - startTime) / 1000;
    $("uptime").textContent = formatUptime(elapsed);
    $("footer-status").textContent = connected
      ? "Refreshing via SSE every ~1s"
      : "Disconnected";
  }

  function formatUptime(seconds) {
    var d = Math.floor(seconds / 86400);
    var h = Math.floor((seconds % 86400) / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return d + "d " + h + "h " + m + "m";
    if (h > 0) return h + "h " + m + "m";
    return m + "m";
  }

  init();
})();
```

- [ ] **Step 2: Verify all tests still pass**

Run: `pytest tests/ -v`
Expected: all 14 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add app/static/app.js
git commit -m "feat: dashboard JavaScript with SSE client, gauges, sparklines, and client table"
```

---

## Chunk 3: Docker and Final Integration

### Task 6: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Usage:
#   docker build -t intel-gpu-monitor .
#   docker run -d \
#     --name intel-gpu-monitor \
#     --device /dev/dri \
#     --cap-add CAP_PERFMON \
#     -p 8080:8080 \
#     -e GPU_TDP_WATTS=60 \
#     intel-gpu-monitor
#
# Then open http://<host-ip>:8080

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends intel-gpu-tools pciutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: Dockerfile with intel-gpu-tools and uvicorn entrypoint"
```

---

### Task 7: Run all tests and verify Docker build

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all 14 tests PASS.

- [ ] **Step 2: Verify Docker build works (syntax check — no Intel GPU on this Mac)**

Run: `docker build --platform linux/amd64 -t intel-gpu-monitor .`
Expected: build completes successfully. (Container won't run on Mac, but the image builds.)

- [ ] **Step 3: Fix any issues and commit if needed**

If any tests failed or the Docker build had issues, fix and commit.
