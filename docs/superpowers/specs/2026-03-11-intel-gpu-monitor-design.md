# Intel GPU Monitor — Design Spec

## Purpose

A Docker container that runs `intel_gpu_top` and serves a live web dashboard showing Intel GPU metrics. Designed for monitoring hardware transcoding on TrueNAS systems running Plex and Tdarr.

## Target Hardware

- Intel UHD 730 (i3-13100) on TrueNAS
- GPU shared across Plex, Tdarr, and this monitoring container via `/dev/dri`

## Architecture

Single Docker container with three layers:

1. **`intel_gpu_top -J`** — runs as a subprocess, streams JSON to stdout every ~1 second
2. **FastAPI (Python)** — parses JSON stream, maintains ring buffer history, serves web UI and API
3. **Vanilla HTML/JS/CSS dashboard** — connects via SSE for real-time updates

### Docker Requirements

- `--device /dev/dri` — GPU access
- `--cap-add CAP_PERFMON` — permission to read GPU performance counters
- Single exposed port (default 8080)
- Base image: `python:3.12-slim` with `intel-gpu-tools` installed via apt

### API Endpoints

- `GET /` — serves the dashboard (static HTML/JS/CSS)
- `GET /api/stream` — SSE endpoint, pushes parsed GPU data every ~1s
- `GET /api/status` — JSON snapshot of current state + recent history

### Data Flow

1. Python subprocess spawns `intel_gpu_top -J -s 1000` (JSON output, 1s interval)
2. Background task reads stdout line by line, parses JSON
3. Each sample is stored in a fixed-size ring buffer of 300 entries (~5 minutes at 1s intervals). Not configurable via env var in v1.
4. Connected SSE clients receive each new sample as it arrives
5. Browser renders gauges, bars, sparklines from SSE events

### `intel_gpu_top -J` Output Format

Each JSON sample from `intel_gpu_top -J` has this structure:

```json
{
  "period": { "unit": "ms", "duration": 1000.0 },
  "frequency": { "unit": "MHz", "requested": 1350.0, "actual": 1300.0 },
  "interrupts": { "unit": "irq/s", "count": 1842.0 },
  "rc6": { "unit": "%", "value": 32.5 },
  "power": { "unit": "W", "GPU": 8.2, "Package": 45.2 },
  "imc-bandwidth": { "unit": "MB/s", "reads": 1024.0, "writes": 512.0 },
  "engines": {
    "Render/3D/0": { "unit": "%", "busy": 42.0, "sema": 0.0, "wait": 2.3 },
    "Video/0": { "unit": "%", "busy": 87.0, "sema": 0.0, "wait": 0.5 },
    "VideoEnhance/0": { "unit": "%", "busy": 65.0, "sema": 0.0, "wait": 0.0 },
    "Blitter/0": { "unit": "%", "busy": 12.0, "sema": 0.0, "wait": 0.0 }
  },
  "clients": {
    "1234": {
      "pid": "4821",
      "name": "Plex Transcoder",
      "engine-classes": {
        "Render/3D": { "busy": "38.0", "unit": "%" },
        "Video": { "busy": "72.0", "unit": "%" }
      }
    }
  }
}
```

Notes:
- Engine names vary by GPU generation. The parser treats them dynamically (iterate `engines` keys).
- Not all fields are present on all platforms. Missing metrics show as `"-"` or are absent. The parser handles both gracefully.
- The JSON stream outputs one object per sample. Objects may be comma-separated (v1.18+). The parser handles both formats.
- Client data requires `CAP_PERFMON` capability. Without it, the `clients` section may be empty.

### GPU Name Detection

The GPU name shown in the header is read once at startup from `/sys/class/drm/card0/device/product_name` or by parsing the output of `lspci | grep VGA`. Falls back to "Intel GPU" if neither is available.

### `/api/status` Response Schema

```json
{
  "gpu_name": "Intel UHD Graphics 730",
  "uptime_seconds": 123456,
  "connected_clients": 2,
  "current": { /* latest intel_gpu_top sample, same schema as above */ },
  "history": [ /* array of up to 300 recent samples, oldest first */ ]
}
```

### SSE Event Format

Events are sent on the `/api/stream` endpoint with event type `gpu_data`:

```
event: gpu_data
data: {"timestamp": 1710000000.0, "period": {...}, "frequency": {...}, ...}
```

The `data` field is the parsed `intel_gpu_top` sample as JSON with an added `timestamp` field (Unix epoch float). The browser parses this with `JSON.parse(event.data)`.

### Error Handling

**Subprocess failure:**
- If `intel_gpu_top` fails to start (missing `/dev/dri`, missing `CAP_PERFMON`), the backend logs the error and retries every 5 seconds.
- If it crashes mid-run, the backend detects the closed stdout pipe and restarts the subprocess after a 2-second delay.
- Malformed JSON lines are logged and skipped; they do not crash the parser.

**Dashboard behavior when GPU stream is unavailable:**
- The SSE connection stays open. The backend sends periodic `event: status` heartbeats (`data: {"status": "waiting", "error": "..."}`) so the browser knows the backend is alive but GPU data is unavailable.
- The dashboard shows a yellow "GPU Unavailable" banner with the error message, replacing live data with last-known values (greyed out) or "N/A" if no data has been received yet.

**SSE reconnection (browser side):**
- Uses the browser's native `EventSource` which auto-reconnects on disconnect.
- On reconnect, the client fetches `/api/status` once to backfill any missed history, then resumes streaming.
- During disconnection, the dashboard shows a "Reconnecting..." indicator in the header.

### Power Bar TDP Reference

The power bar shows wattage relative to the CPU package TDP. Default is 60W (i3-13100 PBP). Configurable via `GPU_TDP_WATTS` environment variable. If `power` data is unavailable from `intel_gpu_top`, the power card shows "N/A".

### iframe Embedding

- No `X-Frame-Options` header is set (allows embedding in any origin).
- Dashboard has a minimum usable width of ~320px.
- No authentication required.

## Dashboard Layout

### Header
- App title ("Intel GPU Monitor")
- GPU name / model
- Live connection status indicator

### Top Row — 3 cards
- **GPU Busy** — circular gauge with percentage, interrupts/s
- **Frequency** — current MHz, requested vs actual
- **Power** — wattage with bar relative to TDP

### Middle Row — 2 cards
- **Engine Utilization** — horizontal bars for Render, Video, VideoEnhance, Copy engines
- **History (5 min)** — SVG sparkline charts per engine

### Bottom — 1 card
- **Active Clients** — table showing PID, process name, per-engine utilization

### Footer
- Container uptime, refresh status

### Visual Style
- Dark theme (background #0f1419, cards #1a2332)
- Color-coded engines: Render=#60a5fa, Video=#34d399, VideoEnhance=#a855f7, Copy=#fbbf24
- Responsive — collapses to single column on narrow viewports
- iframe-friendly for embedding in Homarr or similar dashboards

## Tech Stack

- **Backend:** Python 3.12, FastAPI, uvicorn, sse-starlette
- **Frontend:** Vanilla HTML, CSS, JavaScript (no framework, no build step)
- **Container:** Debian-based (python:3.12-slim), apt install intel-gpu-tools
- **No database** — ring buffer in memory, resets on container restart

## File Structure

```
intel-gpu-top-docker/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py          # FastAPI app, SSE endpoint, subprocess management
│   ├── gpu_monitor.py   # intel_gpu_top parser, ring buffer, data models
│   └── static/
│       ├── index.html    # Dashboard page
│       ├── style.css     # Dashboard styles
│       └── app.js        # SSE client, rendering logic
└── docs/
```

## Non-Goals

- Long-term data storage or Prometheus export (out of scope for v1)
- Multiple GPU support
- Authentication/authorization
- GPU temperature (not available via intel_gpu_top)
