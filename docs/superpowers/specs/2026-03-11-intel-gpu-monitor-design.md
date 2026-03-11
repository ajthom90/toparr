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
3. Each sample is stored in a ring buffer (~300 entries = ~5 minutes)
4. Connected SSE clients receive each new sample as it arrives
5. Browser renders gauges, bars, sparklines from SSE events

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
