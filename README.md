# Toparr

[![Tests](https://github.com/ajthom90/toparr/actions/workflows/test.yml/badge.svg)](https://github.com/ajthom90/toparr/actions/workflows/test.yml)
[![Docker](https://github.com/ajthom90/toparr/actions/workflows/docker.yml/badge.svg)](https://github.com/ajthom90/toparr/actions/workflows/docker.yml)

Real-time Intel GPU monitoring dashboard in Docker. Wraps `intel_gpu_top` (from [igt-gpu-tools](https://gitlab.freedesktop.org/drm/igt-gpu-tools) v2.3) in a web UI with live-updating gauges, sparkline history, and per-process GPU client tracking.

![Toparr Dashboard](screenshots/dashboard.png)

## Features

- **Live GPU metrics** — utilization, frequency, power draw, interrupts, and RC6 residency
- **Engine utilization** — per-engine bars (Render/3D, Blitter, Video, VideoEnhance) with 5-minute sparkline history
- **Per-process client tracking** — see which processes (Plex, Tdarr, Jellyfin, etc.) are using the GPU and how much of each engine they consume
- **SSE streaming** — updates push to the browser every second with no polling
- **Dark theme dashboard** — clean, responsive UI

## Tested Setup

This application has been developed and tested on the following specific configuration:

| Component | Details |
|---|---|
| **CPU** | Intel Core i3-13100 (Alder Lake) |
| **GPU** | Intel UHD Graphics 730 (integrated) |
| **Host OS** | TrueNAS SCALE |
| **Kernel** | Linux 6.12.x (production+truenas) |
| **Driver** | i915 |

> **Disclaimer:** This software is provided as-is and may not work on all hardware configurations, kernels, or Linux distributions. In particular:
>
> - **Only Intel integrated GPUs using the i915 driver are supported.** Discrete Intel Arc GPUs (using the xe driver) have not been tested and may require modifications.
> - **Kernel compatibility varies.** Different kernel versions may have different debugfs layouts, fdinfo formats, or perf counter access methods. The per-client tracking feature requires kernel 5.19+ with fdinfo support.
> - **Container runtime differences** between Docker, Podman, and other runtimes may affect device access, capability handling, or PID namespace behavior.
> - **NAS and appliance OSes** (TrueNAS, Unraid, Synology, etc.) may have custom kernels with non-standard module configurations.
>
> If you encounter issues on a different setup, please open an issue with your hardware details, kernel version (`uname -r`), and GPU info (`lspci | grep VGA`). Contributions to support additional configurations are welcome!

## Requirements

- Linux host with an Intel GPU (see [Tested Setup](#tested-setup) above)
- Docker and Docker Compose
- Kernel 4.16+ (for i915 perf support); kernel 5.19+ recommended for per-client fdinfo

## Quick Start

Using the pre-built image from GitHub Container Registry (no clone needed):

```yaml
services:
  toparr:
    image: ghcr.io/ajthom90/toparr:latest
    container_name: toparr
    restart: unless-stopped
    pid: host
    devices:
      - /dev/dri:/dev/dri
    cap_add:
      - CAP_PERFMON
      - SYS_ADMIN
      - SYS_PTRACE
    volumes:
      - /sys/kernel/debug:/sys/kernel/debug:ro
    ports:
      - "8080:8080"
    environment:
      - GPU_TDP_WATTS=60
```

```bash
docker compose up -d
```

Open **http://localhost:8080** in your browser.

### Image tags

| Tag | Description |
|---|---|
| `latest` | Latest build from `main` branch |
| `1.0.0` | Specific release version |
| `1.0` | Latest patch within a minor version |
| `1` | Latest minor/patch within a major version (not created for `v0.x` releases) |

### Building from source

Alternatively, clone and build locally:

```bash
git clone https://github.com/ajthom90/toparr.git
cd toparr
docker compose up -d --build
```

## Docker Compose

```yaml
services:
  toparr:
    image: ghcr.io/ajthom90/toparr:latest
    # Or build from source:
    # build: .
    container_name: toparr
    restart: unless-stopped
    pid: host
    devices:
      - /dev/dri:/dev/dri
    cap_add:
      - CAP_PERFMON
      - SYS_ADMIN
      - SYS_PTRACE
    volumes:
      - /sys/kernel/debug:/sys/kernel/debug:ro
    ports:
      - "8080:8080"
    environment:
      - GPU_TDP_WATTS=60
```

### Required settings explained

| Setting | Why |
|---|---|
| `devices: /dev/dri` | Access to the GPU DRM device |
| `cap_add: CAP_PERFMON` | Read GPU performance counters |
| `cap_add: SYS_ADMIN` | Access debugfs for GPU client enumeration |
| `cap_add: SYS_PTRACE` | Read `/proc/<pid>/fdinfo` for processes outside the container (needed for per-client GPU usage) |
| `pid: host` | See host PIDs so `intel_gpu_top` can match GPU clients to processes |
| `volumes: /sys/kernel/debug` | Mount debugfs for DRI client discovery |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GPU_TDP_WATTS` | `60` | GPU TDP in watts (used for the power gauge scale) |

## Architecture

```
Browser  <──SSE──>  FastAPI (uvicorn)  <──stdout──>  intel_gpu_top -J
                         │
                    gpu_monitor.py
                    (JSON parser with
                     brace-depth tracking)
```

- **`intel_gpu_top -J -s 1000`** outputs pretty-printed JSON to stdout every second
- **`gpu_monitor.py`** reads stdout line-by-line, accumulates lines using brace-depth tracking to detect complete JSON objects, parses them, and broadcasts to SSE subscribers
- **`main.py`** serves the FastAPI app with SSE streaming at `/api/stream` and a status API at `/api/status`
- **`app.js`** connects via SSE, renders gauges (canvas), sparklines (SVG), engine bars, and the client table

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard UI |
| `GET /api/status` | Current GPU state, history buffer, and metadata |
| `GET /api/stream` | SSE stream of `gpu_data` events (JSON) |

## Development

```bash
pip install -r requirements.txt
pytest
```

Tests run without a real GPU — they test JSON parsing, sample buffering, and the API layer with mocked data. CI runs the test suite on Python 3.11, 3.12, and 3.13 for every push and pull request.

## Contributing

Contributions are welcome! This project has only been tested on one specific hardware/OS combination (see [Tested Setup](#tested-setup)), so there are many opportunities to help:

- **Hardware support** — test on different Intel GPUs (Arc, older generations, different Alder/Raptor Lake SKUs) and report or fix compatibility issues
- **Driver support** — add support for the `xe` driver used by Intel Arc discrete GPUs
- **Kernel compatibility** — test on different kernel versions and distributions, fix any debugfs/fdinfo format differences
- **Container runtimes** — verify and fix behavior on Podman, LXC, or other runtimes
- **NAS platforms** — test on Unraid, Synology, or other NAS distributions

### How to contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run the tests (`pytest tests/ -v`) — all tests must pass
5. Open a pull request with details about your setup and what you changed

When reporting issues, please include:
- Hardware: CPU and GPU model
- OS and kernel version (`uname -a`)
- GPU driver in use (`lspci -k | grep -A2 VGA`)
- Docker version (`docker --version`)
- Container logs (`docker logs toparr`)

## Troubleshooting

**Dashboard shows "Connecting..." with no data**
- Verify the container can access the GPU: `docker exec toparr intel_gpu_top -J -s 1000 -n 2`
- Check container logs: `docker logs toparr`

**No clients in the Active Clients table**
- Ensure `pid: host` is set (the container must see host PIDs)
- Ensure `SYS_PTRACE` capability is added (needed to read fdinfo of other processes)
- Ensure `SYS_ADMIN` is added and debugfs is mounted
- Your kernel must be 5.19+ for the fdinfo-based per-client tracking

**Permission denied errors**
- `CAP_PERFMON` is required for GPU perf counters
- `SYS_ADMIN` + debugfs mount is required for client enumeration
- `SYS_PTRACE` is required to read process fdinfo across container boundaries

## License

[MIT](LICENSE)
