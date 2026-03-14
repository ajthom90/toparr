# Sysfs/fdinfo GPU Backend — Design Spec

**Date:** 2026-03-14
**Status:** Approved
**Goal:** Replace `intel_gpu_top` dependency with direct sysfs/fdinfo reads, supporting both i915 and xe Intel GPU drivers. Architect for future NVIDIA/AMD backend extensibility.

## Motivation

Intel Arc discrete GPUs (Battlemage, etc.) use the `xe` kernel driver. `intel_gpu_top` only supports the `i915` driver, so Toparr doesn't work on these GPUs. Rather than finding another external binary, we read the same kernel interfaces that tools like nvtop use internally — sysfs, hwmon, and DRM fdinfo.

## Data Model

The backend produces a normalized, vendor-agnostic sample dict:

```python
{
    "period": {"duration": 1000.0},                         # sampling interval ms
    "frequency": {"actual": 1300.0, "requested": 1350.0},  # MHz
    "gpu_busy": 67.5,                                       # % (RC6 or engine agg)
    "power": {"gpu": 8.2},                                  # watts (None if unavailable)
    "engines": {
        "Render/3D/0": {"busy": 42.0},
        "Video/0": {"busy": 87.0},
        "VideoEnhance/0": {"busy": 65.0},
        "Blitter/0": {"busy": 12.0},
    },
    "clients": {
        "42": {
            "pid": "4821",
            "name": "Plex Transcoder",
            "engine-classes": {
                "Render/3D": {"busy": "38.0"},
                "Video": {"busy": "72.0"},
            },
            "memory": {
                "system": {"total": "232411136", "resident": "122638336", ...}
            }
        }
    }
}
```

### Changes from current intel_gpu_top format

- `rc6` and `interrupts` fields removed — replaced by top-level `gpu_busy` percentage
- `imc-bandwidth` removed (i915-PMU-specific, no sysfs equivalent)
- `power.Package` removed (only `power.gpu` kept)
- Engine `sema`/`wait` fields removed (fdinfo only provides busy time)
- xe engine names mapped to familiar names: `rcs` -> `Render/3D`, `vcs` -> `Video`, `vecs` -> `VideoEnhance`, `bcs` -> `Blitter`, `ccs` -> `Compute`

## Architecture

Three layers:

```
+---------------------------------------------+
|  GpuMonitor (orchestrator)                   |
|  - 1s async sample loop                      |
|  - ring buffer, SSE broadcast                |
|  - cmdline enrichment                        |
|  - backend auto-detection                    |
+---------------------------------------------+
|  GpuBackend (ABC)                            |
|  - discover_devices() -> list[dict]          |
|  - read_sample(device) -> dict               |
|  - cleanup()                                 |
+-------------+-------------+-----------------+
|  Intel      |  Nvidia     |  AMD            |
|  Backend    |  Backend    |  Backend        |
|  (i915+xe)  |  (future)   |  (future)       |
+-------------+-------------+-----------------+
```

### GpuBackend ABC

Three methods:

- `discover_devices()` — returns list of `{"device": "card0", "name": "Intel Arc B580", "driver": "xe"}`
- `read_sample(device) -> dict` — returns one normalized sample dict
- `cleanup()` — release resources on shutdown

### IntelBackend

Handles both i915 and xe drivers internally.

**Device discovery:** Enumerate `/dev/dri/card*`, check vendor ID `0x8086` via `/sys/class/drm/cardN/device/vendor`.

**Driver detection:** Read `/sys/class/drm/cardN/device/driver` symlink basename -> `i915` or `xe`.

**GPU name detection:** Read `/sys/class/drm/cardN/device/product_name` (preferred) or fall back to `lspci`.

**Frequency:**
- i915: read `gt_cur_freq_mhz` and `gt_max_freq_mhz` from `/sys/class/drm/cardN/`
- xe: read `cur_freq` and `max_freq` from `/sys/class/drm/cardN/device/tile0/gt0/freq0/`

**RC6 / GPU busy:**
- i915: read `gt/gt0/rc6_residency_ms` from sysfs, compute delta over interval -> RC6%, then `gpu_busy = 100 - rc6%`
- xe: attempt similar sysfs path; if unavailable, derive from aggregate engine utilization (max engine busy%)

**Power:**
- Find hwmon device associated with the GPU PCI device
- Read `energy1_input` (microjoules), compute delta over interval -> watts
- Available primarily on discrete Intel GPUs; returns None for integrated GPUs without hwmon

**Per-process engine utilization and memory (fdinfo):**
1. Scan `/proc/*/fdinfo/*` for lines starting with `drm-`
2. For i915: parse `drm-engine-<name>: <ns> ns` (time-based counters)
3. For xe: parse `drm-cycles-<name>` and `drm-total-cycles-<name>` (cycle-based counters)
4. Deduplicate by `drm-client-id`
5. Compute deltas from previous scan:
   - i915: `utilization = delta(engine_ns) / delta(wall_time_ns) * 100`
   - xe: `utilization = delta(cycles) / delta(total_cycles) * 100`
6. Parse memory keys: `drm-total-<region>`, `drm-shared-<region>`, `drm-resident-<region>`, `drm-active-<region>`, `drm-purgeable-<region>`

**Engine name mapping for xe:**
| xe name | Display name |
|---------|-------------|
| `rcs`   | `Render/3D` |
| `vcs`   | `Video`     |
| `vecs`  | `VideoEnhance` |
| `bcs`   | `Blitter`   |
| `ccs`   | `Compute`   |

**Device-level engine utilization:**
Aggregate per-process engine busy% by summing across all clients per engine class. Cap at 100%.

### GpuMonitor (updated)

- On startup: auto-detect backend (try Intel, future: Nvidia, AMD)
- Calls `backend.discover_devices()` to populate GPU list
- Async loop: `backend.read_sample()` every 1s -> `add_sample()` -> broadcast to SSE subscribers
- Retains: ring buffer, subscriber management, cmdline enrichment, error handling, device selection
- API layer (`main.py`) barely changes — still talks to GpuMonitor with same interface

### Frontend changes (minimal)

- Read `sample.gpu_busy` directly instead of computing `100 - sample.rc6.value`
- Hide or remove interrupts display
- Card title changes from "GPU Busy (RC6 inverse)" to "GPU Busy"
- Everything else (engines, clients, sparklines, modal, GPU selector) works unchanged since data shape is preserved
- HTML title/header: "GPU Monitor" instead of "Intel GPU Monitor"

### Dockerfile changes

- Remove the igt-gpu-tools build stage entirely (no more multi-stage build)
- Single stage: Python 3.12-slim with app code
- Keep: pciutils (for lspci fallback in GPU name detection)
- No external binary dependency

## fdinfo Parsing Detail

**Scan algorithm (runs every ~1s):**
1. List `/proc/` for numeric directories (PIDs)
2. For each PID, list `/proc/<pid>/fdinfo/` entries
3. Read each fdinfo file, look for `drm-driver:` line matching target driver
4. If match: extract client-id, engine counters, memory counters
5. Deduplicate: keep highest counters per `drm-client-id` (multiple fds possible)
6. Compare with previous scan's counters to compute delta -> utilization %
7. Look up process name from `/proc/<pid>/comm`, cmdline from `/proc/<pid>/cmdline`

**Performance:** Only processes with open DRM device fds produce matching fdinfo entries — typically a handful. Early filtering (skip non-numeric dirs, skip fds without `drm-` lines) keeps this fast.

**Permissions (unchanged from current):**
- `CAP_PERFMON` or `CAP_SYS_ADMIN` to read other processes' fdinfo
- `pid: host` in Docker to see host PIDs
- `SYS_PTRACE` for cmdline reading

**Thin read helpers for testability:**
- `_read_sysfs(path) -> str` — single file read, easy to mock
- `_scan_fdinfo() -> list[dict]` — full fdinfo scan, mockable for tests
- `_read_hwmon(device) -> dict` — hwmon values, mockable

## Testing Strategy

### Before refactoring
Write tests capturing the current behavioral contract — API responses, data shapes, GpuMonitor processing.

### Test layers

1. **IntelBackend unit tests** (new):
   - Parse i915 fdinfo text -> engine counters
   - Parse xe fdinfo text -> engine counters
   - Compute utilization deltas correctly (both time-based and cycle-based)
   - Read frequency from mock sysfs paths
   - Read power from mock hwmon paths
   - Engine name mapping (xe rcs -> Render/3D)
   - Device discovery with mock sysfs
   - Handle missing/unreadable files gracefully

2. **GpuMonitor tests** (existing + updated):
   - Ring buffer behavior — unchanged
   - Sample enrichment (cmdline) — unchanged
   - Device selection — updated to use backend
   - Error handling when backend fails or returns None

3. **API tests** (existing + updated):
   - `/api/status` returns normalized sample format
   - `/api/gpus` works with backend-based discovery
   - `/api/stream` SSE behavior — unchanged
   - New data model fields validated

4. **Integration-style tests** (new):
   - Full pipeline: mock fdinfo files -> IntelBackend.read_sample() -> normalized dict with correct structure
   - Both i915 and xe driver variants

All sysfs/fdinfo reads are behind thin helper methods, making mocking straightforward — no real GPU hardware needed in CI.

## Metrics Removed

These metrics from the old intel_gpu_top JSON format are intentionally dropped:

- **IMC bandwidth** (reads/writes MB/s) — i915 PMU specific, no sysfs equivalent
- **Interrupts/s** — would require parsing /proc/interrupts per-GPU, non-trivial and low value
- **RC6 residency** (as a separate field) — replaced by computed `gpu_busy` percentage
- **Power.Package** — i915 PMU specific
- **Engine sema/wait** — fdinfo only provides busy time

## File Structure

```
app/
  gpu_monitor.py      # GpuMonitor orchestrator (updated)
  backends/
    __init__.py
    base.py           # GpuBackend ABC
    intel.py           # IntelBackend (i915 + xe)
  main.py              # API layer (minor updates)
  static/              # Frontend (minor updates)
tests/
  conftest.py          # Updated sample data
  test_gpu_monitor.py  # Updated monitor tests
  test_api.py          # Updated API tests
  test_intel_backend.py # New backend tests
```
