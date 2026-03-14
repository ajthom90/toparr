# Sysfs/fdinfo GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `intel_gpu_top` with direct sysfs/fdinfo reads to support both i915 and xe Intel GPU drivers, with pluggable backend architecture.

**Architecture:** Three-layer design: GpuBackend ABC defines the interface, IntelBackend implements sysfs/fdinfo reading for both i915 and xe drivers, GpuMonitor orchestrates sampling and SSE broadcasting. Backend methods are synchronous, called via `asyncio.to_thread()`.

**Tech Stack:** Python 3.12, FastAPI, pytest, asyncio. No external GPU binaries.

**Spec:** `docs/superpowers/specs/2026-03-14-sysfs-fdinfo-gpu-backend-design.md`

---

## Chunk 1: Foundation — Current System Tests + Backend ABC

### Task 1: Write behavioral contract tests for current system

Capture the behavioral contract that must survive the refactor: API response shapes, data flow, SSE behavior.

**Files:**
- Modify: `tests/test_api.py`
- Read: `tests/conftest.py`, `app/main.py`

- [ ] **Step 1: Write test for /api/gpus endpoint**

```python
# Add to tests/test_api.py

@pytest.mark.asyncio
async def test_gpus_endpoint(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpus" in data
    assert "current_device" in data
    assert isinstance(data["gpus"], list)
```

- [ ] **Step 2: Write test for /api/debug endpoint**

```python
# Add to tests/test_api.py

@pytest.mark.asyncio
async def test_debug_endpoint(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/debug")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu_name" in data
    assert "uptime_seconds" in data
    assert "error" in data
    assert "buffer_size" in data
```

- [ ] **Step 3: Write test verifying status response data shape**

```python
# Add to tests/test_api.py

@pytest.mark.asyncio
async def test_status_response_has_expected_fields(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    data = resp.json()
    current = data["current"]
    assert current is not None
    # Verify engines exist
    assert "engines" in current
    assert len(current["engines"]) > 0
    # Verify clients structure
    assert "clients" in current
    # Verify frequency
    assert "frequency" in current
    assert "actual" in current["frequency"]
    assert "requested" in current["frequency"]
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "test: add behavioral contract tests before GPU backend refactor"
```

### Task 2: Create GpuBackend ABC

**Files:**
- Create: `app/backends/__init__.py`
- Create: `app/backends/base.py`
- Create: `tests/test_backend_base.py`

- [ ] **Step 1: Write test for GpuBackend ABC**

```python
# tests/test_backend_base.py
import pytest
from app.backends.base import GpuBackend


def test_gpu_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        GpuBackend()


def test_gpu_backend_subclass_must_implement_methods():
    class IncompleteBackend(GpuBackend):
        pass

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_gpu_backend_subclass_works_when_complete():
    class DummyBackend(GpuBackend):
        def discover_devices(self):
            return [{"device": "card0", "name": "Test GPU", "driver": "test"}]

        def read_sample(self, device):
            return {"gpu_busy": 50.0}

        def cleanup(self):
            pass

    backend = DummyBackend()
    devices = backend.discover_devices()
    assert len(devices) == 1
    assert devices[0]["device"] == "card0"
    sample = backend.read_sample("card0")
    assert sample["gpu_busy"] == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_backend_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backends'`

- [ ] **Step 3: Create the backends package and GpuBackend ABC**

```python
# app/backends/__init__.py
from app.backends.base import GpuBackend

__all__ = ["GpuBackend"]
```

```python
# app/backends/base.py
from abc import ABC, abstractmethod
from typing import Optional


class GpuBackend(ABC):
    """Abstract base class for GPU monitoring backends."""

    @abstractmethod
    def discover_devices(self) -> list[dict]:
        """Discover available GPU devices.

        Returns list of dicts with keys: device, name, driver.
        The driver field is internal metadata.
        """

    @abstractmethod
    def read_sample(self, device: str) -> dict:
        """Read current GPU metrics for the given device.

        Returns a normalized sample dict with keys:
        period, frequency, gpu_busy, power, engines, clients.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clear internal state (delta counters, caches)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_backend_base.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/backends/__init__.py app/backends/base.py tests/test_backend_base.py
git commit -m "feat: add GpuBackend ABC for pluggable GPU monitoring backends"
```

---

## Chunk 2: IntelBackend — Sysfs Reads

### Task 3: IntelBackend device discovery

**Files:**
- Create: `app/backends/intel.py`
- Create: `tests/test_intel_backend.py`

- [ ] **Step 1: Write tests for device discovery**

```python
# tests/test_intel_backend.py
import os
from unittest.mock import patch, MagicMock
from app.backends.intel import IntelBackend


class TestDeviceDiscovery:
    def test_discover_intel_gpu(self, tmp_path):
        """Discovers a GPU when vendor is 0x8086."""
        card_dir = tmp_path / "card0" / "device"
        card_dir.mkdir(parents=True)
        (card_dir / "vendor").write_text("0x8086\n")
        (card_dir / "product_name").write_text("Intel UHD Graphics 730\n")
        # Create driver symlink
        driver_target = tmp_path / "drivers" / "i915"
        driver_target.mkdir(parents=True)
        os.symlink(str(driver_target), str(card_dir / "driver"))

        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert len(devices) == 1
        assert devices[0]["device"] == "card0"
        assert devices[0]["name"] == "Intel UHD Graphics 730"
        assert devices[0]["driver"] == "i915"

    def test_discover_skips_non_intel(self, tmp_path):
        """Skips GPUs with non-Intel vendor IDs."""
        card_dir = tmp_path / "card0" / "device"
        card_dir.mkdir(parents=True)
        (card_dir / "vendor").write_text("0x10de\n")  # NVIDIA

        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert len(devices) == 0

    def test_discover_xe_driver(self, tmp_path):
        """Detects xe driver for Arc GPUs."""
        card_dir = tmp_path / "card0" / "device"
        card_dir.mkdir(parents=True)
        (card_dir / "vendor").write_text("0x8086\n")
        (card_dir / "product_name").write_text("Intel Arc B580\n")
        driver_target = tmp_path / "drivers" / "xe"
        driver_target.mkdir(parents=True)
        os.symlink(str(driver_target), str(card_dir / "driver"))

        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert len(devices) == 1
        assert devices[0]["driver"] == "xe"

    def test_discover_no_product_name_fallback(self, tmp_path):
        """Falls back to 'Intel GPU' when product_name is missing."""
        card_dir = tmp_path / "card0" / "device"
        card_dir.mkdir(parents=True)
        (card_dir / "vendor").write_text("0x8086\n")
        driver_target = tmp_path / "drivers" / "i915"
        driver_target.mkdir(parents=True)
        os.symlink(str(driver_target), str(card_dir / "driver"))

        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert len(devices) == 1
        assert devices[0]["name"] == "Intel GPU"

    def test_discover_empty_drm(self, tmp_path):
        """Returns empty list when no DRM cards found."""
        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert devices == []

    def test_discover_skips_render_nodes(self, tmp_path):
        """Only discovers cardN devices, not renderDN."""
        render_dir = tmp_path / "renderD128" / "device"
        render_dir.mkdir(parents=True)
        (render_dir / "vendor").write_text("0x8086\n")

        backend = IntelBackend()
        with patch.object(backend, "_drm_base", str(tmp_path)):
            devices = backend.discover_devices()

        assert devices == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestDeviceDiscovery -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.backends.intel'`

- [ ] **Step 3: Implement IntelBackend with device discovery**

```python
# app/backends/intel.py
import logging
import os
import subprocess
import time
from typing import Optional

from app.backends.base import GpuBackend

logger = logging.getLogger(__name__)

INTEL_VENDOR_ID = "0x8086"


class IntelBackend(GpuBackend):
    """GPU backend for Intel GPUs using sysfs and fdinfo.

    Supports both i915 and xe kernel drivers.
    """

    XE_ENGINE_MAP = {
        "rcs": "Render/3D",
        "vcs": "Video",
        "vecs": "VideoEnhance",
        "bcs": "Blitter",
        "ccs": "Compute",
    }

    I915_ENGINE_MAP = {
        "render": "Render/3D",
        "video": "Video",
        "video-enhance": "VideoEnhance",
        "copy": "Blitter",
    }

    def __init__(self):
        self._drm_base = "/sys/class/drm"
        self._proc_path = "/proc"
        self._prev_counters: dict = {}
        self._prev_time: float = 0.0
        self._prev_rc6_ms: Optional[float] = None
        self._prev_energy_uj: Optional[float] = None
        self._driver_cache: dict[str, str] = {}

    def _read_sysfs(self, path: str) -> Optional[str]:
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            return None

    def _detect_driver(self, card: str) -> Optional[str]:
        if card in self._driver_cache:
            return self._driver_cache[card]
        link = os.path.join(self._drm_base, card, "device", "driver")
        try:
            driver = os.path.basename(os.readlink(link))
            self._driver_cache[card] = driver
            return driver
        except (FileNotFoundError, OSError):
            return None

    def _detect_gpu_name(self, card: str) -> str:
        name = self._read_sysfs(
            os.path.join(self._drm_base, card, "device", "product_name")
        )
        if name:
            return name
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

    def discover_devices(self) -> list[dict]:
        devices = []
        try:
            entries = sorted(os.listdir(self._drm_base))
        except (FileNotFoundError, OSError):
            return devices

        for entry in entries:
            if not entry.startswith("card"):
                continue
            # Skip renderD nodes
            if not entry[4:].isdigit():
                continue
            vendor = self._read_sysfs(
                os.path.join(self._drm_base, entry, "device", "vendor")
            )
            if not vendor or vendor.strip() != INTEL_VENDOR_ID:
                continue
            driver = self._detect_driver(entry)
            if not driver:
                continue
            name = self._detect_gpu_name(entry)
            devices.append({
                "device": entry,
                "name": name,
                "driver": driver,
            })

        return devices

    def read_sample(self, device: str) -> dict:
        raise NotImplementedError("Will be implemented in Task 6")

    def cleanup(self) -> None:
        self._prev_counters.clear()
        self._prev_time = 0.0
        self._prev_rc6_ms = None
        self._prev_energy_uj = None
        self._driver_cache.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestDeviceDiscovery -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: add IntelBackend with device discovery for i915 and xe"
```

### Task 4: Sysfs reads — frequency, RC6, power

**Files:**
- Modify: `app/backends/intel.py`
- Modify: `tests/test_intel_backend.py`

- [ ] **Step 1: Write tests for frequency reading**

```python
# Add to tests/test_intel_backend.py

class TestFrequencyReading:
    def test_i915_frequency(self, tmp_path):
        """Reads frequency from i915 sysfs paths."""
        card_dir = tmp_path / "card0"
        card_dir.mkdir()
        (card_dir / "gt_act_freq_mhz").write_text("1300\n")
        (card_dir / "gt_cur_freq_mhz").write_text("1350\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        freq = backend._read_frequency("card0", "i915")
        assert freq == {"actual": 1300.0, "requested": 1350.0}

    def test_xe_frequency(self, tmp_path):
        """Reads frequency from xe sysfs paths."""
        freq_dir = tmp_path / "card0" / "device" / "tile0" / "gt0" / "freq0"
        freq_dir.mkdir(parents=True)
        (freq_dir / "act_freq").write_text("2100\n")
        (freq_dir / "cur_freq").write_text("2400\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        freq = backend._read_frequency("card0", "xe")
        assert freq == {"actual": 2100.0, "requested": 2400.0}

    def test_missing_frequency_returns_zeros(self, tmp_path):
        """Returns zeros when sysfs files are missing."""
        (tmp_path / "card0").mkdir()

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        freq = backend._read_frequency("card0", "i915")
        assert freq == {"actual": 0.0, "requested": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestFrequencyReading -v`
Expected: FAIL — `AttributeError: 'IntelBackend' object has no attribute '_read_frequency'`

- [ ] **Step 3: Implement _read_frequency**

Add to `app/backends/intel.py` in the `IntelBackend` class:

```python
    def _read_frequency(self, card: str, driver: str) -> dict:
        if driver == "xe":
            base = os.path.join(
                self._drm_base, card, "device", "tile0", "gt0", "freq0"
            )
            actual = self._read_sysfs(os.path.join(base, "act_freq"))
            requested = self._read_sysfs(os.path.join(base, "cur_freq"))
        else:
            base = os.path.join(self._drm_base, card)
            actual = self._read_sysfs(os.path.join(base, "gt_act_freq_mhz"))
            requested = self._read_sysfs(os.path.join(base, "gt_cur_freq_mhz"))
        return {
            "actual": float(actual) if actual else 0.0,
            "requested": float(requested) if requested else 0.0,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestFrequencyReading -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Write tests for RC6 / GPU busy reading**

```python
# Add to tests/test_intel_backend.py

class TestRc6Reading:
    def test_i915_rc6(self, tmp_path):
        """Reads RC6 residency from i915 sysfs path."""
        rc6_dir = tmp_path / "card0" / "gt" / "gt0"
        rc6_dir.mkdir(parents=True)
        (rc6_dir / "rc6_residency_ms").write_text("50000\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_rc6_ms("card0", "i915")
        assert val == 50000.0

    def test_xe_idle_residency(self, tmp_path):
        """Reads idle residency from xe sysfs path."""
        idle_dir = (
            tmp_path / "card0" / "device" / "tile0" / "gt0" / "gtidle"
        )
        idle_dir.mkdir(parents=True)
        (idle_dir / "idle_residency_ms").write_text("32000\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_rc6_ms("card0", "xe")
        assert val == 32000.0

    def test_missing_rc6_returns_none(self, tmp_path):
        """Returns None when RC6 sysfs file is missing."""
        (tmp_path / "card0").mkdir()

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_rc6_ms("card0", "i915")
        assert val is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestRc6Reading -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 7: Implement _read_rc6_ms**

Add to `app/backends/intel.py`:

```python
    def _read_rc6_ms(self, card: str, driver: str) -> Optional[float]:
        if driver == "xe":
            path = os.path.join(
                self._drm_base, card,
                "device", "tile0", "gt0", "gtidle", "idle_residency_ms",
            )
        else:
            path = os.path.join(
                self._drm_base, card, "gt", "gt0", "rc6_residency_ms"
            )
        val = self._read_sysfs(path)
        return float(val) if val else None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestRc6Reading -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Write tests for power reading**

```python
# Add to tests/test_intel_backend.py

class TestPowerReading:
    def test_read_energy_from_hwmon(self, tmp_path):
        """Reads energy counter from hwmon."""
        hwmon_dir = tmp_path / "card0" / "device" / "hwmon" / "hwmon3"
        hwmon_dir.mkdir(parents=True)
        (hwmon_dir / "energy1_input").write_text("5000000\n")  # 5J in uJ

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_energy_uj("card0")
        assert val == 5000000.0

    def test_no_hwmon_returns_none(self, tmp_path):
        """Returns None when hwmon directory is missing."""
        (tmp_path / "card0" / "device").mkdir(parents=True)

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_energy_uj("card0")
        assert val is None

    def test_no_energy_file_returns_none(self, tmp_path):
        """Returns None when energy file is missing in hwmon."""
        hwmon_dir = tmp_path / "card0" / "device" / "hwmon" / "hwmon3"
        hwmon_dir.mkdir(parents=True)

        backend = IntelBackend()
        backend._drm_base = str(tmp_path)
        val = backend._read_energy_uj("card0")
        assert val is None
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestPowerReading -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 11: Implement _find_hwmon and _read_energy_uj**

Add to `app/backends/intel.py`:

```python
    def _find_hwmon(self, card: str) -> Optional[str]:
        hwmon_dir = os.path.join(self._drm_base, card, "device", "hwmon")
        try:
            for entry in sorted(os.listdir(hwmon_dir)):
                if entry.startswith("hwmon"):
                    return os.path.join(hwmon_dir, entry)
        except (FileNotFoundError, OSError):
            pass
        return None

    def _read_energy_uj(self, card: str) -> Optional[float]:
        hwmon = self._find_hwmon(card)
        if not hwmon:
            return None
        val = self._read_sysfs(os.path.join(hwmon, "energy1_input"))
        return float(val) if val else None
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestPowerReading -v`
Expected: All 3 tests PASS

- [ ] **Step 13: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: add sysfs reads for frequency, RC6, and power"
```

---

## Chunk 3: IntelBackend — fdinfo Parsing

### Task 5: Parse i915 fdinfo format

**Files:**
- Modify: `app/backends/intel.py`
- Modify: `tests/test_intel_backend.py`

- [ ] **Step 1: Write tests for i915 fdinfo parsing**

```python
# Add to tests/test_intel_backend.py

I915_FDINFO = """\
pos:	0
flags:	02100002
mnt_id:	15
ino:	1234
drm-driver:	i915
drm-pdev:	0000:00:02.0
drm-client-id:	7
drm-engine-render:	9288864723 ns
drm-engine-copy:	2035071108 ns
drm-engine-video:	52567609040 ns
drm-engine-video-enhance:	0 ns
"""

I915_FDINFO_WITH_MEMORY = """\
pos:	0
flags:	02100002
drm-driver:	i915
drm-client-id:	12
drm-engine-render:	1000000 ns
drm-total-system:	232411136
drm-shared-system:	0
drm-resident-system:	122638336
drm-active-system:	4018176
drm-purgeable-system:	634880
"""


class TestI915FdinfoParsing:
    def test_parse_i915_fdinfo(self):
        """Parses engine counters from i915 fdinfo."""
        backend = IntelBackend()
        result = backend._parse_fdinfo(I915_FDINFO, "i915")
        assert result is not None
        assert result["client_id"] == "7"
        assert "render" in result["engines"]
        assert result["engines"]["render"]["ns"] == 9288864723
        assert "video" in result["engines"]
        assert result["engines"]["video"]["ns"] == 52567609040
        assert "copy" in result["engines"]
        assert "video-enhance" in result["engines"]

    def test_parse_i915_with_memory(self):
        """Parses memory counters from i915 fdinfo."""
        backend = IntelBackend()
        result = backend._parse_fdinfo(I915_FDINFO_WITH_MEMORY, "i915")
        assert result is not None
        mem = result["memory"]
        assert "system" in mem
        assert mem["system"]["total"] == 232411136
        assert mem["system"]["resident"] == 122638336
        assert mem["system"]["active"] == 4018176
        assert mem["system"]["purgeable"] == 634880

    def test_parse_wrong_driver_returns_none(self):
        """Returns None when fdinfo is for a different driver."""
        backend = IntelBackend()
        result = backend._parse_fdinfo(I915_FDINFO, "xe")
        assert result is None

    def test_parse_non_drm_fdinfo_returns_none(self):
        """Returns None for fdinfo without drm-driver line."""
        content = "pos:\t0\nflags:\t02100002\n"
        backend = IntelBackend()
        result = backend._parse_fdinfo(content, "i915")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestI915FdinfoParsing -v`
Expected: FAIL — `AttributeError: 'IntelBackend' object has no attribute '_parse_fdinfo'`

- [ ] **Step 3: Implement _parse_fdinfo**

Add to `app/backends/intel.py`:

```python
    def _parse_fdinfo(self, content: str, driver: str) -> Optional[dict]:
        """Parse a single fdinfo file for DRM metrics."""
        data = {}
        for line in content.splitlines():
            if ":\t" not in line and ": " not in line:
                continue
            if ":\t" in line:
                key, _, value = line.partition(":\t")
            else:
                key, _, value = line.partition(": ")
            data[key.strip()] = value.strip()

        if data.get("drm-driver") != driver:
            return None

        client_id = data.get("drm-client-id")
        if not client_id:
            return None

        entry = {"client_id": client_id, "engines": {}, "memory": {}}

        for key, value in data.items():
            if not key.startswith("drm-"):
                continue

            if driver == "i915" and key.startswith("drm-engine-"):
                engine_name = key[len("drm-engine-"):]
                ns_str = value.replace(" ns", "").strip()
                try:
                    entry["engines"][engine_name] = {"ns": int(ns_str)}
                except ValueError:
                    pass

            elif driver == "xe" and key.startswith("drm-cycles-"):
                engine_name = key[len("drm-cycles-"):]
                try:
                    entry["engines"].setdefault(engine_name, {})["cycles"] = int(value)
                except ValueError:
                    pass

            elif driver == "xe" and key.startswith("drm-total-cycles-"):
                engine_name = key[len("drm-total-cycles-"):]
                try:
                    entry["engines"].setdefault(engine_name, {})["total_cycles"] = int(value)
                except ValueError:
                    pass

            elif driver == "xe" and key.startswith("drm-engine-capacity-"):
                engine_name = key[len("drm-engine-capacity-"):]
                try:
                    entry["engines"].setdefault(engine_name, {})["capacity"] = int(value)
                except ValueError:
                    pass

            else:
                # Memory keys: drm-{stat}-{region}
                # e.g., drm-total-system, drm-resident-vram0
                # But NOT drm-total-cycles-* (already handled above for xe)
                for stat in ("total", "shared", "resident", "active", "purgeable"):
                    prefix = f"drm-{stat}-"
                    if key.startswith(prefix) and not key.startswith(f"drm-{stat}-cycles"):
                        region = key[len(prefix):]
                        val_str = value.split()[0] if value else "0"
                        try:
                            entry["memory"].setdefault(region, {})[stat] = int(val_str)
                        except ValueError:
                            pass
                        break

        return entry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestI915FdinfoParsing -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: add i915 fdinfo parsing for engine and memory counters"
```

### Task 6: Parse xe fdinfo format

**Files:**
- Modify: `tests/test_intel_backend.py`

- [ ] **Step 1: Write tests for xe fdinfo parsing**

```python
# Add to tests/test_intel_backend.py

XE_FDINFO = """\
pos:	0
flags:	02100002
drm-driver:	xe
drm-client-id:	42
drm-cycles-rcs:	28257900
drm-total-cycles-rcs:	7655183225
drm-cycles-bcs:	0
drm-total-cycles-bcs:	7655183225
drm-cycles-vcs:	0
drm-total-cycles-vcs:	7655183225
drm-engine-capacity-vcs:	2
drm-cycles-vecs:	0
drm-total-cycles-vecs:	7655183225
drm-engine-capacity-vecs:	2
drm-cycles-ccs:	0
drm-total-cycles-ccs:	7655183225
drm-engine-capacity-ccs:	4
drm-total-system:	1048576
drm-shared-system:	0
drm-resident-system:	524288
drm-active-system:	262144
drm-purgeable-system:	0
drm-total-vram0:	67108864
drm-shared-vram0:	0
drm-resident-vram0:	33554432
"""


class TestXeFdinfoParsing:
    def test_parse_xe_fdinfo(self):
        """Parses cycle counters from xe fdinfo."""
        backend = IntelBackend()
        result = backend._parse_fdinfo(XE_FDINFO, "xe")
        assert result is not None
        assert result["client_id"] == "42"
        assert "rcs" in result["engines"]
        assert result["engines"]["rcs"]["cycles"] == 28257900
        assert result["engines"]["rcs"]["total_cycles"] == 7655183225
        assert "vcs" in result["engines"]
        assert result["engines"]["vcs"]["capacity"] == 2
        assert "ccs" in result["engines"]
        assert result["engines"]["ccs"]["capacity"] == 4

    def test_parse_xe_memory(self):
        """Parses memory from xe fdinfo including VRAM."""
        backend = IntelBackend()
        result = backend._parse_fdinfo(XE_FDINFO, "xe")
        assert result is not None
        mem = result["memory"]
        assert "system" in mem
        assert mem["system"]["total"] == 1048576
        assert "vram0" in mem
        assert mem["vram0"]["total"] == 67108864
        assert mem["vram0"]["resident"] == 33554432

    def test_engine_name_mapping(self):
        """Maps xe engine names to display names."""
        backend = IntelBackend()
        assert backend._map_engine_name("rcs", "xe") == "Render/3D"
        assert backend._map_engine_name("vcs", "xe") == "Video"
        assert backend._map_engine_name("vecs", "xe") == "VideoEnhance"
        assert backend._map_engine_name("bcs", "xe") == "Blitter"
        assert backend._map_engine_name("ccs", "xe") == "Compute"

    def test_engine_name_mapping_i915(self):
        """Maps i915 engine names to display names."""
        backend = IntelBackend()
        assert backend._map_engine_name("render", "i915") == "Render/3D"
        assert backend._map_engine_name("video", "i915") == "Video"
        assert backend._map_engine_name("video-enhance", "i915") == "VideoEnhance"
        assert backend._map_engine_name("copy", "i915") == "Blitter"

    def test_unknown_engine_name_passthrough(self):
        """Unknown engine names pass through unchanged."""
        backend = IntelBackend()
        assert backend._map_engine_name("unknown-engine", "i915") == "unknown-engine"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestXeFdinfoParsing -v`
Expected: FAIL — `AttributeError: 'IntelBackend' object has no attribute '_map_engine_name'`

- [ ] **Step 3: Implement _map_engine_name**

Add to `app/backends/intel.py`:

```python
    def _map_engine_name(self, name: str, driver: str) -> str:
        engine_map = self.XE_ENGINE_MAP if driver == "xe" else self.I915_ENGINE_MAP
        return engine_map.get(name, name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestXeFdinfoParsing -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: add xe fdinfo parsing and engine name mapping"
```

### Task 7: Utilization delta computation

**Files:**
- Modify: `app/backends/intel.py`
- Modify: `tests/test_intel_backend.py`

- [ ] **Step 1: Write tests for utilization computation**

```python
# Add to tests/test_intel_backend.py

class TestUtilizationComputation:
    def test_i915_utilization_delta(self):
        """Computes i915 engine utilization from nanosecond deltas."""
        backend = IntelBackend()

        prev = {
            "7": {"engines": {"render": {"ns": 1_000_000_000}}}
        }
        curr = {
            "7": {"engines": {"render": {"ns": 1_500_000_000}}}
        }
        # 500ms busy in 1000ms wall time = 50%
        result = backend._compute_utilization(
            prev, curr, wall_time_s=1.0, driver="i915"
        )
        assert "7" in result
        render_busy = result["7"]["engines"]["render"]
        assert abs(render_busy - 50.0) < 0.1

    def test_xe_utilization_delta(self):
        """Computes xe engine utilization from cycle deltas."""
        backend = IntelBackend()

        prev = {
            "42": {"engines": {"rcs": {"cycles": 100, "total_cycles": 1000}}}
        }
        curr = {
            "42": {"engines": {"rcs": {"cycles": 200, "total_cycles": 2000}}}
        }
        # 100 busy cycles / 1000 total cycles = 10%
        result = backend._compute_utilization(
            prev, curr, wall_time_s=1.0, driver="xe"
        )
        assert "42" in result
        rcs_busy = result["42"]["engines"]["rcs"]
        assert abs(rcs_busy - 10.0) < 0.1

    def test_new_client_gets_zero(self):
        """A client not in previous scan gets 0% utilization."""
        backend = IntelBackend()

        prev = {}
        curr = {
            "7": {"engines": {"render": {"ns": 1_000_000_000}}}
        }
        result = backend._compute_utilization(
            prev, curr, wall_time_s=1.0, driver="i915"
        )
        assert "7" in result
        assert result["7"]["engines"]["render"] == 0.0

    def test_utilization_capped_at_100(self):
        """Utilization is capped at 100% for edge cases."""
        backend = IntelBackend()

        prev = {
            "7": {"engines": {"render": {"ns": 0}}}
        }
        curr = {
            "7": {"engines": {"render": {"ns": 2_000_000_000}}}
        }
        # 2000ms busy in 1000ms wall time — cap at 100%
        result = backend._compute_utilization(
            prev, curr, wall_time_s=1.0, driver="i915"
        )
        assert result["7"]["engines"]["render"] == 100.0

    def test_disappeared_client_excluded(self):
        """A client in previous but not current scan is excluded."""
        backend = IntelBackend()

        prev = {"7": {"engines": {"render": {"ns": 1_000_000_000}}}}
        curr = {}
        result = backend._compute_utilization(
            prev, curr, wall_time_s=1.0, driver="i915"
        )
        assert "7" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestUtilizationComputation -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement _compute_utilization**

Add to `app/backends/intel.py`:

```python
    def _compute_utilization(
        self,
        prev: dict,
        curr: dict,
        wall_time_s: float,
        driver: str,
    ) -> dict:
        """Compute per-client, per-engine utilization from counter deltas.

        Returns: {client_id: {"engines": {engine_name: busy_pct}}}
        """
        result = {}
        wall_time_ns = wall_time_s * 1_000_000_000

        for client_id, client_data in curr.items():
            prev_client = prev.get(client_id, {})
            prev_engines = prev_client.get("engines", {})
            engine_utils = {}

            for engine_name, counters in client_data.get("engines", {}).items():
                prev_counters = prev_engines.get(engine_name, {})

                if driver == "i915":
                    curr_ns = counters.get("ns", 0)
                    prev_ns = prev_counters.get("ns", 0)
                    if prev_ns == 0 and client_id not in prev:
                        busy_pct = 0.0
                    elif wall_time_ns > 0:
                        delta_ns = curr_ns - prev_ns
                        busy_pct = (delta_ns / wall_time_ns) * 100
                    else:
                        busy_pct = 0.0
                else:  # xe
                    curr_cycles = counters.get("cycles", 0)
                    prev_cycles = prev_counters.get("cycles", 0)
                    curr_total = counters.get("total_cycles", 0)
                    prev_total = prev_counters.get("total_cycles", 0)
                    delta_cycles = curr_cycles - prev_cycles
                    delta_total = curr_total - prev_total
                    if client_id not in prev:
                        busy_pct = 0.0
                    elif delta_total > 0:
                        busy_pct = (delta_cycles / delta_total) * 100
                    else:
                        busy_pct = 0.0

                # Clamp to [0, 100] — negative deltas possible on counter reset
                engine_utils[engine_name] = max(0.0, min(busy_pct, 100.0))

            result[client_id] = {"engines": engine_utils}

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestUtilizationComputation -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: add utilization delta computation for i915 and xe"
```

---

## Chunk 4: IntelBackend — read_sample Assembly

### Task 8: Implement read_sample

This is the integration point that assembles all the individual reads into a single normalized sample.

**Files:**
- Modify: `app/backends/intel.py`
- Modify: `tests/test_intel_backend.py`

- [ ] **Step 1: Write test for _scan_fdinfo**

```python
# Add to tests/test_intel_backend.py

class TestFdinfoScan:
    def test_scan_fdinfo_finds_drm_clients(self, tmp_path):
        """Scans /proc for DRM clients matching the target driver."""
        # Create fake /proc/1234/fdinfo/5 with i915 content
        fdinfo_dir = tmp_path / "1234" / "fdinfo"
        fdinfo_dir.mkdir(parents=True)
        (fdinfo_dir / "5").write_text(I915_FDINFO)
        # Create /proc/1234/comm
        (tmp_path / "1234" / "comm").write_text("ffmpeg\n")

        backend = IntelBackend()
        results = backend._scan_fdinfo(str(tmp_path), "i915")

        assert len(results) == 1
        assert results[0]["pid"] == "1234"
        assert results[0]["name"] == "ffmpeg"
        assert results[0]["client_id"] == "7"
        assert "render" in results[0]["engines"]

    def test_scan_fdinfo_skips_wrong_driver(self, tmp_path):
        """Skips fdinfo entries for a different driver."""
        fdinfo_dir = tmp_path / "1234" / "fdinfo"
        fdinfo_dir.mkdir(parents=True)
        (fdinfo_dir / "5").write_text(I915_FDINFO)

        backend = IntelBackend()
        results = backend._scan_fdinfo(str(tmp_path), "xe")
        assert len(results) == 0

    def test_scan_fdinfo_handles_missing_proc(self, tmp_path):
        """Returns empty list when proc dir is unreadable."""
        backend = IntelBackend()
        results = backend._scan_fdinfo(str(tmp_path / "nonexistent"), "i915")
        assert results == []

    def test_scan_fdinfo_deduplicates_by_client_id(self, tmp_path):
        """Keeps only one entry per drm-client-id (highest counters)."""
        fdinfo_dir = tmp_path / "1234" / "fdinfo"
        fdinfo_dir.mkdir(parents=True)
        # Same client-id in two fds
        (fdinfo_dir / "5").write_text(I915_FDINFO)
        (fdinfo_dir / "6").write_text(I915_FDINFO)
        (tmp_path / "1234" / "comm").write_text("ffmpeg\n")

        backend = IntelBackend()
        results = backend._scan_fdinfo(str(tmp_path), "i915")
        assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestFdinfoScan -v`
Expected: FAIL — `AttributeError: 'IntelBackend' object has no attribute '_scan_fdinfo'`

- [ ] **Step 3: Implement _scan_fdinfo**

Add to `app/backends/intel.py`:

```python
    def _scan_fdinfo(self, proc_path: str, driver: str) -> list[dict]:
        """Scan /proc for DRM clients matching the given driver."""
        results = {}  # client_id -> entry (dedup)
        try:
            pids = [p for p in os.listdir(proc_path) if p.isdigit()]
        except OSError:
            return []

        for pid in pids:
            fdinfo_dir = os.path.join(proc_path, pid, "fdinfo")
            try:
                fds = os.listdir(fdinfo_dir)
            except (FileNotFoundError, PermissionError):
                continue

            for fd in fds:
                try:
                    with open(os.path.join(fdinfo_dir, fd)) as f:
                        content = f.read()
                except (FileNotFoundError, PermissionError, OSError):
                    continue

                if "drm-driver:" not in content:
                    continue

                entry = self._parse_fdinfo(content, driver)
                if not entry:
                    continue

                client_id = entry["client_id"]
                entry["pid"] = pid
                comm = self._read_sysfs(os.path.join(proc_path, pid, "comm"))
                entry["name"] = comm or "unknown"

                # Dedup: keep entry with highest counters
                if client_id not in results:
                    results[client_id] = entry
                # If duplicate, keep existing (first found is fine)

        return list(results.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestFdinfoScan -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Write integration test for read_sample**

```python
# Add to tests/test_intel_backend.py
import time


class TestReadSample:
    def test_read_sample_i915(self, tmp_path):
        """Full read_sample produces normalized output for i915."""
        # Set up sysfs
        card_dir = tmp_path / "drm" / "card0"
        device_dir = card_dir / "device"
        device_dir.mkdir(parents=True)
        (device_dir / "vendor").write_text("0x8086\n")
        (device_dir / "product_name").write_text("Intel UHD 730\n")
        driver_target = tmp_path / "drivers" / "i915"
        driver_target.mkdir(parents=True)
        os.symlink(str(driver_target), str(device_dir / "driver"))
        (card_dir / "gt_act_freq_mhz").write_text("1300\n")
        (card_dir / "gt_cur_freq_mhz").write_text("1350\n")
        rc6_dir = card_dir / "gt" / "gt0"
        rc6_dir.mkdir(parents=True)
        (rc6_dir / "rc6_residency_ms").write_text("50000\n")

        # Set up fake proc with fdinfo
        proc_dir = tmp_path / "proc"
        fdinfo_dir = proc_dir / "1234" / "fdinfo"
        fdinfo_dir.mkdir(parents=True)
        (fdinfo_dir / "5").write_text(I915_FDINFO)
        (proc_dir / "1234" / "comm").write_text("ffmpeg\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path / "drm")
        backend._proc_path = str(proc_dir)
        backend._driver_cache["card0"] = "i915"

        # First sample sets baseline (deltas will be 0)
        sample1 = backend.read_sample("card0")
        assert sample1 is not None
        assert "frequency" in sample1
        assert sample1["frequency"]["actual"] == 1300.0
        assert sample1["frequency"]["requested"] == 1350.0
        assert "engines" in sample1
        assert "clients" in sample1
        assert "gpu_busy" in sample1
        assert "period" in sample1

    def test_read_sample_xe_gpu_busy_fallback(self, tmp_path):
        """When no RC6/idle_residency, gpu_busy falls back to max engine utilization."""
        card_dir = tmp_path / "drm" / "card0"
        device_dir = card_dir / "device"
        device_dir.mkdir(parents=True)
        # xe frequency paths
        freq_dir = device_dir / "tile0" / "gt0" / "freq0"
        freq_dir.mkdir(parents=True)
        (freq_dir / "act_freq").write_text("2100\n")
        (freq_dir / "cur_freq").write_text("2400\n")
        # No gtidle directory — forces fallback

        # Set up fake proc with xe fdinfo
        proc_dir = tmp_path / "proc"
        fdinfo_dir = proc_dir / "5678" / "fdinfo"
        fdinfo_dir.mkdir(parents=True)
        (fdinfo_dir / "3").write_text(XE_FDINFO)
        (proc_dir / "5678" / "comm").write_text("test\n")

        backend = IntelBackend()
        backend._drm_base = str(tmp_path / "drm")
        backend._proc_path = str(proc_dir)
        backend._driver_cache["card0"] = "xe"

        # First sample (baseline)
        sample1 = backend.read_sample("card0")
        # gpu_busy should be 0 on first sample (no previous delta)
        assert sample1["gpu_busy"] == 0.0

    def test_read_sample_structure(self, tmp_path):
        """Verifies all required keys are present in the sample."""
        card_dir = tmp_path / "drm" / "card0"
        (card_dir / "device").mkdir(parents=True)
        card_dir.mkdir(exist_ok=True)

        backend = IntelBackend()
        backend._drm_base = str(tmp_path / "drm")
        backend._proc_path = str(tmp_path / "proc")
        backend._driver_cache["card0"] = "i915"

        sample = backend.read_sample("card0")
        required_keys = ["period", "frequency", "gpu_busy", "power", "engines", "clients"]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestReadSample -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 7: Implement read_sample**

Replace the `read_sample` stub in `app/backends/intel.py` (the `__init__` already has `_proc_path` from Task 3):

```python
    def read_sample(self, device: str) -> dict:
        now = time.time()
        driver = self._detect_driver(device) or "i915"
        wall_time_s = now - self._prev_time if self._prev_time > 0 else 1.0

        # Frequency
        frequency = self._read_frequency(device, driver)

        # RC6 / GPU busy
        rc6_ms = self._read_rc6_ms(device, driver)
        gpu_busy = None
        if rc6_ms is not None and self._prev_rc6_ms is not None and wall_time_s > 0:
            delta_rc6_ms = rc6_ms - self._prev_rc6_ms
            rc6_pct = (delta_rc6_ms / (wall_time_s * 1000)) * 100
            gpu_busy = max(0.0, min(100.0, 100.0 - rc6_pct))
        self._prev_rc6_ms = rc6_ms

        # Power
        energy_uj = self._read_energy_uj(device)
        power_watts = None
        if (
            energy_uj is not None
            and self._prev_energy_uj is not None
            and wall_time_s > 0
        ):
            delta_uj = energy_uj - self._prev_energy_uj
            power_watts = delta_uj / (wall_time_s * 1_000_000)
        self._prev_energy_uj = energy_uj

        # Per-process fdinfo
        raw_clients = self._scan_fdinfo(self._proc_path, driver)

        # Build current counter snapshot for delta computation
        curr_counters = {}
        for client in raw_clients:
            curr_counters[client["client_id"]] = {
                "engines": client["engines"],
            }

        # Compute utilization deltas
        util = self._compute_utilization(
            self._prev_counters, curr_counters, wall_time_s, driver
        )
        self._prev_counters = curr_counters
        self._prev_time = now

        # Build per-engine device-level utilization (aggregate from clients)
        engine_totals: dict[str, float] = {}
        for client_id, client_util in util.items():
            for engine, busy_pct in client_util["engines"].items():
                display_name = self._map_engine_name(engine, driver)
                key = f"{display_name}/0"
                engine_totals[key] = engine_totals.get(key, 0.0) + busy_pct

        engines = {}
        for key, total in engine_totals.items():
            engines[key] = {"busy": min(total, 100.0)}

        # If no RC6 data, derive gpu_busy from max engine utilization
        if gpu_busy is None and engines:
            gpu_busy = max(e["busy"] for e in engines.values())
        elif gpu_busy is None:
            gpu_busy = 0.0

        # Build clients dict
        clients = {}
        for client in raw_clients:
            client_id = client["client_id"]
            client_util = util.get(client_id, {}).get("engines", {})
            engine_classes = {}
            for engine, busy_pct in client_util.items():
                display_name = self._map_engine_name(engine, driver)
                engine_classes[display_name] = {"busy": round(busy_pct, 1)}

            memory = {}
            if client.get("memory"):
                memory = {"system": client["memory"].get("system", {})}
                # Include vram if available (xe)
                for region, data in client["memory"].items():
                    if region != "system":
                        memory[region] = data

            client_entry = {
                "pid": client["pid"],
                "name": client["name"],
                "engine-classes": engine_classes,
            }
            if memory:
                client_entry["memory"] = memory
            clients[client_id] = client_entry

        power = {"GPU": round(power_watts, 1)} if power_watts is not None else None

        return {
            "period": {"duration": wall_time_s * 1000},
            "frequency": frequency,
            "gpu_busy": round(gpu_busy, 1),
            "power": power,
            "engines": engines,
            "clients": clients,
        }
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py::TestReadSample -v`
Expected: All 2 tests PASS

- [ ] **Step 9: Run all backend tests**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_intel_backend.py -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add app/backends/intel.py tests/test_intel_backend.py
git commit -m "feat: implement read_sample assembling sysfs, fdinfo, and deltas"
```

---

## Chunk 5: GpuMonitor Refactor

### Task 9: Refactor GpuMonitor to use backend

**Files:**
- Modify: `app/gpu_monitor.py`
- Modify: `tests/test_gpu_monitor.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Update conftest.py with new-format fixtures**

```python
# tests/conftest.py
import pytest


SAMPLE_GPU_JSON = {
    "period": {"duration": 1000.0},
    "frequency": {"actual": 1300.0, "requested": 1350.0},
    "gpu_busy": 67.5,
    "power": {"GPU": 8.2},
    "engines": {
        "Render/3D/0": {"busy": 42.0},
        "Video/0": {"busy": 87.0},
        "VideoEnhance/0": {"busy": 65.0},
        "Blitter/0": {"busy": 12.0},
    },
    "clients": {
        "1234": {
            "pid": "4821",
            "name": "Plex Transcoder",
            "engine-classes": {
                "Render/3D": {"busy": 38.0},
                "Video": {"busy": 72.0},
            },
            "memory": {
                "system": {
                    "total": 232411136,
                    "shared": 0,
                    "resident": 122638336,
                    "purgeable": 634880,
                    "active": 4018176,
                }
            },
        }
    },
}

SAMPLE_GPU_JSON_MINIMAL = {
    "period": {"duration": 1000.0},
    "frequency": {"actual": 300.0, "requested": 300.0},
    "gpu_busy": 2.0,
    "power": None,
    "engines": {
        "Render/3D/0": {"busy": 0.0},
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

- [ ] **Step 2: Rewrite GpuMonitor to use backend**

Rewrite `app/gpu_monitor.py`:

```python
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
            # Strip internal 'driver' field for API
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
        # Return without driver field for API compatibility
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
```

- [ ] **Step 3: Update test_gpu_monitor.py**

```python
# tests/test_gpu_monitor.py
import json
import time
from unittest.mock import patch, mock_open

from tests.conftest import SAMPLE_GPU_JSON, SAMPLE_GPU_JSON_MINIMAL

from app.gpu_monitor import GpuMonitor


class TestRingBuffer:
    def test_add_sample_and_get_current(self):
        monitor = GpuMonitor(buffer_size=300)
        monitor.add_sample(SAMPLE_GPU_JSON.copy())
        current = monitor.get_current()
        assert current is not None
        assert "timestamp" in current
        assert current["frequency"]["requested"] == 1350.0

    def test_buffer_respects_max_size(self):
        monitor = GpuMonitor(buffer_size=5)
        for i in range(10):
            sample = SAMPLE_GPU_JSON.copy()
            sample["gpu_busy"] = float(i)
            monitor.add_sample(sample)
        history = monitor.get_history()
        assert len(history) == 5
        assert history[0]["gpu_busy"] == 5.0

    def test_get_current_when_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        assert monitor.get_current() is None

    def test_get_history_when_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        assert monitor.get_history() == []


class TestCmdlineEnrichment:
    @patch("builtins.open", mock_open(read_data=b"/usr/bin/ffmpeg\x00-i\x00video.mkv"))
    def test_read_cmdline_success(self):
        result = GpuMonitor._read_cmdline("12345")
        assert result == "/usr/bin/ffmpeg -i video.mkv"

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_read_cmdline_process_gone(self, _):
        result = GpuMonitor._read_cmdline("99999")
        assert result is None

    @patch("builtins.open", side_effect=PermissionError)
    def test_read_cmdline_permission_denied(self, _):
        result = GpuMonitor._read_cmdline("1")
        assert result is None

    @patch.object(GpuMonitor, "_read_cmdline", return_value="/usr/bin/plex")
    def test_enrich_clients_adds_cmdline(self, _):
        monitor = GpuMonitor(buffer_size=300)
        data = {
            "period": {"duration": 1000.0},
            "clients": {
                "1234": {"pid": "4821", "name": "Plex Transcoder"},
            },
        }
        monitor._enrich_clients(data)
        assert data["clients"]["1234"]["cmdline"] == "/usr/bin/plex"

    def test_enrich_clients_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        data = {"period": {"duration": 1000.0}, "clients": {}}
        monitor._enrich_clients(data)
        assert data["clients"] == {}


class TestDeviceManagement:
    def test_discover_gpus_with_backend(self):
        from app.backends.base import GpuBackend

        class FakeBackend(GpuBackend):
            def discover_devices(self):
                return [
                    {"device": "card0", "name": "Test GPU", "driver": "i915"},
                ]

            def read_sample(self, device):
                return {}

            def cleanup(self):
                pass

        monitor = GpuMonitor(buffer_size=300, backend=FakeBackend())
        result = monitor.discover_gpus()
        assert len(result) == 1
        assert result[0]["device"] == "card0"
        assert result[0]["name"] == "Test GPU"
        # driver field should NOT be exposed
        assert "driver" not in result[0]

    def test_available_gpus_strips_driver(self):
        monitor = GpuMonitor(buffer_size=300)
        monitor._available_gpus = [
            {"device": "card0", "name": "Test", "driver": "i915"}
        ]
        gpus = monitor.available_gpus
        assert "driver" not in gpus[0]

    def test_discover_gpus_no_backend(self):
        monitor = GpuMonitor(buffer_size=300)
        result = monitor.discover_gpus()
        assert result == []
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/test_gpu_monitor.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/gpu_monitor.py tests/test_gpu_monitor.py tests/conftest.py
git commit -m "refactor: replace intel_gpu_top subprocess with pluggable backend in GpuMonitor"
```

---

## Chunk 6: API, Frontend, Dockerfile Updates

### Task 10: Update main.py

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Update main.py to use IntelBackend**

```python
# app/main.py
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.backends.intel import IntelBackend
from app.gpu_monitor import GpuMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tdp = int(os.environ.get("GPU_TDP_WATTS", "60"))

# Auto-detect backend
backend = IntelBackend()
monitor = GpuMonitor(buffer_size=300, backend=backend)


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpus = monitor.discover_gpus()
    if gpus:
        monitor.gpu_name = gpus[0]["name"]
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


@app.get("/api/gpus")
async def gpus():
    return {
        "gpus": monitor.available_gpus,
        "current_device": monitor.current_device,
    }


@app.post("/api/gpus/select")
async def select_gpu(request: Request):
    body = await request.json()
    device = body.get("device")
    valid_devices = [g["device"] for g in monitor.available_gpus]
    if device is not None and device not in valid_devices:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown device: {device}"},
        )
    await monitor.select_device(device)
    return {
        "status": "ok",
        "device": device,
        "gpu_name": monitor.gpu_name,
    }


@app.get("/api/debug")
async def debug():
    current = monitor.get_current()

    checks = {}
    checks["pid_namespace"] = "host" if os.path.exists("/proc/1/cmdline") else "container"
    try:
        with open("/proc/1/cmdline", "rb") as f:
            checks["pid1_cmdline"] = f.read().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
    except (FileNotFoundError, PermissionError) as e:
        checks["pid1_cmdline"] = str(e)

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

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cap"):
                    key = line.split(":")[0].strip()
                    checks[key] = line.split(":")[1].strip()
    except (FileNotFoundError, PermissionError):
        checks["capabilities"] = "unreadable"

    return {
        "gpu_name": monitor.gpu_name,
        "uptime_seconds": monitor.uptime_seconds,
        "error": monitor.get_error(),
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
```

- [ ] **Step 2: Update test_api.py for new data model**

```python
# tests/test_api.py
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


@pytest.mark.asyncio
async def test_gpus_endpoint(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpus" in data
    assert "current_device" in data
    assert isinstance(data["gpus"], list)


@pytest.mark.asyncio
async def test_debug_endpoint(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/debug")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu_name" in data
    assert "uptime_seconds" in data
    assert "error" in data
    assert "buffer_size" in data


@pytest.mark.asyncio
async def test_status_response_has_expected_fields(monitor):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/status")
    data = resp.json()
    current = data["current"]
    assert current is not None
    assert "engines" in current
    assert len(current["engines"]) > 0
    assert "clients" in current
    assert "frequency" in current
    assert "actual" in current["frequency"]
    assert "requested" in current["frequency"]
    assert "gpu_busy" in current
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "refactor: update API layer to use backend, fix select_gpu 400 response"
```

### Task 11: Update frontend

**Files:**
- Modify: `app/static/app.js`
- Modify: `app/static/index.html`

- [ ] **Step 1: Update app.js**

Changes needed:
1. Add `Compute` to `ENGINE_COLORS`
2. Update `renderGpuBusy` to use `sample.gpu_busy` instead of `100 - sample.rc6.value`
3. Remove interrupts rendering
4. Update `gpuName` default from `"Intel GPU"` to `"GPU"`

In `app/static/app.js`:

Replace `ENGINE_COLORS`:
```javascript
  var ENGINE_COLORS = {
    "Render/3D": "#60a5fa",
    "Video": "#34d399",
    "VideoEnhance": "#a855f7",
    "Blitter": "#fbbf24",
    "Compute": "#f472b6",
  };
```

Replace `gpuName` default in `fetchStatus`:
```javascript
      gpuName = data.gpu_name || "GPU";
```

Replace `renderGpuBusy` function:
```javascript
  function renderGpuBusy(sample) {
    var busy = sample.gpu_busy || 0;
    busy = Math.max(0, Math.min(100, busy));
    var circumference = 213.6;
    var offset = circumference - (busy / 100) * circumference;
    $("gpu-busy-arc").style.strokeDashoffset = offset;
    $("gpu-busy-pct").textContent = busy.toFixed(0) + "%";
  }
```

- [ ] **Step 2: Update index.html**

In `app/static/index.html`:

Replace `<h1>Intel GPU Monitor</h1>` with `<h1>GPU Monitor</h1>`.

Replace the GPU Busy card title and remove the interrupts sub-element:
```html
    <div class="card">
      <div class="card-title">GPU Busy</div>
      <div class="gauge-container">
        <div class="gauge">
          <svg viewBox="0 0 80 80">
            <circle class="gauge-bg" cx="40" cy="40" r="34"/>
            <circle class="gauge-fill" id="gpu-busy-arc" cx="40" cy="40" r="34"
              stroke-dasharray="213.6" stroke-dashoffset="213.6"/>
          </svg>
          <div class="gauge-text" id="gpu-busy-pct">--%</div>
        </div>
      </div>
    </div>
```

Replace the footer text:
```html
    GPU Monitor — <span id="footer-status">Connecting...</span> — Uptime: <span id="uptime">--</span>
```

- [ ] **Step 3: Commit**

```bash
git add app/static/app.js app/static/index.html
git commit -m "feat: update frontend for new data model (gpu_busy, compute engine, no interrupts)"
```

### Task 12: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Simplify Dockerfile (remove igt-gpu-tools build stage)**

```dockerfile
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pciutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Run all tests one final time**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "refactor: simplify Dockerfile, remove intel_gpu_top build stage"
```

### Task 13: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/ajthom90/projects/toparr && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify no import errors**

Run: `cd /Users/ajthom90/projects/toparr && python -c "from app.main import app; from app.backends.intel import IntelBackend; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Review git log for clean commit history**

Run: `cd /Users/ajthom90/projects/toparr && git log --oneline -10`
Expected: Clean sequence of commits for this feature
