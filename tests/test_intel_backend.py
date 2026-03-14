"""Tests for IntelBackend — sysfs/fdinfo-based GPU monitoring."""
import os
import textwrap

import pytest

from app.backends.intel import IntelBackend


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def backend():
    return IntelBackend()


def _make_card(tmp_path, card_name, vendor="0x8086", driver="i915",
               product_name=None):
    """Create a minimal sysfs card tree under *tmp_path*."""
    card = tmp_path / card_name
    device = card / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor)
    # Create a driver symlink — the target doesn't need to be real for
    # os.path.basename to work.
    driver_link = device / "driver"
    os.symlink(f"/fake/drivers/{driver}", str(driver_link))
    if product_name:
        (device / "product_name").write_text(product_name)
    return card


# ── Task 3 — Device discovery ────────────────────────────────────────

class TestReadSysfs:
    def test_read_existing_file(self, tmp_path, backend):
        f = tmp_path / "test_file"
        f.write_text("  hello world  \n")
        assert backend._read_sysfs(str(f)) == "hello world"

    def test_read_missing_file(self, backend):
        assert backend._read_sysfs("/nonexistent/path/xyz") is None


class TestDetectDriver:
    def test_detect_xe_driver(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", driver="xe")
        assert backend._detect_driver("card0") == "xe"

    def test_detect_i915_driver(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", driver="i915")
        assert backend._detect_driver("card0") == "i915"

    def test_driver_cached(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", driver="i915")
        backend._detect_driver("card0")
        # Remove the symlink — cached value should still be returned
        os.unlink(str(tmp_path / "card0" / "device" / "driver"))
        assert backend._detect_driver("card0") == "i915"

    def test_missing_driver_returns_none(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        card = tmp_path / "card0" / "device"
        card.mkdir(parents=True)
        assert backend._detect_driver("card0") is None


class TestDetectGpuName:
    def test_product_name_from_sysfs(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", product_name="Intel Arc A770")
        assert backend._detect_gpu_name("card0") == "Intel Arc A770"

    def test_fallback_to_intel_gpu(self, tmp_path, backend):
        """Without product_name and without lspci, falls back to 'Intel GPU'."""
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0")
        # _detect_gpu_name tries lspci which will fail in tests
        name = backend._detect_gpu_name("card0")
        # Should be either a real lspci result or the final fallback
        assert isinstance(name, str) and len(name) > 0


class TestDiscoverDevices:
    def test_discover_intel_gpu(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", driver="i915",
                   product_name="Intel UHD 770")
        devices = backend.discover_devices()
        assert len(devices) == 1
        assert devices[0]["device"] == "card0"
        assert devices[0]["name"] == "Intel UHD 770"
        assert devices[0]["driver"] == "i915"

    def test_skip_non_intel(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", vendor="0x10de", driver="nvidia")
        devices = backend.discover_devices()
        assert devices == []

    def test_skip_renderD_nodes(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "renderD128", driver="i915")
        _make_card(tmp_path, "card0", driver="i915",
                   product_name="Intel GPU")
        devices = backend.discover_devices()
        assert len(devices) == 1
        assert devices[0]["device"] == "card0"

    def test_empty_drm(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        devices = backend.discover_devices()
        assert devices == []

    def test_multiple_intel_cards(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        _make_card(tmp_path, "card0", driver="i915",
                   product_name="Intel UHD 770")
        _make_card(tmp_path, "card1", driver="xe",
                   product_name="Intel Arc A770")
        devices = backend.discover_devices()
        assert len(devices) == 2
        names = {d["device"] for d in devices}
        assert names == {"card0", "card1"}


class TestCleanup:
    def test_cleanup_clears_state(self, backend):
        backend._prev_counters = {"some": "data"}
        backend._prev_time = 12345.0
        backend._prev_rc6_ms = {"card0": 100.0}
        backend._prev_energy_uj = {"card0": 500.0}
        backend._driver_cache = {"card0": "i915"}
        backend.cleanup()
        assert backend._prev_counters == {}
        assert backend._prev_time is None
        assert backend._prev_rc6_ms == {}
        assert backend._prev_energy_uj == {}
        assert backend._driver_cache == {}


# ── Task 4 — Sysfs reads: frequency, RC6, power ─────────────────────

class TestReadFrequency:
    def test_i915_frequency(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        card = tmp_path / "card0"
        card.mkdir(parents=True)
        (card / "gt_act_freq_mhz").write_text("1350\n")
        (card / "gt_cur_freq_mhz").write_text("1500\n")
        freq = backend._read_frequency("card0", "i915")
        assert freq == {"actual": 1350.0, "requested": 1500.0}

    def test_xe_frequency(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        freq_dir = tmp_path / "card0" / "device" / "tile0" / "gt0" / "freq0"
        freq_dir.mkdir(parents=True)
        (freq_dir / "act_freq").write_text("2100\n")
        (freq_dir / "cur_freq").write_text("2400\n")
        freq = backend._read_frequency("card0", "xe")
        assert freq == {"actual": 2100.0, "requested": 2400.0}

    def test_missing_frequency_returns_zeros(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        (tmp_path / "card0").mkdir(parents=True)
        freq = backend._read_frequency("card0", "i915")
        assert freq == {"actual": 0.0, "requested": 0.0}

    def test_missing_xe_frequency_returns_zeros(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        (tmp_path / "card0").mkdir(parents=True)
        freq = backend._read_frequency("card0", "xe")
        assert freq == {"actual": 0.0, "requested": 0.0}


class TestReadRC6:
    def test_i915_rc6(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        rc6_dir = tmp_path / "card0" / "gt" / "gt0"
        rc6_dir.mkdir(parents=True)
        (rc6_dir / "rc6_residency_ms").write_text("12345\n")
        result = backend._read_rc6_ms("card0", "i915")
        assert result == 12345.0

    def test_xe_idle_residency(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        idle_dir = tmp_path / "card0" / "device" / "tile0" / "gt0" / "gtidle"
        idle_dir.mkdir(parents=True)
        (idle_dir / "idle_residency_ms").write_text("98765\n")
        result = backend._read_rc6_ms("card0", "xe")
        assert result == 98765.0

    def test_missing_rc6_returns_none(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        (tmp_path / "card0").mkdir(parents=True)
        assert backend._read_rc6_ms("card0", "i915") is None
        assert backend._read_rc6_ms("card0", "xe") is None


class TestHwmonEnergy:
    def test_find_hwmon_and_read_energy(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        hwmon = tmp_path / "card0" / "device" / "hwmon" / "hwmon3"
        hwmon.mkdir(parents=True)
        (hwmon / "energy1_input").write_text("5000000\n")
        hwmon_path = backend._find_hwmon("card0")
        assert hwmon_path is not None
        energy = backend._read_energy_uj("card0")
        assert energy == 5000000.0

    def test_no_hwmon_returns_none(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        (tmp_path / "card0" / "device").mkdir(parents=True)
        assert backend._find_hwmon("card0") is None

    def test_no_energy_file_returns_none(self, tmp_path, backend):
        backend._drm_base = str(tmp_path)
        hwmon = tmp_path / "card0" / "device" / "hwmon" / "hwmon0"
        hwmon.mkdir(parents=True)
        # hwmon exists but no energy1_input file
        assert backend._read_energy_uj("card0") is None


# ── Tasks 5 & 6 — fdinfo parsing and engine name mapping ────────────

I915_FDINFO = """\
pos:\t0
flags:\t02100002
mnt_id:\t15
ino:\t1234
drm-driver:\ti915
drm-pdev:\t0000:00:02.0
drm-client-id:\t7
drm-engine-render:\t9288864723 ns
drm-engine-copy:\t2035071108 ns
drm-engine-video:\t52567609040 ns
drm-engine-video-enhance:\t0 ns
"""

I915_FDINFO_WITH_MEMORY = """\
pos:\t0
flags:\t02100002
drm-driver:\ti915
drm-client-id:\t12
drm-engine-render:\t1000000 ns
drm-total-system:\t232411136
drm-shared-system:\t0
drm-resident-system:\t122638336
drm-active-system:\t4018176
drm-purgeable-system:\t634880
"""

XE_FDINFO = """\
pos:\t0
flags:\t02100002
drm-driver:\txe
drm-client-id:\t42
drm-cycles-rcs:\t28257900
drm-total-cycles-rcs:\t7655183225
drm-cycles-bcs:\t0
drm-total-cycles-bcs:\t7655183225
drm-cycles-vcs:\t0
drm-total-cycles-vcs:\t7655183225
drm-engine-capacity-vcs:\t2
drm-cycles-vecs:\t0
drm-total-cycles-vecs:\t7655183225
drm-engine-capacity-vecs:\t2
drm-cycles-ccs:\t0
drm-total-cycles-ccs:\t7655183225
drm-engine-capacity-ccs:\t4
drm-total-system:\t1048576
drm-shared-system:\t0
drm-resident-system:\t524288
drm-active-system:\t262144
drm-purgeable-system:\t0
drm-total-vram0:\t67108864
drm-shared-vram0:\t0
drm-resident-vram0:\t33554432
"""


class TestParseFdinfoI915:
    def test_parse_i915_engines(self, backend):
        result = backend._parse_fdinfo(I915_FDINFO, "i915")
        assert result is not None
        assert result["client_id"] == "7"
        engines = result["engines"]
        assert engines["render"]["ns"] == 9288864723
        assert engines["copy"]["ns"] == 2035071108
        assert engines["video"]["ns"] == 52567609040
        assert engines["video-enhance"]["ns"] == 0

    def test_parse_i915_memory(self, backend):
        result = backend._parse_fdinfo(I915_FDINFO_WITH_MEMORY, "i915")
        assert result is not None
        mem = result["memory"]
        assert mem["system"]["total"] == 232411136
        assert mem["system"]["shared"] == 0
        assert mem["system"]["resident"] == 122638336
        assert mem["system"]["active"] == 4018176
        assert mem["system"]["purgeable"] == 634880

    def test_wrong_driver_returns_none(self, backend):
        result = backend._parse_fdinfo(I915_FDINFO, "xe")
        assert result is None

    def test_non_drm_fdinfo_returns_none(self, backend):
        content = "pos:\t0\nflags:\t02100002\n"
        result = backend._parse_fdinfo(content, "i915")
        assert result is None


class TestParseFdinfoXe:
    def test_parse_xe_cycles(self, backend):
        result = backend._parse_fdinfo(XE_FDINFO, "xe")
        assert result is not None
        assert result["client_id"] == "42"
        engines = result["engines"]
        assert engines["rcs"]["cycles"] == 28257900
        assert engines["rcs"]["total_cycles"] == 7655183225
        assert engines["bcs"]["cycles"] == 0
        assert engines["bcs"]["total_cycles"] == 7655183225
        assert engines["vcs"]["capacity"] == 2
        assert engines["ccs"]["capacity"] == 4

    def test_parse_xe_memory(self, backend):
        result = backend._parse_fdinfo(XE_FDINFO, "xe")
        assert result is not None
        mem = result["memory"]
        assert mem["system"]["total"] == 1048576
        assert mem["system"]["resident"] == 524288
        assert mem["vram0"]["total"] == 67108864
        assert mem["vram0"]["resident"] == 33554432

    def test_total_cycles_not_parsed_as_memory(self, backend):
        """drm-total-cycles-rcs must NOT end up in memory dict."""
        result = backend._parse_fdinfo(XE_FDINFO, "xe")
        assert result is not None
        # "cycles-rcs" should not appear as a memory region
        for region in result["memory"]:
            assert "cycles" not in region


class TestEngineNameMapping:
    def test_i915_mapping(self, backend):
        assert backend._map_engine_name("render", "i915") == "Render/3D"
        assert backend._map_engine_name("video", "i915") == "Video"
        assert backend._map_engine_name("video-enhance", "i915") == "VideoEnhance"
        assert backend._map_engine_name("copy", "i915") == "Blitter"

    def test_xe_mapping(self, backend):
        assert backend._map_engine_name("rcs", "xe") == "Render/3D"
        assert backend._map_engine_name("vcs", "xe") == "Video"
        assert backend._map_engine_name("vecs", "xe") == "VideoEnhance"
        assert backend._map_engine_name("bcs", "xe") == "Blitter"
        assert backend._map_engine_name("ccs", "xe") == "Compute"

    def test_unknown_engine_passthrough(self, backend):
        assert backend._map_engine_name("unknown-engine", "i915") == "unknown-engine"
        assert backend._map_engine_name("unknown-engine", "xe") == "unknown-engine"
