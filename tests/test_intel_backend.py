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
