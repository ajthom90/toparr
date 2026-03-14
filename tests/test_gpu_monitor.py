import json
import time
from unittest.mock import patch, mock_open

import pytest

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
                return [{"device": "card0", "name": "Test GPU", "driver": "i915"}]
            def read_sample(self, device):
                return {}
            def cleanup(self):
                pass

        monitor = GpuMonitor(buffer_size=300, backend=FakeBackend())
        result = monitor.discover_gpus()
        assert len(result) == 1
        assert result[0]["device"] == "card0"
        assert result[0]["name"] == "Test GPU"
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

    @pytest.mark.asyncio
    async def test_select_device_resets_state(self):
        from app.backends.base import GpuBackend

        class FakeBackend(GpuBackend):
            def __init__(self):
                self.cleanup_called = False

            def discover_devices(self):
                return [
                    {"device": "card0", "name": "GPU A", "driver": "i915"},
                    {"device": "card1", "name": "GPU B", "driver": "xe"},
                ]

            def read_sample(self, device):
                return {}

            def cleanup(self):
                self.cleanup_called = True

        backend = FakeBackend()
        monitor = GpuMonitor(buffer_size=300, backend=backend)
        monitor.discover_gpus()
        monitor.add_sample(SAMPLE_GPU_JSON.copy())
        assert monitor.get_current() is not None
        assert len(monitor.get_history()) == 1

        await monitor.select_device("card1")

        assert monitor.get_current() is None
        assert monitor.get_history() == []
        assert monitor.get_error() is None
        assert monitor.current_device == "card1"
        assert monitor.gpu_name == "GPU B"
        assert backend.cleanup_called
