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
            "period": {"unit": "ms", "duration": 1000.0},
            "clients": {
                "1234": {"pid": "4821", "name": "Plex Transcoder"},
            },
        }
        monitor._enrich_clients(data)
        assert data["clients"]["1234"]["cmdline"] == "/usr/bin/plex"

    def test_enrich_clients_empty(self):
        monitor = GpuMonitor(buffer_size=300)
        data = {"period": {"unit": "ms", "duration": 1000.0}, "clients": {}}
        monitor._enrich_clients(data)
        assert data["clients"] == {}


class TestGpuDiscovery:
    @patch("subprocess.run")
    def test_list_gpus(self, mock_run):
        mock_run.return_value.stdout = (
            "card0   Intel UHD Graphics 730\n"
            "card1   Intel Arc A770\n"
        )
        gpus = GpuMonitor.list_gpus()
        assert len(gpus) == 2
        assert gpus[0]["device"] == "card0"
        assert gpus[0]["name"] == "Intel UHD Graphics 730"
        assert gpus[1]["device"] == "card1"
        assert gpus[1]["name"] == "Intel Arc A770"

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_list_gpus_no_binary(self, _):
        gpus = GpuMonitor.list_gpus()
        assert gpus == []

    def test_discover_gpus_stores_result(self):
        with patch.object(GpuMonitor, "list_gpus", return_value=[
            {"device": "card0", "name": "Test GPU"},
        ]):
            monitor = GpuMonitor(buffer_size=300)
            result = monitor.discover_gpus()
            assert len(result) == 1
            assert monitor.available_gpus == result
