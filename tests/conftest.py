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
