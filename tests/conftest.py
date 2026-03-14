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
