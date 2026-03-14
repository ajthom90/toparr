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
