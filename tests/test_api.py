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
