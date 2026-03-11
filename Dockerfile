# Usage:
#   docker build -t intel-gpu-monitor .
#   docker run -d \
#     --name intel-gpu-monitor \
#     --device /dev/dri \
#     --cap-add CAP_PERFMON \
#     --pid host \
#     -p 8080:8080 \
#     -e GPU_TDP_WATTS=60 \
#     intel-gpu-monitor
#
# Then open http://<host-ip>:8080

# Stage 1: Build intel_gpu_top from igt-gpu-tools v2.3
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    meson ninja-build build-essential curl ca-certificates flex bison \
    libdrm-dev libproc2-dev libkmod-dev libpciaccess-dev libunwind-dev libdw-dev \
    libpixman-1-dev libcairo2-dev libudev-dev \
    pkg-config python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN curl -L https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/archive/v2.3/igt-gpu-tools-v2.3.tar.gz | tar xz

WORKDIR /src/igt-gpu-tools-v2.3
RUN meson setup build && ninja -C build tools/intel_gpu_top

# Stage 2: Runtime image
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pciutils libdrm2 libproc2-0 libkmod2 libpciaccess0 libunwind8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/igt-gpu-tools-v2.3/build/tools/intel_gpu_top /usr/local/bin/intel_gpu_top

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
