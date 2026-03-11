# Usage:
#   docker build -t intel-gpu-monitor .
#   docker run -d \
#     --name intel-gpu-monitor \
#     --device /dev/dri \
#     --cap-add CAP_PERFMON \
#     -p 8080:8080 \
#     -e GPU_TDP_WATTS=60 \
#     intel-gpu-monitor
#
# Then open http://<host-ip>:8080

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends intel-gpu-tools pciutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
