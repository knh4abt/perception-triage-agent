# Runs the full pipeline in a container. The LLM (Ollama) stays on the host.
#
#   docker build -t perception-triage .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" perception-triage
#
# On Linux also pass --add-host=host.docker.internal:host-gateway so the container can reach Ollama.

FROM python:3.11-slim

# OpenCV, pulled in by ultralytics, needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default Linux wheel bundles CUDA (~2 GB) that this image never uses.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Requirements before code: Docker caches this layer, so editing code does not reinstall packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image so a run needs no download.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY configs/ configs/
COPY scripts/ scripts/
COPY triage_agent/ triage_agent/

# ChatOllama reads OLLAMA_HOST; host.docker.internal is the host machine seen from the container.
ENV OLLAMA_HOST=http://host.docker.internal:11434

ENTRYPOINT ["python", "-m", "triage_agent.cli"]
CMD ["--images", "data/sample", "--out", "reports"]
