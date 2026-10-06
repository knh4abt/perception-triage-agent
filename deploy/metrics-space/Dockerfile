# The metrics tools in a small box for a free cloud host (Render): a web page people can
# click and an MCP server agents can call, both from app.py. No YOLO, no torch, no LLM:
# it only serves the numbers in reports/metrics.json.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY triage_agent/ triage_agent/
# Results of one local run; the app reads them, it never computes them.
COPY reports/ reports/
COPY app.py .

# 0.0.0.0 = accept visitors from outside the box, not only from inside it.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PYTHONUNBUFFERED=1

# The host tells the app which port to use through $PORT (Render: 10000); 7860 locally.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python app.py"]
