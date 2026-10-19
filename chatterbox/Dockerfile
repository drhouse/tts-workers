FROM python:3.11-slim
ENV HF_HOME=/models PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# V3 (t3_model=) is only in the source tree; the PyPI release predates it.
RUN pip install --no-cache-dir git+https://github.com/resemble-ai/chatterbox.git runpod soundfile
# Bake weights (CPU load at build: the runner has no GPU).
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu', t3_model='v3')"
COPY common/serve.py chatterbox/handler.py ./
CMD ["python", "-u", "handler.py"]
