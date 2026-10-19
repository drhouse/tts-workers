FROM python:3.12-slim
ENV HF_HOME=/models PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir qwen-tts runpod soundfile
RUN python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-TTS-12Hz-1.7B-Base'); d('Qwen/Qwen3-TTS-Tokenizer-12Hz')"
COPY common/serve.py qwen3tts/handler.py ./
CMD ["python", "-u", "handler.py"]
