FROM python:3.10-slim
ENV HF_HOME=/models PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg sox libsox-dev build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice
WORKDIR /opt/CosyVoice
# Some sdists import pkg_resources in setup.py, which setuptools >= 70 dropped; the
# constraint also applies inside pip's isolated build environments.
RUN echo "setuptools<70" > /tmp/c.txt && PIP_CONSTRAINT=/tmp/c.txt pip install --no-cache-dir -r requirements.txt \n && pip install --no-cache-dir runpod soundfile
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"
WORKDIR /app
COPY common/serve.py cosyvoice2/handler.py ./
CMD ["python", "-u", "handler.py"]
