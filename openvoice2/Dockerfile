FROM python:3.10-slim
ENV HF_HOME=/models PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg unzip wget build-essential && rm -rf /var/lib/apt/lists/*
# OpenVoice pins av==10.0.0 (no wheel; will not compile against current ffmpeg) and other
# 2023-era versions, so it goes in without deps and gets a current set. setuptools<70
# keeps pkg_resources available to old sdists, including inside isolated builds.
RUN echo "setuptools<70" > /tmp/c.txt && export PIP_CONSTRAINT=/tmp/c.txt \
 && pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git runpod soundfile \
 && pip install --no-cache-dir --no-deps git+https://github.com/myshell-ai/OpenVoice.git \
 && pip install --no-cache-dir faster-whisper whisper-timestamped wavmark pydub eng_to_ipa inflect unidecode pypinyin cn2an jieba langid librosa \
 && python -m unidic download
RUN wget -q https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -O /tmp/ckpt.zip \
 && unzip -q /tmp/ckpt.zip -d /opt && rm /tmp/ckpt.zip && ls /opt/checkpoints_v2
# Bake MeloTTS weights + NLTK data by speaking once on CPU at build time.
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')" \
 && python -c "from melo.api import TTS; m=TTS(language='EN_NEWEST', device='cpu'); m.tts_to_file('Warm up.', m.hps.data.spk2id['EN-Newest'], '/tmp/w.wav')"
WORKDIR /app
COPY common/serve.py openvoice2/handler.py ./
CMD ["python", "-u", "handler.py"]
