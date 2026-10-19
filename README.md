# tts-workers

Runpod Serverless voice-cloning workers that all speak one contract, so any model
can be swapped into the same client. Weights are baked into each image.

| Folder | Model | Licence | Image |
|---|---|---|---|
| chatterbox | Chatterbox Multilingual V3 | MIT | ghcr.io/drhouse/tts-chatterbox |
| qwen3tts | Qwen3-TTS 12Hz 1.7B Base | Apache-2.0 | ghcr.io/drhouse/tts-qwen3tts |
| cosyvoice2 | CosyVoice 2 0.5B (cross-lingual mode) | Apache-2.0 | ghcr.io/drhouse/tts-cosyvoice2 |
| openvoice2 | OpenVoice V2 + MeloTTS EN base | MIT | ghcr.io/drhouse/tts-openvoice2 |

Request (queue endpoint, `/runsync`):

```json
{"input": {"language": "en", "items": [{"id": "x", "text": "Hello", "ref_b64": "<wav>", "ref_text": "<transcript>"}]}}
```

Response: `{"results": [{"id", "wav_b64", "seconds"}], "device", "elapsed"}`, or
`{"error"}` for every job if the model failed to load (never a silent crash loop).

To add a model: a folder with a `Dockerfile` and a `handler.py` exposing
`load()` and `synth(model, text, ref_path, ref_text, language) -> (wav, rate)`,
then add the folder to the build matrix.

Deploy new images at `workers.max = 1`, prove one real job, then scale.
