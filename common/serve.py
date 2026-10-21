"""Shared Runpod Serverless wrapper: every model in this repo speaks one contract.

Input:  {"input": {"items": [{"id", "text", "ref_b64", "ref_text"}, ...], "language": "en"}}
Output: {"results": [{"id", "wav_b64", "seconds"} | {"id", "error"}], "device", "elapsed"}

A model module only provides load() and synth(model, text, ref_path, ref_text, language)
-> (float32 numpy array, sample_rate). It may also provide synth_batch(model, texts,
ref_paths, ref_texts, language) -> (list of arrays, sample_rate); a job with
"batch": N then sends N lines per GPU call instead of one. Load failures are reported through every job
instead of crash-looping a worker that looks "ready" while it bills.
"""
import base64
import io
import os
import tempfile
import time
import traceback

import runpod
import soundfile as sf


def start(load, synth, synth_batch=None):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "CPU-FALLBACK"
    except Exception:
        device = "unknown"
    try:
        model, load_error = load(), None
    except Exception as exc:
        model, load_error = None, f"load failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
        print(load_error, flush=True)

    try:
        import torch
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = "unknown"
    settings = {k: v for k, v in os.environ.items() if k.startswith("TTS_")}

    def handler(job):
        if load_error:
            return {"error": load_error, "device": device}
        inp = job["input"]
        language = inp.get("language", "en")
        started, out = time.time(), []
        items = inp["items"]
        size = int(inp.get("batch", 1)) if synth_batch else 1
        for at in range(0, len(items), size):
            chunk = items[at:at + size]
            refs = []
            try:
                for item in chunk:
                    handle, ref = tempfile.mkstemp(suffix=".wav")
                    os.close(handle)
                    refs.append(ref)
                    with open(ref, "wb") as fh:
                        fh.write(base64.b64decode(item["ref_b64"]))
                if size > 1:
                    wavs, rate = synth_batch(model, [i["text"] for i in chunk], refs,
                                             [i.get("ref_text", "") for i in chunk], language)
                else:
                    wav, rate = synth(model, chunk[0]["text"], refs[0], chunk[0].get("ref_text", ""), language)
                    wavs = [wav]
                for item, wav in zip(chunk, wavs):
                    buf = io.BytesIO()
                    sf.write(buf, wav, rate, format="WAV", subtype="PCM_16")
                    out.append({"id": item["id"], "wav_b64": base64.b64encode(buf.getvalue()).decode(),
                                "seconds": round(len(wav) / rate, 3)})
            except Exception as exc:  # one bad chunk must not cost the job
                out.extend({"id": i.get("id"), "error": f"{type(exc).__name__}: {exc}"} for i in chunk)
            finally:
                for ref in refs:
                    os.unlink(ref)
        return {"results": out, "device": device, "elapsed": round(time.time() - started, 2),
                "gpu": gpu, "batch": size, "settings": settings}

    runpod.serverless.start({"handler": handler})
