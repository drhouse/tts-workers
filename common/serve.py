"""Shared Runpod Serverless wrapper: every model in this repo speaks one contract.

Input:  {"input": {"items": [{"id", "text", "ref_b64", "ref_text"}, ...], "language": "en"}}
Output: {"results": [{"id", "wav_b64", "seconds"} | {"id", "error"}], "device", "elapsed"}

A model module only provides load() and synth(model, text, ref_path, ref_text, language)
-> (float32 numpy array, sample_rate). Load failures are reported through every job
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


def start(load, synth):
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

    def handler(job):
        if load_error:
            return {"error": load_error, "device": device}
        inp = job["input"]
        language = inp.get("language", "en")
        started, out = time.time(), []
        for item in inp["items"]:
            handle, ref = tempfile.mkstemp(suffix=".wav")
            os.close(handle)
            try:
                with open(ref, "wb") as fh:
                    fh.write(base64.b64decode(item["ref_b64"]))
                wav, rate = synth(model, item["text"], ref, item.get("ref_text", ""), language)
                buf = io.BytesIO()
                sf.write(buf, wav, rate, format="WAV", subtype="PCM_16")
                out.append({"id": item["id"], "wav_b64": base64.b64encode(buf.getvalue()).decode(),
                            "seconds": round(len(wav) / rate, 3)})
            except Exception as exc:  # one bad line must not cost the batch
                out.append({"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            finally:
                os.unlink(ref)
        return {"results": out, "device": device, "elapsed": round(time.time() - started, 2)}

    runpod.serverless.start({"handler": handler})
