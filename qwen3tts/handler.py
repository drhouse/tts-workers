"""Qwen3-TTS 1.7B Base (Alibaba, Apache-2.0). Voice clone from reference audio + its transcript.

Speed settings come from endpoint env so one image serves every test:
  TTS_ATTN     attention kernel: sdpa | eager | flash_attention_2 (default: library default)
  TTS_COMPILE  "1" wraps the talker in torch.compile
"""
import os
import sys

sys.path.insert(0, "/app")
from serve import start  # noqa: E402

LANG = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}


def load():
    import torch
    from qwen_tts import Qwen3TTSModel
    kw = {"attn_implementation": os.environ["TTS_ATTN"]} if os.environ.get("TTS_ATTN") else {}
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                          device_map="cuda:0", dtype=torch.bfloat16, **kw)
    if os.environ.get("TTS_COMPILE") == "1":
        # Compile the talker's forward only; generate() and the tokenizer stay eager.
        talker = model.model.talker if hasattr(model.model, "talker") else model.model
        talker.forward = torch.compile(talker.forward, dynamic=True)
    return model


def synth(model, text, ref, ref_text, language):
    wavs, sr = model.generate_voice_clone(text=text, language=LANG.get(language, "Auto"),
                                          ref_audio=ref, ref_text=ref_text)
    return wavs[0], sr


def synth_batch(model, texts, refs, ref_texts, language):
    lang = LANG.get(language, "Auto")
    wavs, sr = model.generate_voice_clone(text=texts, language=[lang] * len(texts),
                                          ref_audio=refs, ref_text=ref_texts)
    return list(wavs), sr


if __name__ == "__main__":
    start(load, synth, synth_batch)
