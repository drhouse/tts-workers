"""Qwen3-TTS 1.7B Base (Alibaba, Apache-2.0). Voice clone from reference audio + its transcript."""
import sys

sys.path.insert(0, "/app")
from serve import start  # noqa: E402

LANG = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}


def load():
    import torch
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                         device_map="cuda:0", dtype=torch.bfloat16)


def synth(model, text, ref, ref_text, language):
    wavs, sr = model.generate_voice_clone(text=text, language=LANG.get(language, "Auto"),
                                          ref_audio=ref, ref_text=ref_text)
    return wavs[0], sr


if __name__ == "__main__":
    start(load, synth)
