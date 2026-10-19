"""Chatterbox Multilingual V3 (Resemble AI, MIT). Cross-lingual: English text, any-language reference."""
import sys

sys.path.insert(0, "/app")
from serve import start  # noqa: E402


def load():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")


def synth(model, text, ref, ref_text, language):
    wav = model.generate(text, language_id=language, audio_prompt_path=ref)
    return wav.squeeze(0).cpu().numpy(), model.sr


if __name__ == "__main__":
    start(load, synth)
