"""CosyVoice 2 0.5B (Alibaba, Apache-2.0). inference_cross_lingual: target-language text, reference in any language."""
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/opt/CosyVoice")
sys.path.insert(0, "/opt/CosyVoice/third_party/Matcha-TTS")
from serve import start  # noqa: E402


def load():
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(model_dir="/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B")


def synth(model, text, ref, ref_text, language):
    import torch
    chunks = [j["tts_speech"] for j in model.inference_cross_lingual(f"<|{language}|>{text}", ref, stream=False)]
    return torch.cat(chunks, dim=1).squeeze(0).cpu().numpy(), model.sample_rate


if __name__ == "__main__":
    start(load, synth)
