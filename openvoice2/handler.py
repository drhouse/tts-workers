"""OpenVoice V2 (MyShell, MIT). Two stages: MeloTTS speaks the line with a native English
base voice, then the tone-colour converter moves only the timbre onto the reference voice.
Delivery and accent come from the English base speaker, not the reference."""
import os
import sys
import tempfile

sys.path.insert(0, "/app")
from serve import start  # noqa: E402

CKPT = "/opt/checkpoints_v2"
BASE = os.environ.get("OPENVOICE_BASE", "EN-Newest")  # MeloTTS speaker key


def load():
    import torch
    from melo.api import TTS
    from openvoice.api import ToneColorConverter
    conv = ToneColorConverter(f"{CKPT}/converter/config.json", device="cuda:0")
    conv.load_ckpt(f"{CKPT}/converter/checkpoint.pth")
    melo = TTS(language="EN_NEWEST", device="cuda:0")
    src_se = torch.load(f"{CKPT}/base_speakers/ses/{BASE.lower().replace('_', '-')}.pth", map_location="cuda:0")
    return conv, melo, src_se


def synth(model, text, ref, ref_text, language):
    import soundfile as sf
    conv, melo, src_se = model
    # Voice embedding from the whole reference clip. se_extractor.get_se first splits
    # long audio with faster-whisper (needs cuBLAS) or Silero VAD (a torch.hub download
    # that fails at run time); a game line reference is a few seconds, so skip the split.
    tgt_se = conv.extract_se([ref])
    work = tempfile.mkdtemp()
    base, out = os.path.join(work, "base.wav"), os.path.join(work, "out.wav")
    melo.tts_to_file(text, melo.hps.data.spk2id[BASE], base, speed=1.0)
    conv.convert(audio_src_path=base, src_se=src_se, tgt_se=tgt_se, output_path=out)
    wav, rate = sf.read(out, dtype="float32")
    return wav, rate


if __name__ == "__main__":
    start(load, synth)
