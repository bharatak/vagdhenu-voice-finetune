"""Vocoder copy-synthesis test (run ON THE POD): real wav -> F5 vocos-mel -> vocoder -> wav.
No TTS model involved, so any grain in the output comes from the vocoder alone.
    python train/copysynth.py OUTDIR in1.wav [in2.wav ...]
Writes <name>_bigvgan.wav (Prathosh's fine-tuned BigVGAN, the production vocoder) and <name>_vocos.wav.
"""
import sys, os
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio, bigvgan
from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import load_vocoder

VOC = os.environ.get("VOC", "/workspace/vagdhenu/models/voc_bigvgan_EMA_2026-06-11.pth")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
mel = MelSpec(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos")
g = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
sd = torch.load(VOC, map_location="cpu"); g.load_state_dict(sd.get("model", sd)); g.remove_weight_norm(); g = g.cuda().eval()
vocos = load_vocoder("vocos", device="cuda")
for p in sys.argv[2:]:
    y, sr = torchaudio.load(p); y = y.mean(0, keepdim=True)
    if sr != 24000: y = torchaudio.functional.resample(y, sr, 24000)
    m = mel(y).cuda()                                   # (1, 100, T)
    with torch.no_grad():
        b = g(m).squeeze().cpu().numpy(); v = vocos.decode(m).squeeze().cpu().numpy()
    n = Path(p).stem
    sf.write(out / f"{n}_bigvgan.wav", b / max(1, np.abs(b).max()), 24000)
    sf.write(out / f"{n}_vocos.wav", v / max(1, np.abs(v).max()), 24000)
    print("ok", n)
