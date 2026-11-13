"""Fine-tune the BigVGAN-v2 vocoder (mel picture -> sound) on YOUR recordings. Run ON THE POD.

    python train/finetune_vocoder.py --data_dir $MV/data/dataset --save_dir $W/voc --eval $MV/data/originals_wav

Why: the copy-synthesis test (train/copysynth.py) showed Prathosh's vocoder adds a 'grrr' to your
voice even when it is handed your REAL mel — it was tuned only on his voice. The DiT is fine.

How (a GAN, the way NVIDIA trains BigVGAN — BigVGAN/train.py, v2 config):
  * generator  = Prathosh's vocoder (already speaks F5's vocos-mel, which stock BigVGAN does not);
  * critics    = NVIDIA's pretrained discriminators (MPD: periodic structure; CQT: pitch/harmonics),
                 they learn to tell your real audio from the generator's;
  * generator loss = 15 x multi-scale mel L1 (sound right) + adversarial + 2 x feature matching
                 (fool the critics -> crisp harmonics, no fuzz).
Every --save_per steps it saves G and re-synthesises the --eval wavs (held-out verses), printing
their mel distance to the originals: lower = closer to your real voice.
"""
import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np

SR, HOP, SEG = 24000, 256, 32768          # 32768 samples = 128 mel frames = 1.37 s
VOC_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"


def crop(a: np.ndarray, seg: int, rng) -> np.ndarray:
    """A random `seg`-sample window of the clip (zero-padded if the clip is shorter)."""
    if len(a) >= seg:
        s = int(rng.integers(0, len(a) - seg + 1))
        return a[s:s + seg]
    return np.pad(a, (0, seg - len(a)))


def main():
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio
    from torch.utils.tensorboard import SummaryWriter
    from huggingface_hub import hf_hub_download
    import bigvgan
    from discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator
    from loss import MultiScaleMelSpectrogramLoss, feature_loss, generator_loss, discriminator_loss
    from f5_tts.model.modules import MelSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--warm", default="/workspace/vagdhenu/models/voc_bigvgan_EMA_2026-06-11.pth")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save_per", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.save_per, a.bs = 10, 5, 2
    dev = "cuda"
    save = Path(a.save_dir); save.mkdir(parents=True, exist_ok=True)

    # generator: BigVGAN-v2 architecture, Prathosh's vocos-mel weights (weight-norm form = trainable)
    g = bigvgan.BigVGAN.from_pretrained(VOC_REPO, use_cuda_kernel=False)
    sd = torch.load(a.warm, map_location="cpu"); g.load_state_dict(sd.get("model", sd))
    h = g.h; g = g.to(dev).train()
    # critics: NVIDIA's pretrained discriminators
    mpd, cqtd = MultiPeriodDiscriminator(h).to(dev), MultiScaleSubbandCQTDiscriminator(h).to(dev)
    dsd = torch.load(hf_hub_download(VOC_REPO, "bigvgan_discriminator_optimizer.pt"), map_location="cpu")
    mpd.load_state_dict(dsd["mpd"]); cqtd.load_state_dict(dsd["mrd"]); mpd.train(); cqtd.train()
    opt_g = torch.optim.AdamW(g.parameters(), a.lr, betas=(h.adam_b1, h.adam_b2))
    opt_d = torch.optim.AdamW(itertools.chain(mpd.parameters(), cqtd.parameters()), a.lr, betas=(h.adam_b1, h.adam_b2))
    mel_loss = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(dev)
    f5mel = MelSpec(n_fft=1024, hop_length=HOP, win_length=1024, n_mel_channels=100,
                    target_sample_rate=SR, mel_spec_type="vocos").to(dev)

    wavs = sorted(Path(a.data_dir, "wavs").glob("*.wav"))
    audio = [sf.read(p, dtype="float32")[0] for p in wavs]
    audio = [x.mean(1) if x.ndim > 1 else x for x in audio]
    evals = sorted(Path(a.eval).glob("*.wav")) if a.eval else []
    print(f"[voc] {len(audio)} training clips, {sum(map(len, audio)) / SR / 60:.1f} min; "
          f"{len(evals)} eval wavs; bs {a.bs} x {SEG / SR:.2f} s; {a.steps} steps", flush=True)
    rng = np.random.default_rng(0)
    tb = SummaryWriter(str(save / "tb"))

    def evaluate(step):
        g.eval(); out = save / f"eval_{step}"; out.mkdir(exist_ok=True); dists = []
        with torch.no_grad():
            for p in evals:
                y, sr = torchaudio.load(str(p)); y = y.mean(0, keepdim=True)
                if sr != SR: y = torchaudio.functional.resample(y, sr, SR)
                m = f5mel(y.to(dev)); yh = g(m).squeeze(0)[..., :y.shape[-1]]
                dists.append(F.l1_loss(f5mel(yh), m).item())
                sf.write(out / p.name, yh.squeeze().cpu().numpy(), SR)
        g.train()
        if dists:
            print(f"[eval {step}] mel distance to your real voice: {np.mean(dists):.4f}", flush=True)
            tb.add_scalar("eval/mel_l1", np.mean(dists), step)

    evaluate(0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        y = torch.from_numpy(np.stack([crop(audio[i], SEG, rng) for i in rng.integers(0, len(audio), a.bs)]))
        y = y.to(dev).unsqueeze(1)                                  # (B,1,SEG)
        with torch.no_grad():
            m = f5mel(y.squeeze(1))                                 # (B,100,T) — what the DiT draws
        yg = g(m)[..., :SEG]
        # 1) critics: real -> 1, generated -> 0
        r, f_, _, _ = mpd(y, yg.detach()); ld_f, _, _ = discriminator_loss(r, f_)
        r, f_, _, _ = cqtd(y, yg.detach()); ld_s, _, _ = discriminator_loss(r, f_)
        opt_d.zero_grad(); (ld_f + ld_s).backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(mpd.parameters(), cqtd.parameters()), h.clip_grad_norm)
        opt_d.step()
        # 2) generator: sound right (mel) + fool the critics
        lm = mel_loss(y, yg) * h.lambda_melloss
        _, f_, fr, fg = mpd(y, yg); lfm = feature_loss(fr, fg); la, _ = generator_loss(f_)
        _, f_, fr, fg = cqtd(y, yg); lfm = lfm + feature_loss(fr, fg); la2, _ = generator_loss(f_)
        lg = lm + lfm + la + la2
        opt_g.zero_grad(); lg.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), h.clip_grad_norm); opt_g.step()
        if step % 20 == 0 or a.smoke:
            vals = dict(d=(ld_f + ld_s).item(), mel=lm.item(), fm=lfm.item(), adv=(la + la2).item())
            print(f"step {step}: " + " ".join(f"{k}={v:.3f}" for k, v in vals.items())
                  + f"  {(time.time() - t0) / step:.2f}s/step", flush=True)
            for k, v in vals.items(): tb.add_scalar(f"train/{k}", v, step)
        if step % a.save_per == 0 or step == a.steps:
            torch.save({"model": g.state_dict(), "step": step}, save / f"voc_{step}.pth")
            evaluate(step)
    print("done", flush=True)


if __name__ == "__main__":
    main()
