# Vāgdhenu in my own voice

A hands-on learning experiment: take [Vāgdhenu](https://prathosh.in/vagdhenu/), Prathosh's Sanskrit
chant text-to-speech system, and teach it to chant in **my** voice, while keeping Prathosh's melody.
I record verses on my phone, a cloud GPU fine-tunes the model, and the result runs on my Mac.

The goal was education: to see every step of a modern TTS fine-tune end to end, from recording to
data prep, GPU training, evaluation and on-device inference, and to understand what each step does.

> This repo holds only code and notes. My recordings, the dataset and the trained models stay on my
> machine and are not published.

## What the result is

```bash
PYTHONPATH=. ../vagdhenu/.venv/bin/python -m mv.render --model models/mlx-myvoice \
  --text "अथाहमुत्तरं देव्या पुनरुक्तः ससम्भ्रमः । तव स्नेहान्नरव्याघ्र सौहार्दादनुमान्य वै ॥" \
  --out data/renders/verse.wav
```

Or simply (my model, anuṣṭubh and polish are the defaults; it writes a randomly named wav, prints
a clickable path and starts playing it):

```bash
scripts/chant.sh "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
```

Type any anuṣṭubh verse in Devanagari and get it chanted in my voice to Prathosh's melody, in about
20 s on an M4 Pro. No GPU is needed after training.

## How Vāgdhenu chants (the three parts)

```
text ──► DiT (337 M params, IndicF5 / F5-TTS) ──► mel picture ──► BigVGAN vocoder ──► sound
            ▲                                                          ▲
            reference clip: a few seconds of a voice + its exact text  (mel → waveform)
```

* **DiT (Diffusion Transformer)** draws a *mel spectrogram* (100 frequency bands × 94 columns per
  second) by *flow matching*. Starting from noise, it takes 64 small steps toward a clean picture.
* **Reference clip.** F5 continues a short audio prompt. Hand it a clip of a voice plus that
  clip's exact words, and the new text comes out in that voice and style (in-context cloning).
* **BigVGAN vocoder** turns the mel picture into a waveform.

## What I did

| Step | What happens | Code |
|---|---|---|
| 1. Record | Sundarakāṇḍa sarga 1 (210 verses) on an iPhone, one hemistich per line from an on-screen script | `RECORDING.md`, `mv/script_page.py`, `mv/intake.py` |
| 2. Cut | Split each recording into one clip per hemistich, using silences and the expected syllable counts | `mv/segment.py` |
| 3. Check | Does each clip *sound like* its label? Compare it with Vāgdhenu renders of nearby lines using DTW on log-mel fingerprints | `mv/check.py` |
| 4. Review | A page to listen to each clip, drop bad ones and relabel shifted ones | `mv/review.py` |
| 5. Dataset | 375 clips, **26.8 min**, text in the renderer's exact form (Devanagari → Kannada script) | `mv/dataset.py` |
| 6. Reference bank | The cleanest 4–8 s clip of me per meter (only anuṣṭubh so far) | `mv/bank.py` |
| 7. Zero-shot baseline | Prathosh's model with *my* reference clip: it sounded like me, but drifted at the pāda join (~4 s) | `mv/render.py` |
| 8. Fine-tune the DiT | RunPod A40: 3000 updates, warm-started from Prathosh's checkpoint | `train/finetune.py`, `gpu/` |
| 9. Fix the vocoder | Fine-tune BigVGAN on my voice (a GAN), after a test proved it was adding a buzz | `train/copysynth.py`, `train/finetune_vocoder.py` |
| 10. Bring home | Convert to MLX and render on the Mac, with a light polish filter | `vagdhenu_mlx.convert`, `mv/render.py`, `mv/polish.py` |

Sarga 68 (verses 5.68.1–3) is **held out**. It is never trained on, and every checkpoint is judged
by ear on it. That is how you tell "learned my voice" from "memorised my recordings".

## Training run 1 (2026-11-13)

**DiT fine-tune:** NVIDIA A40 48 GB on RunPod at $0.59/h.
- lr 1e-5: warmup over 200 updates, then linear decay.
- Frame batches of 19 200 mel frames (≈ 205 s ≈ 48 clips), so 8 updates per epoch.
- 3000 updates = each clip seen 375 times.
- EMA weights; 2.0 s per update; GPU memory 44 of 46 GB.

| Update | 1 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| Loss | 0.835 | 0.64 | 0.50 | 0.53 | 0.46 | 0.52 | 0.515 |

The loss flattens by update 1000, so listening decided: **update 3000** sounded most like me.

**The "grrr" and the vocoder fix.** Every render had a faint buzz. A *copy-synthesis* test found
the cause. I took my *real* recording, turned it into a mel picture, and sent it straight through
the vocoder, with no TTS model involved. It still buzzed. So the vocoder was the problem, not the
DiT: Prathosh's BigVGAN was tuned only on his voice. (Stock NVIDIA BigVGAN was worse, because it
expects a different mel format.)

I fine-tuned the vocoder as a **GAN**:
- The generator is Prathosh's vocoder.
- The critics are NVIDIA's pretrained discriminators (MPD for periodic structure, CQT for
  harmonics).
- The loss is 15 × multi-scale mel L1 + adversarial + feature matching.
- 2000 steps, batch 4 × 1.37 s, lr 2e-5, 1.1 s per step.

A lesson from this run: the *mel distance* to my real voice got **worse** (0.173 → 0.21) while the
buzz got **better** by ear. GAN vocoders trade a little spectral distance for crisp harmonics.
Trust your ears over a single metric. I chose vocoder step 1250.

**Polish.** Forwarding a render on WhatsApp made it sound perfect. The Opus codec drops the quiet
fuzz between harmonic lines. `mv/polish.py` does the same on purpose: a soft cut above 8 kHz, plus
a per-moment spectral gate on the inter-harmonic valleys. A test makes sure held vowels aren't
mistaken for noise. It is the default (`--polish none` gives the raw render).

**Cost:** $1.60 of GPU time for both trainings plus sample renders.

## What it can and can't do

* ✅ Anuṣṭubh verses in my voice with Prathosh's melody, on the Mac.
* ⚠️ A little residual "grrr" remains, mostly removed by the polish filter.
* ⚠️ **Other meters:** my bank only has an anuṣṭubh clip. Other meters fall back to it (wrong
  melody), or use `--bank upstream --meter <name>` for Prathosh's reference in that meter (right
  melody, voice pulled toward his). Fix: record a few verses in that meter.
* ⚠️ The meter detector needs the pādas marked. Pass `--meter drutavilambita` (etc.) when it
  prints "fallback".
* ❌ Not speech, and not other languages. It is a chant model.

## Repo layout

```
RECORDING.md          how to record (same room, distance and pace every day)
mv/                   the Mac pipeline: script_page, intake, segment, check, review, dataset, bank, render, polish
train/                GPU scripts: finetune.py (DiT), finetune_vocoder.py, copysynth.py, export_ema.py, samples.json (held-out verses)
gpu/                  RunPod: RUNBOOK.md (step by step, with a cost log), setup_pod.sh, train.sh, sample.sh, env.sh
scripts/chant.sh      one verse → a clickable wav (defaults to my model)
scripts/build_all.sh  recordings → dataset + bank in one command (only new recordings are processed)
tests/                pytest suite
docs/                 design spec and implementation plan
```

## Running it yourself

Needs the upstream repo cloned next to this one (`git clone https://github.com/prathoshap/vagdhenu ../vagdhenu`, set up with its `scripts/setup_mlx.sh`) and
a training venv: `uv venv .venv-train --python 3.10 && uv pip install -r requirements-train.txt`.

```bash
scripts/build_all.sh                       # new recordings in inbox/ → sessions; lists those to review
PYTHONPATH=. ../vagdhenu/.venv/bin/python -m mv.review <session> --open   # listen; then --done
scripts/build_all.sh                       # reviewed sessions → data/dataset + data/bank
.venv-train/bin/python -m pytest -q        # tests
```
GPU training: follow `gpu/RUNBOOK.md`. A full run of DiT + vocoder takes about 2.5 h on an A40, under $2.

## Credits

* **Vāgdhenu** by Prathosh ([github.com/prathoshap/vagdhenu](https://github.com/prathoshap/vagdhenu), Apache 2.0): the model, the chant-specific text frontend, the reference-bank method
  and the released weights. This project only adapts it, and never modifies the upstream repo.
* **IndicF5** (AI4Bharat), built on **F5-TTS**: the DiT.
* **BigVGAN-v2** (NVIDIA): the vocoder architecture and pretrained discriminators.

Model weights fall under their original licences and are not redistributed here.

## Licence

Apache 2.0 (see `LICENSE`), the same as Vāgdhenu.
