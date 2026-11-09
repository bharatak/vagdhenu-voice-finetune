# My-Voice Vāgdhenu — Design

*2026-11-09 · hands-on learning project*

## Intent — why this exists

The user chants Sundarakāṇḍa every day and wants Vāgdhenu (the Sanskrit chant TTS cloned at
`../vagdhenu/`) to chant ślokas **in their own voice**. The deeper goal is **education**: learn how a
modern speech model is fine-tuned by doing every step on a real project — recording, data prep,
GPU training, evaluating, and bringing the model home to run on the Mac — and to understand *why*
each step is done well enough to explain it.

**Success =**
1. Type a Sundarakāṇḍa śloka on the M4 Pro → hear it chanted in the user's voice (MLX backend).
2. The user can explain each stage: recording rules, segmentation & labelling, what fine-tuning
   changes in the model, how to read loss/samples to judge training, checkpoint → MLX conversion,
   and why the reference clip controls melody.

## The user (shapes every explanation)

- Strong in Python; comfortable pair-coding with Claude.
- **Very little ML-training experience** → explain training from first principles (what the model
  learns, what the loss means, why a low learning rate / warm-start), keep code light and readable.
- Has done speech-engine tuning before (FluentEdge) — details unknown; don't assume.
- **Small steps, each ending in something visible/audible.**

## Goals

- Two-track plan:
  - **Track 1 — Recording (background, spread over many days):** daily Sundarakāṇḍa chanting,
    recorded with near-zero extra effort. Long duration here is fine.
  - **Track 2 — Learning/building (compressed, ~2 sittings):** this must be *quick*. Long
    multi-day learning stretches are the main risk — the user will lose interest.
- First GPU run happens **early**, on ~30–60 min of recordings, to see the whole pipeline end to end.
  Later re-training on more data must be a single command, no new learning.
- Every step leaves an artifact the user can see/hear (a clip list, a waveform, a loss curve,
  audio samples per checkpoint, the final chant).

## Non-goals

- Matching Prathosh's MOS 4.6 on the first run. Learning > polish.
- Vedic svaras, prose (gadya), multi-speaker / zero-shot-any-voice.
- Rewriting the model or prosody architecture (the tech report §14 says text-side prosody
  conditioning is inert — don't go there).
- Training on the Mac. Mac = data prep + inference; training = rented CUDA GPU.

## Anti-goals (would be failure even if it "works")

- A black box: scripts the user ran but can't explain.
- Long learning sessions spread over many days.
- Burning many GPU hours on avoidable setup mistakes (everything must be validated on the Mac first;
  GPU time is for training only).
- Modifying the cloned upstream repo (`../vagdhenu/`). Our work lives beside it.

## Constraints & environment

- **Mac:** Apple M4 Pro, 48 GB, macOS. Existing working setup: `../vagdhenu/.venv` (Python 3.12,
  MLX 0.32), weights in `../vagdhenu/models/`, MLX weights in `../vagdhenu/models/mlx/`. Rendering
  verified (RTF ≈ 2 on this chip). Disk is ~90% full (~46 GB free) — keep raw audio + datasets lean;
  `voice_armA…pt`/`voice_steer…pt` are 1.3 GB each.
- **Recording device:** iPhone, Voice Memos set to **Lossless**, on a stand ~20 cm from the mouth,
  same room/position/distance every session, soft-furnished room, fan/AC off, tanpura in headphones
  only (not in the recording). A 30-s test clip is checked (noise floor, echo) before day 1;
  USB mic (~₹5–10k) only if the test is poor.
- **GPU:** rented by the hour when ready (budget: a few hours). Single NVIDIA GPU, ≥ 40 GB
  (A100/H100/L40S class). Upstream trained with frame-batch 12 800–19 200 (~28 GB); size the batch
  to the rented card.
- **Text:** Sundarakāṇḍa Devanagari source text needed (per-sarga, per-verse numbered).

## Decisions made with the user

- **Target = the user's voice (timbre) only. Melody/tune can stay Prathosh's.** Success on listening =
  "sounds like me", not "chants my tune". This makes ~30–60 min of data + a short warm-started
  fine-tune plausible; the per-checkpoint listening check is about timbre.
- **User's voice:** adult male, similar register to Prathosh → reuse the released BigVGAN vocoder
  unchanged. *Still verify on the Mac before GPU time:* mel → BigVGAN re-synthesis of the user's
  test clip must sound clean.
- **Text the user chants is flexible:** they normally read one sarga/day but will chant **whatever is
  shown on screen**. So the builder picks a clean digital Sundarakāṇḍa source and the recording
  script is generated from it — the labels are known *by construction*, not inferred.

- Warm-start fine-tuning from **Prathosh's trained checkpoint**, not base IndicF5 (it already knows
  Sanskrit + chant; training only shifts the voice). *(Claude's call, user agreed.)*
- Recording = their normal daily Sundarakāṇḍa pārāyaṇa; AirDrop to a Mac `inbox/` folder.
- Two learning sittings: (1) Mac: data pipeline + how training works; (2) GPU: train, watch, listen,
  bring home, chant.
- New work goes in `/Users/bharatak/Documents/vagdhenu/myvoice/` (sibling of the clone).
- Documentation/notes format for later recall: **deferred** — "figure out as we progress".

## How it fits together

```
iPhone (lossless m4a, one file per session)
   └─ AirDrop → myvoice/inbox/
        └─ [Mac] segment: split session into verse clips at daṇḍa/verse pauses
             └─ [Mac] align: pair each clip with its verse text (sarga.verse), user spot-checks
                  └─ [Mac] text → model_text (Deva → Kannada routing, upstream prep_text.py)
                       └─ [Mac] build F5 dataset (24 kHz mono wav + metadata → raw.arrow + duration.json)
                            └─ [GPU] fine-tune DiT from Prathosh ckpt (upstream finetune_indicf5.py)
                                 ├─ loss curve + audio samples every N steps
                                 └─ best EMA ckpt → download
                                      └─ [Mac] vagdhenu_mlx.convert → models/mlx-myvoice/
                                           └─ [Mac] own reference bank (1 clean half-śloka per meter)
                                                └─ render_mlx.py → chant in my voice
```

### Recording rules (from tech report §7, simplified for the user)
- Each pāda one breath; **no audible gaps between words**; pause only at daṇḍa (`।` short, `॥` longer).
- Natural pārāyaṇa pace, steady pitch (tanpura in ears), same time of day if possible.
- Hold long vowels, sustain visarga, crisp retroflexes/aspirates.
- Say nothing else mid-file (or a clear verse-number announcement convention — builder decides
  how segmentation maps clips→verses; the user should be told the convention before recording).

### Data constraints (from upstream)
- Clips 4–15 s, 24 kHz mono, trimmed (~0.1 s head/tail), peak < −1 dB.
- Text is the verse/hemistich actually chanted, run through `prep_text.model_text` (Kannada script),
  tokenized with the **same `vocab.txt`** as the released model.
- Half-verse (hemistich) clips are the natural unit (matches how the bank/renderer works).

### Training (from upstream, adapted)
- F5 `Trainer` via `training/finetune_indicf5.py` pattern: DiT dim1024/depth22/heads16, vocos mel
  100-band 24 kHz, lr ~1e-5, short warmup, frequent saves, EMA weights.
- Logging the user can watch live (TensorBoard or W&B) + generated audio samples at intervals.
- Converter `vagdhenu_mlx.convert` reads `ema_model_state_dict` → the saved ckpt must have that key.
- Vocoder: reuse the released fine-tuned BigVGAN unchanged.

## Calls made by Claude (user may override)

- **First GPU run = pipeline check**, training capped at ~1–2 h; a polished voice is not its bar.
- **Scope = Sundarakāṇḍa first.** Other meters in the user's voice later need one dedicated
  reference half-śloka per meter (a small extra recording, announced when needed); until then the
  renderer would fall back to Prathosh's references for those meters.
- **On-screen recording script** (Mac/iPad page showing the numbered verses/hemistichs to chant, in
  order) so segmentation can map clips → known text; the user chants from the screen.
- Our renders use a small wrapper around upstream `Renderer` with our own bank path (upstream
  `render_mlx.py` has no `--bank` flag; don't modify upstream).
- Warm-start loader must accept `ema_model_state_dict` (released ckpts) — upstream
  `finetune_indicf5.py` reads `model_state_dict`; handle in our copy, plus the GRN key quirk
  (`training/indicf5_grnfix.py`).

## Left to the builder

- Exact segmentation method (silence-based split + text alignment vs. ASR-assisted alignment via
  IndicConformer as upstream did) — pick the simplest that works on phone audio; must be
  checkable by ear in a few minutes.
- Which Prathosh checkpoint to warm-start (`voice_armA…` reference-driven vs `voice_steer…`).
- GPU provider, container image, exact batch size, steps, save cadence.
- Source and cleaning of the Sundarakāṇḍa text (user is flexible; pick a clean, numbered edition).
- Clip unit (hemistich vs full verse) once the user's pace is measured — upstream wants clips ≥4 s.
- How the reference bank for the user's voice is recorded/selected.
- Breaking the work into the two sittings with an audible checkpoint at the end of each step.
