# CLAUDE.md

An educational project: fine-tune Vāgdhenu (Sanskrit chant TTS: IndicF5/F5-TTS DiT + BigVGAN-v2
vocoder) on the owner's chanting voice. README.md has the story and the results. The owner is
learning, so explain each step briefly and give them something to listen to after each change.

## Hard rules
- **Never modify the upstream repo** `../vagdhenu` (`mv/paths.py` → `UPSTREAM`, overridable with
  `VAGDHENU_UPSTREAM`). Import from it (`mv.paths` puts `src/` on sys.path); wrap, don't patch.
- **Never commit or publish personal audio or weights**: `data/`, `models/`, `inbox/` and
  `originals/` are gitignored. Keep it that way. No pod IPs, keys or tokens in tracked files.
- **Sarga 68 is held out** (`train/samples.json`: 5.68.1–3). `mv.dataset` excludes it; don't record
  it for training, and judge checkpoints on it.
- **Judge audio by ear, not by a metric alone.** In run 1 the vocoder's mel distance rose while the
  buzz fell. Pick checkpoints by listening, and render a comparison file with beeps between takes.
- Billable cloud resources (RunPod): state the hourly price before creating; delete only pods this
  conversation created; confirm the pod list is empty afterwards. Log cost in `gpu/RUNBOOK.md`.

## Two Python environments
- **MLX / rendering:** `../vagdhenu/.venv/bin/python` (has `mlx`). Use it for `mv.render`,
  `mv.check`, `mv.segment`, `mv.bank`, `mv.review`. Run with `PYTHONPATH=.` from this folder.
- **Training / tests:** `.venv-train/bin/python` (torch, f5_tts, `datasets`). Use it for
  `mv.dataset`, `vagdhenu_mlx.convert` (with `PYTHONPATH=../vagdhenu/src`) and `pytest`.

## Commands
```bash
.venv-train/bin/python -m pytest -q                       # all tests (must stay green)
scripts/chant.sh "<verse>"                                # quickest: my model → data/renders/chant_*.wav, opens it
scripts/build_all.sh                                      # inbox/*.m4a → sessions → dataset + bank (idempotent)
PYTHONPATH=. ../vagdhenu/.venv/bin/python -m mv.render --model models/mlx-myvoice --text "<verse । … ॥>" --out x.wav
#   --meter <name>   force the meter (detector needs pādas)   --bank upstream   Prathosh's references
#   --polish filter|opus|none   (filter is the default)       --seed N   another take
PYTHONPATH=../vagdhenu/src .venv-train/bin/python -m vagdhenu_mlx.convert --voice models/run1/ema_<N>.pt \
  --vocoder models/voc/voc_<V>.pth --vocab ../vagdhenu/models/vocab.txt --out models/mlx-<name>
```

## Current models (local, not in git)
- `models/mlx-myvoice/`: **production**. DiT update 3000 + own vocoder step 1250.
- `models/mlx-run1-3000/`: the same DiT with Prathosh's vocoder (buzzes; kept as a backup).
- `models/run1/ema_{500..3000}.pt`: DiT EMA checkpoints. `models/voc/voc_{1250,2000}.pth`: vocoders.

## Gotchas learned the hard way
- Text must go through the renderer's exact chain (`mv.dataset.train_text`: Devanagari → SLP1 →
  Kannada + fixes). Any character outside the released `vocab.txt` is unseen by the model.
- Half-reference rule: a bank entry's `ref_text` must match its wav exactly, or the render drifts.
- The vocoder must be one trained on **F5 vocos-mel**. Stock NVIDIA BigVGAN expects a different mel
  and sounds broken; Prathosh's (or ours) is required. A new voice needs its own vocoder fine-tune.
- `train/finetune_vocoder.py`: batch 8 runs out of memory on a 48 GB A40; use `--bs 4`. It needs
  `nnAudio` (in `requirements-train.txt`).
- Upstream `src/render.py` (torch, on the pod) is CUDA-only, needs `no_sandhi` in each clip and an
  existing `--outdir`.
- RunPod: `/workspace` is a network filesystem, so `rsync --no-owner --no-group`. Use direct TCP SSH
  (the proxy can't transfer files). Put SSH options in a config file (zsh doesn't word-split `$VAR`).
  Training uses 44 of 46 GB, so don't render alongside it.
- `git push` to GitHub: plain git isn't wired to gh's login; use
  `git -c credential.helper='!gh auth git-credential' push`, or have the owner run `gh auth setup-git`.

## Layout
`mv/` Mac pipeline · `train/` GPU training · `gpu/` RunPod scripts + RUNBOOK · `scripts/` build_all ·
`tests/` pytest · `docs/` spec + plan · `RECORDING.md` recording guide.
