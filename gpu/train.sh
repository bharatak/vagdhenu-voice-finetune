#!/usr/bin/env bash
# Run ON THE POD: starts training + TensorBoard in tmux (survives SSH disconnects).
#   tmux attach -t train    (watch; detach with Ctrl-b then d)
#   WATCH=1 gpu/train.sh    also render samples as checkpoints arrive — only on a GPU with > 60 GB:
#                           training alone uses 44 of the A40's 46 GB (run 1), so a second model OOMs.
# Resuming a stopped run is NOT supported (the pinned trainer crashes loading model_last.pt):
# start fresh with an empty $CK.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd $MV
MAX=${MAX_UPDATES:-3000}
if compgen -G "$CK/*.pt" >/dev/null; then
  echo "$CK already has checkpoints; the trainer would try to resume from them (unsupported)." >&2
  echo "Bring home what you need, then: rm -rf $CK" >&2; exit 1
fi
tmux new -d -s train "$PY train/finetune.py --warm $UP/models/voice_armA_ema_2026-06-11.pt \
  --data_dir $MV/data/dataset --save_dir $CK --vocab $UP/models/vocab.txt \
  --bs 19200 --save_per 500 --max_updates $MAX 2>&1 | tee -a $W/train.log"
tmux new -d -s tb "$W/venv/bin/tensorboard --logdir $MV/runs --host 0.0.0.0 --port 6006"
[ "${WATCH:-0}" = 1 ] && tmux new -d -s samples "bash $MV/gpu/sample.sh --watch 2>&1 | tee -a $W/samples.log"
tmux ls
