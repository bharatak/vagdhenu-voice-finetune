#!/usr/bin/env bash
# Run ON THE POD: starts training + TensorBoard + the sample watcher in tmux (survives SSH disconnects).
#   tmux attach -t train    (watch; detach with Ctrl-b then d)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd $MV
MAX=${MAX_UPDATES:-3000}
tmux new -d -s train "$PY train/finetune.py --warm $UP/models/voice_armA_ema_2026-06-11.pt \
  --data_dir $MV/data/dataset --save_dir $CK --vocab $UP/models/vocab.txt \
  --bs 19200 --save_per 500 --max_updates $MAX 2>&1 | tee -a $W/train.log"
tmux new -d -s tb "$W/venv/bin/tensorboard --logdir $MV/runs --host 0.0.0.0 --port 6006"
tmux new -d -s samples "bash $MV/gpu/sample.sh --watch 2>&1 | tee -a $W/samples.log"
tmux ls
