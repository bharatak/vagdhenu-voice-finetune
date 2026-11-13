#!/usr/bin/env bash
# Run ON THE POD once, after the Mac has rsynced myvoice/ up:  bash /workspace/myvoice/gpu/setup_pod.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd $W
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
command -v uv   >/dev/null || pip install -q uv
[ -d $UP/.git ] || git clone -q https://github.com/prathoshap/vagdhenu.git $UP
git -C $UP checkout -q $UP_COMMIT
[ -d $UP/BigVGAN/.git ] || git clone -q --depth 1 https://github.com/NVIDIA/BigVGAN.git $UP/BigVGAN

# Same Python + pins as the Mac's .venv-train (PyPI's Linux torch 2.5.1 wheel ships CUDA 12.4).
[ -x $PY ] || uv venv -q $W/venv --python 3.10
uv pip install -q --python $PY -r $MV/requirements-train.txt

# Only the weights we need (skips the 1.3 GB steered checkpoint).
$PY - <<PY
from huggingface_hub import hf_hub_download
for f in ["voice_armA_ema_2026-06-11.pt", "voc_bigvgan_EMA_2026-06-11.pth", "vocab.txt"]:
    print("↓", f); hf_hub_download("prathoshap/vagdhenu", f, local_dir="$UP/models")
PY

# raw.arrow holds Mac paths; point it at the pod's copy.
cd $MV && $PY -m mv.dataset --rebase $MV/data/dataset

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda OK:', torch.cuda.get_device_name(0))"
echo "✓ setup done — next: bash $MV/gpu/train.sh"
