#!/usr/bin/env bash
# Run ON THE POD. Render the held-out verses (train/samples.json, 5.68.1–3) with a checkpoint.
#   bash sample.sh <ckpt.pt> [name]   one checkpoint -> $S/<name>/   (name defaults to its step)
#   bash sample.sh --watch            render each new model_<step>.pt as training saves it, then delete
#                                     that 5.4 GB file, keeping its 1.3 GB EMA copy (disk stays small)
set -euo pipefail
source "$(dirname "$0")/env.sh"

step_of() { local b; b=$(basename "$1" .pt); echo "${b#model_}"; }

render() {   # $1 = checkpoint, $2 = output name
  local out=$S/$2 ema=$CK/ema_$2.pt
  mkdir -p "$out"
  [ -f "$ema" ] || $PY $MV/train/export_ema.py "$1" "$ema" >/dev/null
  (cd $UP && $PY src/render.py --shard $MV/train/samples.json --voice "$ema" \
      --bank $MV/data/bank/bank.json --outdir "$out" --results "$out/res.json")
  echo "✓ $2 -> $out"
}

if [ "${1:-}" = "--watch" ]; then
  mkdir -p $CK
  while true; do
    for ck in $CK/model_[0-9]*.pt; do
      [ -e "$ck" ] || continue
      st=$(step_of "$ck")
      [ -f "$S/$st/res.json" ] || { sleep 20; render "$ck" "$st"; }   # sleep: let the save finish
      rm -f "$ck"
    done
    sleep 60
  done
else
  ck=${1:?usage: sample.sh <ckpt.pt> [name] | --watch}
  render "$ck" "${2:-$(step_of "$ck")}"
fi
