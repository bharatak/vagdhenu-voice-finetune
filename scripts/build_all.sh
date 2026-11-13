#!/usr/bin/env bash
# One command from recordings to a training set. Safe to re-run: only NEW recordings are processed.
#
#   scripts/build_all.sh
#
#   1. segment  every inbox/sarga_*.m4a that has no data/sessions/<name>/clips.json yet
#   2. check    every session without check.json (content check: does each clip sound like its line?)
#   3. dataset  rebuild data/dataset/ from the kept clips of REVIEWED sessions (held-out verses excluded)
#   4. bank     rebuild your reference bank from reviewed sessions (one clean clip of you per meter)
#   5. list     every session still waiting for your review (mv.review <name> --open, then --done)
#
# Then retrain on a GPU: gpu/RUNBOOK.md ("Retrain later").
set -euo pipefail
cd "$(dirname "$0")/.."

UP=${VAGDHENU_UPSTREAM:-../vagdhenu}
MLXPY=${MLXPY:-$UP/.venv/bin/python}        # segment / check / bank (renders with MLX)
TRAINPY=${TRAINPY:-.venv-train/bin/python}  # dataset (needs the 'datasets' package)
export PYTHONPATH=".:$UP/src${PYTHONPATH:+:$PYTHONPATH}"

new=()
shopt -s nullglob
for src in inbox/sarga_*.m4a; do
  name=$(basename "$src" .m4a)
  if [ ! -f "data/sessions/$name/clips.json" ]; then
    echo "── segment $name"
    "$MLXPY" -m mv.segment "$src"
    new+=("$name")
  fi
done

for d in data/sessions/*/; do
  name=$(basename "$d")
  [ -f "$d/clips.json" ] || continue
  if [ ! -f "$d/check.json" ]; then
    echo "── check $name"
    "$MLXPY" -m mv.check "$name"
  fi
done

echo "── dataset"
"$TRAINPY" -m mv.dataset
echo "── bank"
"$MLXPY" -m mv.bank

echo
echo "${#new[@]} new session(s)."
for d in data/sessions/*/; do
  name=$(basename "$d")
  [ -f "$d/clips.json" ] && [ ! -f "$d/reviewed" ] || continue
  echo "  NOT REVIEWED (not in the dataset yet): $name"
  echo "    PYTHONPATH=. $MLXPY -m mv.review $name --open   # listen; drop.txt / relabel.txt"
  echo "    PYTHONPATH=. $MLXPY -m mv.review $name --done   # then re-run this script"
done
