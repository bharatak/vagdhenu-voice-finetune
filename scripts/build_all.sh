#!/usr/bin/env bash
# One command from recordings to a training set. Safe to re-run: only NEW recordings are processed.
#
#   scripts/build_all.sh
#
#   1. segment  every inbox/sarga_*.m4a that has no data/sessions/<name>/clips.json yet
#   2. check    every session without check.json (content check: does each clip sound like its line?)
#   3. list     the new sessions you still need to review by ear (mv.review <name> --open)
#   4. dataset  rebuild data/dataset/ from all kept clips (held-out verses in train/samples.json excluded)
#   5. bank     rebuild your reference bank (one clean clip of you per meter)
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
for n in ${new[@]+"${new[@]}"}; do
  echo "  review by ear, then re-run this script:  PYTHONPATH=. $MLXPY -m mv.review $n --open"
done
