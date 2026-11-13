#!/usr/bin/env bash
# Chant an anuṣṭubh verse in my voice -> a .wav you can click (and it starts playing).
#
#   scripts/chant.sh "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
#   echo "<verse>" | scripts/chant.sh          # or pipe the text in
#
# Output: data/renders/chant_<date>_<random>.wav (path printed). Defaults: my model
# (models/mlx-myvoice), my reference bank, anuṣṭubh, filter polish. Override with env vars:
#   MODEL=models/mlx-run1-3000  METER=drutavilambita  SEED=61  OPEN=0  scripts/chant.sh "<verse>"
set -euo pipefail
cd "$(dirname "$0")/.."

text=${1:-$(cat)}
[ -n "${text// /}" ] || { echo "usage: scripts/chant.sh \"<verse with । and ॥>\"" >&2; exit 1; }

UP=${VAGDHENU_UPSTREAM:-../vagdhenu}
PY=${PY:-$UP/.venv/bin/python}
MODEL=${MODEL:-models/mlx-myvoice}
METER=${METER:-anushtubh}
SEED=${SEED:-60}
[ -d "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

out="data/renders/chant_$(date +%Y%m%d_%H%M%S)_$(printf %04x $RANDOM).wav"
PYTHONPATH=".:$UP/src" "$PY" -m mv.render --model "$MODEL" --meter "$METER" --seed "$SEED" \
  --text "$text" --out "$out" >/dev/null

abs="$(pwd)/$out"
echo "$abs"
echo "file://$abs"
[ "${OPEN:-1}" = 1 ] && open "$abs" || true
