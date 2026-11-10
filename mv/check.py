"""Step 3b: check each clip's *content* — does it sound like its own line, or a neighbour's?

    python -m mv.check sarga_01_v101_187

The line-up (mv.segment) only knows durations, so a repeated or skipped line among same-length
lines shifts every label after it. This check listens instead:
  1. render every candidate line with Vāgdhenu (Prathosh's voice, cached in data/check/renders/),
  2. turn clip and rendering into log-mel "fingerprints" (40 bands, 100 per second, each band
     normalised so the speaker and loudness matter less than *which sounds come in which order*),
  3. dynamic time warping: stretch/squeeze time to line them up; the leftover mismatch is the cost,
  4. the clip "sounds like" whichever of its label ±3 lines costs least.
Validated against 15 clips labelled by ear (7 shifted, 8 correct): 15/15.
Results go to check.json and appear on the review page.
"""
import argparse
import json
import re

import librosa
import numpy as np
import soundfile as sf

from mv import paths

FEAT_SR = 16000
WINDOW = 3
MIN_MARGIN = 0.05        # below this the check is a near tie: drop rather than guess
RENDERS = paths.DATA / "check" / "renders"


def features(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if sr != FEAT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=FEAT_SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    m = np.log(librosa.feature.melspectrogram(y=y, sr=FEAT_SR, n_mels=40, hop_length=160) + 1e-6)
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-6)


def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    D, wp = librosa.sequence.dtw(a, b, metric="cosine")
    return float(D[-1, -1] / len(wp))


def suggest(rows, hems, feat_of_clip, feat_of_line, window: int = WINDOW) -> list:
    """For each clip row, rank its label ±window lines (in sarga order) by DTW cost."""
    index = {h.id: i for i, h in enumerate(hems)}
    out = []
    for r in rows:
        i = index[r["id"]]
        cf = feat_of_clip(r)
        costs = {h.id: dtw_cost(cf, feat_of_line(h)) for h in hems[max(0, i - window): i + window + 1]}
        ranked = sorted(costs, key=costs.get)
        best = ranked[0]
        runner = costs[ranked[1]] if len(ranked) > 1 else costs[best]
        margin = (runner - costs[best]) if best == r["id"] else (costs[r["id"]] - costs[best])
        out.append({"id": r["id"], "best": best, "shifted": best != r["id"], "margin": round(margin, 3),
                    "label_cost": round(costs[r["id"]], 3), "best_cost": round(costs[best], 3)})
    return out


def proposal(checks: list, damaged: set, min_margin: float = MIN_MARGIN):
    """check.json rows -> (relabel {clip id: line id}, drop [clip ids]).
    Each line goes to at most one clip (the strongest match); weak or damaged-text matches drop."""
    drop, claims = [], {}
    for c in checks:
        if c["margin"] < min_margin or c["best"] in damaged:
            drop.append(c["id"])
        else:
            claims.setdefault(c["best"], []).append(c)
    relabel = {}
    for line, cs in claims.items():
        cs.sort(key=lambda c: -c["margin"])
        if cs[0]["id"] != line:
            relabel[cs[0]["id"]] = line
        drop += [c["id"] for c in cs[1:]]
    return relabel, sorted(drop)


class _Lines:
    """Renders (once, cached on disk) and fingerprints script lines."""

    def __init__(self):
        self._r = None
        self._cache = {}

    def _render(self, h):
        f = RENDERS / f"{h.id}.wav"
        if not f.exists():
            if self._r is None:
                from vagdhenu_mlx import Renderer
                self._r = Renderer(str(paths.MLX_UP), nfe=16, log=lambda *a, **k: None)
            from vagdhenu_mlx import detect_meter_key
            sr, y = self._r.render_one(h.deva, detect_meter_key(h.deva))
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(f, y, sr)
        return f

    def __call__(self, h):
        if h.id not in self._cache:
            y, sr = sf.read(self._render(h))
            self._cache[h.id] = features(y, sr)
        return self._cache[h.id]


def run(session: str) -> list:
    from mv.text import load_sarga
    d = paths.DATA / "sessions" / session
    sarga = int(re.match(r"sarga_(\d+)", session).group(1))
    hems = [h for v in load_sarga(sarga, include_suspect=True) for h in v.hemistichs]
    rows = json.loads((d / "clips.json").read_text(encoding="utf-8"))

    def feat_of_clip(r):
        y, sr = sf.read(r["wav"])
        return features(y, sr)

    out = suggest(rows, hems, feat_of_clip, _Lines())
    (d / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    shifted = [o for o in out if o["shifted"]]
    print(f"{session}: {len(out)} clips checked, {len(shifted)} sound like a different line")
    for o in shifted:
        print(f"  {o['id']:12} sounds like {o['best']:12} (margin {o['margin']:.3f})")
    worst = sorted((o for o in out if not o["shifted"]), key=lambda o: -o["label_cost"])[:5]
    print("  poorest matches to their own line (text typo? slip?): "
          + ", ".join(f"{o['id']} ({o['label_cost']:.2f})" for o in worst))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    run(ap.parse_args().session)
