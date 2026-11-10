"""Step 4: gather every kept clip into one training set, in the format IndicF5's trainer reads.

    python -m mv.dataset                         # all sessions -> data/dataset/
    python -m mv.dataset --rebase /workspace/dataset   # on the GPU box, after copying the folder

One training example = (audio, text). The trainer turns the audio into a 100-band log-mel at
24 kHz — the same picture the model draws when it chants — and reads the text one character
at a time against vocab.txt. So the text must be in exactly the form the renderer feeds the
model at inference time: Devanagari -> SLP1 -> Kannada letters, plus the renderer's small fixes
(anusvāra, visarga, daṇḍa, hna, vocalic ḷ). Any character outside the released vocab would be
a symbol the model has never seen.

Output folder:
  raw.arrow      audio_path | text | duration      (what CustomDataset reads)
  duration.json  {"duration": [...]}                (the trainer batches by length with this)
  metadata.csv   audio_path|text                    (human-readable copy)
  stats.json     clip count, minutes, min/median/max seconds
  wavs/          copies of the clips, so the folder can be moved to the GPU as one piece
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from mv import paths
from mv.review import kept_clips

MIN_S, MAX_S = 2.5, 15.0
SAMPLES_JSON = paths.ROOT / "train" / "samples.json"
OUT = paths.DATA / "dataset"


def train_text(deva: str) -> str:
    """The renderer's own text chain (vagdhenu_mlx.Renderer.render_one, no_sandhi=True)."""
    import prep_text as PT
    from chant_text import _anusvara_m, _danda_fix, _hna_metathesis, _vocalic_l, _visarga_ksha
    return _vocalic_l(_hna_metathesis(_danda_fix(_visarga_ksha(_anusvara_m(PT.model_text(deva))))))


def _heldout_default() -> set:
    if not SAMPLES_JSON.exists():
        return set()
    return {s["id"] for s in json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))}


def rows(session_dirs: list, heldout: set = None) -> list:
    heldout = _heldout_default() if heldout is None else heldout
    out = []
    for d in session_dirs:
        for c in kept_clips(Path(d)):
            if ".".join(c["id"].split(".")[:3]) in heldout:
                continue
            dur = sf.info(c["wav"]).duration
            if not MIN_S <= dur <= MAX_S:
                continue
            out.append({"id": c["id"], "audio_path": str(Path(c["wav"]).resolve()),
                        "text": train_text(c["deva"]), "duration": round(float(dur), 3)})
    return out


def _write_arrow(records: list, path: Path) -> None:
    from datasets.arrow_writer import ArrowWriter
    tmp = path.with_suffix(".tmp")
    with ArrowWriter(path=str(tmp)) as w:
        for r in records:
            w.write(r)
        w.finalize()
    tmp.replace(path)


def write_f5_dataset(rows_: list, out: Path) -> None:
    out = Path(out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    records = []
    for r in rows_:
        dst = out / "wavs" / f"{r['id']}.wav"
        shutil.copyfile(r["audio_path"], dst)
        records.append({"audio_path": str(dst), "text": r["text"], "duration": r["duration"]})
    _write_arrow(records, out / "raw.arrow")
    durs = [r["duration"] for r in records]
    (out / "duration.json").write_text(json.dumps({"duration": durs}))
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_path", "text"])
        w.writerows([r["audio_path"], r["text"]] for r in records)
    stats = {"n": len(durs), "minutes": round(sum(durs) / 60, 2),
             "min_s": round(min(durs), 2) if durs else 0, "median_s": round(float(np.median(durs)), 2) if durs else 0,
             "max_s": round(max(durs), 2) if durs else 0}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


def rebase(out: Path, new_root: str) -> None:
    """Point raw.arrow at the wavs/ folder under new_root (after moving the dataset folder)."""
    from datasets import Dataset
    out = Path(out)
    ds = Dataset.from_file(str(out / "raw.arrow"))
    records = [{"audio_path": f"{new_root.rstrip('/')}/wavs/{Path(r['audio_path']).name}",
                "text": r["text"], "duration": r["duration"]} for r in ds]
    del ds
    _write_arrow(records, out / "raw.arrow")


def _histogram(durs: list) -> str:
    edges = np.arange(MIN_S, MAX_S + 0.5, 0.5)
    h, _ = np.histogram(durs, bins=edges)
    return "\n".join(f"  {a:4.1f}–{a + 0.5:4.1f}s {'█' * int(n)} {n}" for a, n in zip(edges, h) if n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebase", metavar="NEW_ROOT")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.rebase:
        rebase(a.out, a.rebase)
        print(f"raw.arrow now points at {a.rebase}/wavs/")
    else:
        sessions = sorted(p.parent for p in (paths.DATA / "sessions").glob("*/clips.json"))
        all_kept = [c for d in sessions for c in kept_clips(d)]
        short = sum(1 for c in all_kept if sf.info(c["wav"]).duration < MIN_S)
        r = rows(sessions)
        write_f5_dataset(r, a.out)
        s = json.loads((a.out / "stats.json").read_text())
        print(f"{len(sessions)} sessions -> {s['n']} clips, {s['minutes']} min "
              f"(min {s['min_s']} / median {s['median_s']} / max {s['max_s']} s)")
        print(f"clips under {MIN_S}s: {short}/{len(all_kept)} ({100 * short / max(1, len(all_kept)):.0f} %)")
        print(_histogram([x["duration"] for x in r]))
        print(a.out)
