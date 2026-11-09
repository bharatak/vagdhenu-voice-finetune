"""Step 2: cut a session recording into one clip per hemistich, each paired with its text.

    python -m mv.segment inbox/sarga_01.m4a          # verses from 1
    python -m mv.segment inbox/sarga_01_v11.m4a      # verses from 11

How: (1) find stretches of sound separated by silence; (2) line them up with the hemistichs you
were shown, in order, using one fact — a line's duration is roughly (its syllables × your pace).
Dynamic programming finds the best line-up, allowing a line to be split by a breath (merge),
a stumble or a noise to be thrown away (skip segment), or a line to be missed (skip hemistich).
"""
import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from mv import paths
from mv.intake import SR, to_wav_24k_mono
from mv.text import Hemistich, load_sarga

MAX_MERGE = 3            # a line may be split into at most 3 segments
MERGE_PENALTY = 0.10     # per extra segment merged into one line
SKIP_SEG = 0.70          # throw away a segment ...
SKIP_SEG_RESTART = 0.30  # ... cheaper when followed by >= RESTART_GAP s silence (stumble protocol)
SKIP_SEG_SHORT = 0.05    # ... very cheap when it is shorter than SHORT_SEG (click, door, cough)
RESTART_GAP = 2.5
SHORT_SEG = 0.5
SKIP_HEM = 1.50          # a line was not chanted
ACCEPT = 0.25            # keep a clip only if |ln(actual/expected duration)| < this (≈ ±28 %)
PAD = 0.10
HIGHPASS_HZ = 70         # removes room rumble below the voice


@dataclass
class Seg:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Clip:
    hem: Hemistich
    start: float
    end: float
    cost: float          # |ln(actual / expected duration)|


def speech_segments(wav: np.ndarray, sr: int, top_db: float = 30, min_gap: float = 0.25) -> list:
    iv = librosa.effects.split(np.asarray(wav, dtype=np.float32), top_db=top_db,
                               frame_length=1024, hop_length=256)
    segs = []
    for a, b in iv:
        s, e = a / sr, b / sr
        if segs and s - segs[-1].end < min_gap:
            segs[-1].end = e
        else:
            segs.append(Seg(s, e))
    return segs


def split_long(segs: list, wav: np.ndarray, sr: int, factor: float = 1.6) -> list:
    """A segment ~2× a typical line is usually two lines whose daṇḍa pause was too short or
    filled with room echo. Split it at the quietest 50 ms inside its middle 30–70 %, repeatedly."""
    typical = float(np.median([s.dur for s in segs if s.dur > 1.5] or [s.dur for s in segs]))
    n = int(0.05 * sr)
    out, todo = [], list(segs)
    while todo:
        s = todo.pop(0)
        if s.dur <= factor * typical:
            out.append(s)
            continue
        a, b = int((s.start + 0.3 * s.dur) * sr), int((s.start + 0.7 * s.dur) * sr)
        x = np.asarray(wav[a:b], dtype=np.float64)
        rms = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="valid"))
        cut = s.start + 0.3 * s.dur + (int(np.argmin(rms)) + n / 2) / sr
        todo[:0] = [Seg(s.start, cut), Seg(cut, s.end)]
    return sorted(out, key=lambda z: z.start)


def _skip_cost(segs, j):
    if segs[j].dur < SHORT_SEG:
        return SKIP_SEG_SHORT
    gap_after = segs[j + 1].start - segs[j].end if j + 1 < len(segs) else math.inf
    return SKIP_SEG_RESTART if gap_after >= RESTART_GAP else SKIP_SEG


def _dp(segs, hems, pace):
    N, M = len(hems), len(segs)
    INF = math.inf
    cost = [[INF] * (M + 1) for _ in range(N + 1)]
    back = [[None] * (M + 1) for _ in range(N + 1)]
    cost[0][0] = 0.0
    for i in range(N + 1):
        for j in range(M + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if j < M:                                    # skip segment j
                nc = c + _skip_cost(segs, j)
                if nc < cost[i][j + 1]:
                    cost[i][j + 1], back[i][j + 1] = nc, ("seg", i, j)
            if i < N:                                    # skip hemistich i
                nc = c + SKIP_HEM
                if nc < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = nc, ("hem", i, j)
                expected = pace * hems[i].n_syll
                for k in range(1, MAX_MERGE + 1):        # hemistich i <- segments j..j+k-1
                    if j + k > M:
                        break
                    d = segs[j + k - 1].end - segs[j].start
                    dc = abs(math.log(d / expected))
                    nc = c + dc + MERGE_PENALTY * (k - 1)
                    if nc < cost[i + 1][j + k]:
                        cost[i + 1][j + k], back[i + 1][j + k] = nc, ("match", i, j, k, dc)
    # the recording may stop before the sarga ends: lines after the last chanted one are free
    for i in range(N):
        if cost[i][M] < cost[N][M]:
            cost[N][M], back[N][M] = cost[i][M], ("stop", i, M)
    clips, i, j = [], N, M
    while (i, j) != (0, 0):
        b = back[i][j]
        if b[0] == "match":
            _, pi, pj, k, dc = b
            clips.append(Clip(hems[pi], segs[pj].start, segs[pj + k - 1].end, dc))
        i, j = b[1], b[2]
    return clips[::-1]


def align(segs: list, hems: list) -> list:
    """Best line-up of segments to hemistichs; returns accepted clips (cost < ACCEPT), in order."""
    if not segs or not hems:
        return []
    # first guess at pace from typical segments vs typical lines — robust to a session that
    # covers only part of the sarga (sum-over-everything would not be)
    long = [s.dur for s in segs if s.dur > 1.5] or [s.dur for s in segs]
    pace = float(np.median(long)) / float(np.median([h.n_syll for h in hems]))
    for _ in range(4):                                   # refine pace from the matched lines
        clips = _dp(segs, hems, pace)
        if not clips:
            break
        new = float(np.median([(c.end - c.start) / c.hem.n_syll for c in clips]))
        if abs(new - pace) < 1e-3:
            break
        pace = new
    return [c for c in clips if c.cost < ACCEPT]


def _session_name(src: Path):
    m = re.match(r"sarga_(\d+)(?:_v(\d+))?$", src.stem)
    if not m:
        raise SystemExit(f"name the file sarga_NN.m4a or sarga_NN_vSS.m4a (got {src.name})")
    return src.stem, int(m.group(1)), int(m.group(2) or 1)


def _clip_audio(wav, sr, c: Clip):
    a, b = max(0, int((c.start - PAD) * sr)), min(len(wav), int((c.end + PAD) * sr))
    y = wav[a:b]
    return (y / (np.abs(y).max() + 1e-9) * 10 ** (-1.5 / 20)).astype(np.float32)


def run(src: Path) -> dict:
    name, sarga, start = _session_name(src)
    out = paths.DATA / "sessions" / name
    (out / "clips").mkdir(parents=True, exist_ok=True)
    wav_path = to_wav_24k_mono(src, out / "session.wav")
    wav, sr = sf.read(wav_path)
    wav = sosfiltfilt(butter(4, HIGHPASS_HZ, "highpass", fs=sr, output="sos"), wav)

    hems = [h for v in load_sarga(sarga, start=start) for h in v.hemistichs]
    segs = split_long(speech_segments(wav, sr), wav, sr)
    # the session may stop early: only consider hemistichs the audio could plausibly cover
    clips = align(segs, hems)
    last = max((hems.index(c.hem) for c in clips), default=-1)
    hems_covered = hems[: last + 1]

    rows = []
    for c in clips:
        f = out / "clips" / f"{c.hem.id}.wav"
        sf.write(f, _clip_audio(wav, sr, c), sr)
        rows.append({"id": c.hem.id, "deva": c.hem.deva, "n_syll": c.hem.n_syll,
                     "start": round(c.start, 3), "end": round(c.end, 3), "cost": round(c.cost, 3),
                     "wav": str(f)})
    (out / "clips.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    kept = {r["id"] for r in rows}
    missing = [h.id for h in hems_covered if h.id not in kept]
    pace = float(np.median([(r["end"] - r["start"]) / r["n_syll"] for r in rows])) if rows else 0
    summary = {"session": name, "segments": len(segs), "hemistichs_covered": len(hems_covered),
               "kept": len(rows), "not_kept": missing, "pace_s_per_syll": round(pace, 3),
               "minutes_kept": round(sum(r["end"] - r["start"] for r in rows) / 60, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{name}: {len(segs)} segments → {len(rows)}/{len(hems_covered)} hemistichs kept "
          f"({summary['minutes_kept']} min), pace {pace:.3f} s/syll")
    if missing:
        print("  not kept (low confidence / not found): " + ", ".join(missing))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    run(ap.parse_args().src)
