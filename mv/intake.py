"""Step 1 of the pipeline: get a recording off the iPhone into the format everything else uses,
and measure whether it is clean enough to train on.

    python -m mv.intake check inbox/test.m4a

Writes data/test/orig.wav (what you recorded) and data/test/roundtrip.wav (your recording turned
into a mel spectrogram and back into sound by the released BigVGAN vocoder). If the two sound the
same, the vocoder can reproduce your voice and we can reuse it unchanged.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from mv import paths

SR = 24000
FRAME_S = 0.05
TARGETS = {  # what a training-quality recording looks like
    "noise_floor_dbfs": ("<=", -60.0),
    "snr_db": (">=", 35.0),
    "peak_dbfs": ("<=", -1.0),
    "clipped_samples": ("<=", 0),
}


def to_wav_24k_mono(src: Path, dst: Path) -> Path:
    """Any Core Audio–readable file (iPhone .m4a/ALAC, wav, …) -> 24 kHz mono 16-bit WAV.
    Uses macOS's built-in afconvert, so no ffmpeg dependency."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(src), str(dst)],
                   check=True)
    return dst


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def level_report(wav: np.ndarray, sr: int) -> dict:
    """Loudness numbers that matter for training data.
    noise floor = 10th-percentile RMS of 50 ms frames (the quiet bits between phrases);
    speech level = 90th percentile (the chanting itself); SNR = their difference."""
    wav = np.asarray(wav, dtype=np.float64)
    n = int(sr * FRAME_S)
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    noise, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    return {
        "duration_s": len(wav) / sr,
        "peak_dbfs": _db(np.abs(wav).max()),
        "noise_floor_dbfs": _db(noise),
        "speech_rms_dbfs": _db(speech),
        "snr_db": _db(speech) - _db(noise),
        "clipped_samples": int((np.abs(wav) >= 0.999).sum()),
    }


def vocoder_roundtrip(wav24k: np.ndarray) -> np.ndarray:
    """wave -> log-mel (exactly what the TTS model predicts) -> released BigVGAN -> wave."""
    import mlx.core as mx
    from vagdhenu_mlx.audio import log_mel
    from vagdhenu_mlx.model import load_bigvgan

    voc = load_bigvgan(str(paths.MLX_UP), dtype=mx.float32)
    mel = log_mel(np.asarray(wav24k, dtype=np.float32))          # (frames, 100)
    y = voc(mx.array(mel[None]))
    return np.asarray(y, dtype=np.float32)[0]


def _verdict(key: str, value: float) -> str:
    op, target = TARGETS[key]
    ok = value <= target if op == "<=" else value >= target
    return f"{'ok  ' if ok else 'WARN'} (target {op} {target})"


def check(src: Path) -> dict:
    out = paths.DATA / "test"
    wav_path = to_wav_24k_mono(src, out / "orig.wav")
    wav, sr = sf.read(wav_path)
    rep = level_report(wav, sr)
    print(f"\n{src.name}: {rep['duration_s']:.1f} s")
    for k in ["noise_floor_dbfs", "speech_rms_dbfs", "snr_db", "peak_dbfs", "clipped_samples"]:
        v = rep[k]
        line = f"  {k:18s} {v:8.1f}" if isinstance(v, float) else f"  {k:18s} {v:8d}"
        print(line + ("   " + _verdict(k, v) if k in TARGETS else ""))
    rt = vocoder_roundtrip(wav)
    sf.write(out / "roundtrip.wav", rt, SR)
    print(f"\n  listen:  afplay {out/'orig.wav'}\n           afplay {out/'roundtrip.wav'}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="convert + measure + vocoder round-trip a test clip")
    c.add_argument("src", type=Path)
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.src)
