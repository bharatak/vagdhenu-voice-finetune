"""Post-render polish: tame the vocoder's residual 'grrr' (faint fuzz between the upper harmonics).

    polish(y, sr, "filter")   gentle: soft low-pass above 8 kHz + spectral gate on the fuzz
    polish(y, sr, "opus")     what WhatsApp does: encode with Opus at 32 kbps and decode back
    polish(y, sr, "none")     unchanged

Why it works: the fuzz is quiet energy sitting BETWEEN the loud harmonic lines. A lossy codec like
Opus spends its few bits on the loud lines and drops the rest — which is why a WhatsApp-forwarded
render sounds clean. The filter does the same on purpose: per frequency it estimates the quiet
'floor' and turns down anything near it (by at most 12 dB, smoothed over time so it doesn't warble).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CUTOFF_HZ = 8000
OPUS_KBPS = 32
MAX_CUT_DB = 12.0      # never turn the floor down by more than this (deeper sounds 'underwater')
GATE = 2.0             # bins below GATE x the per-frequency floor count as fuzz


def _filter(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import butter, istft, sosfiltfilt, stft
    y = sosfiltfilt(butter(4, CUTOFF_HZ, "low", fs=sr, output="sos"), y)          # zero-phase soft cut
    _, _, X = stft(y, sr, nperseg=1024, noverlap=768)
    mag = np.abs(X)
    # the fuzz level in each moment = the valleys between harmonic lines: the minimum over a ~300 Hz
    # window (covers two harmonics of a chanting voice), smoothed. NOT a per-frequency time floor —
    # that would treat a long held vowel as 'noise' and mute it.
    win = max(3, int(round(300 / (sr / 1024))))
    floor = uniform_filter1d(minimum_filter1d(mag, size=win, axis=0), size=win, axis=0) + 1e-9
    gain = np.clip((mag - GATE * floor) / (mag + 1e-9), 10 ** (-MAX_CUT_DB / 20), 1.0)
    gain = uniform_filter1d(gain, size=5, axis=1)                                 # smooth over ~50 ms
    _, out = istft(X * gain, sr, nperseg=1024, noverlap=768)
    return out[:len(y)].astype(np.float32)


def _opus(y: np.ndarray, sr: int) -> np.ndarray:
    import soundfile as sf
    if not shutil.which("ffmpeg"):
        raise RuntimeError("--polish opus needs ffmpeg (brew install ffmpeg)")
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d, "in.wav"), Path(d, "x.ogg"), Path(d, "out.wav")
        sf.write(src, y, sr)
        q = ["ffmpeg", "-loglevel", "error", "-y"]
        subprocess.run(q + ["-i", src, "-c:a", "libopus", "-b:a", f"{OPUS_KBPS}k", enc], check=True)
        subprocess.run(q + ["-i", enc, "-ar", str(sr), "-ac", "1", dec], check=True)
        out, _ = sf.read(dec, dtype="float32")
    return np.pad(out, (0, max(0, len(y) - len(out))))[:len(y)]


def polish(y: np.ndarray, sr: int, mode: str = "filter") -> np.ndarray:
    if mode == "none":
        return y
    if mode == "filter":
        return _filter(y, sr)
    if mode == "opus":
        return _opus(y, sr)
    raise ValueError(f"unknown polish mode {mode!r} (none | filter | opus)")
