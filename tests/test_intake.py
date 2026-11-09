import subprocess

import numpy as np
import soundfile as sf

from mv.intake import level_report, to_wav_24k_mono


def test_to_wav_24k_mono(tmp_path):
    sr = 48000
    t = np.arange(2 * sr) / sr
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)], axis=1)
    wav48 = tmp_path / "src.wav"
    sf.write(wav48, stereo, sr, subtype="PCM_24")
    m4a = tmp_path / "src.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "alac", str(wav48), str(m4a)], check=True)

    out = to_wav_24k_mono(m4a, tmp_path / "out.wav")

    data, out_sr = sf.read(out)
    assert out_sr == 24000
    assert data.ndim == 1
    assert abs(len(data) / out_sr - 2.0) < 0.01


def test_level_report_snr():
    sr = 24000
    rng = np.random.default_rng(0)
    noise_amp = 10 ** (-70 / 20)
    noise = rng.normal(0, noise_amp, sr)          # RMS ≈ -70 dBFS
    t = np.arange(sr) / sr
    tone = np.sqrt(2) * 10 ** (-20 / 20) * np.sin(2 * np.pi * 220 * t)  # RMS = -20 dBFS
    rep = level_report(np.concatenate([noise, tone]).astype(np.float32), sr)

    assert 45 <= rep["snr_db"] <= 55
    assert rep["clipped_samples"] == 0
    assert abs(rep["duration_s"] - 2.0) < 1e-6
