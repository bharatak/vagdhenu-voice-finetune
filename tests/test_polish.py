import shutil

import numpy as np
import pytest

from mv.polish import polish

SR = 24000


def _band_energy(y, lo, hi):
    f = np.fft.rfftfreq(len(y), 1 / SR)
    P = np.abs(np.fft.rfft(y)) ** 2
    return P[(f >= lo) & (f < hi)].sum()


def _voice_plus_fuzz(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(SR * 2) / SR
    voice = sum(0.2 / k * np.sin(2 * np.pi * 200 * k * t) for k in range(1, 10))   # harmonics to 1.8 kHz
    return (voice + 0.01 * rng.standard_normal(len(t))).astype(np.float32), voice


def test_filter_keeps_the_voice_and_removes_fuzz():
    y, voice = _voice_plus_fuzz()
    p = polish(y, SR, "filter")
    assert len(p) == len(y)
    assert _band_energy(p, 150, 2000) > 0.9 * _band_energy(voice, 150, 2000)        # voice kept
    assert _band_energy(p, 9000, 12000) < 0.1 * _band_energy(y, 9000, 12000)        # hiss above 8 kHz cut
    assert _band_energy(p, 2500, 7000) < 0.5 * _band_energy(y, 2500, 7000)          # fuzz between harmonics down


def test_none_is_identity():
    y, _ = _voice_plus_fuzz()
    assert polish(y, SR, "none") is y


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_opus_round_trip_keeps_length_and_voice():
    y, voice = _voice_plus_fuzz()
    p = polish(y, SR, "opus")
    assert len(p) == len(y)
    assert _band_energy(p, 150, 2000) > 0.8 * _band_energy(voice, 150, 2000)


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        polish(np.zeros(10, np.float32), SR, "shiny")


def test_filter_leaves_sibilants_and_visarga_alone():
    """ś/ṣ/s/h are noise by nature; the fuzz gate must not mistake them for vocoder fuzz."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(SR).astype(np.float32)
    for lo, hi in [(3000, 6000), (4000, 7900)]:                       # ś/ṣ-like, s-like (below the 8 kHz cut)
        burst = sosfiltfilt(butter(6, [lo, hi], "band", fs=SR, output="sos"), noise).astype(np.float32)
        loss_db = 10 * np.log10(_band_energy(burst, lo, hi) / _band_energy(polish(burst, SR, "filter"), lo, hi))
        assert loss_db < 2.0, f"{lo}-{hi} Hz noise lost {loss_db:.1f} dB"
