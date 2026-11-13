import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_crop_returns_exact_segment_and_pads_short_clips():
    from train.finetune_vocoder import crop
    rng = np.random.default_rng(0)
    long = np.arange(100_000, dtype=np.float32)
    c = crop(long, 32768, rng)
    assert len(c) == 32768 and c[1] - c[0] == 1            # a contiguous slice of the clip
    short = np.ones(1000, dtype=np.float32)
    s = crop(short, 32768, rng)
    assert len(s) == 32768 and s[:1000].sum() == 1000 and s[1000:].sum() == 0


def test_segment_is_whole_mel_frames():
    from train.finetune_vocoder import HOP, SEG
    assert SEG % HOP == 0
