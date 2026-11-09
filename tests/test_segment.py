import numpy as np

from mv.segment import Seg, align, speech_segments
from mv.text import Hemistich

PACE = 0.3  # s per syllable


def hems(n, syll=16):
    return [Hemistich(f"5.1.{i // 2 + 1}.{i % 2 + 1}", "x", syll, i % 2 == 1) for i in range(n)]


def layout(durs, gaps):
    """Seg list from per-segment durations and the gaps after each."""
    out, t = [], 1.0
    for d, g in zip(durs, gaps + [0]):
        out.append(Seg(t, t + d)); t += d + g
    return out


def test_align_clean():
    H = hems(4)
    clips = align(layout([4.8] * 4, [0.4] * 3), H)
    assert [c.hem.id for c in clips] == [h.id for h in H]
    assert all(c.cost < 0.1 for c in clips)


def test_align_merges_split_line():
    H = hems(3)
    segs = layout([4.8, 2.2, 2.4, 4.8], [0.4, 0.4, 0.4])   # hem 2 split in two by a breath
    clips = align(segs, H)
    assert len(clips) == 3
    assert clips[1].start == segs[1].start and clips[1].end == segs[2].end


def test_align_skips_restart():
    H = hems(3)
    segs = layout([4.8, 1.9, 4.8, 4.8], [0.4, 3.0, 0.4])   # partial hem 2, 3 s silence, full hem 2
    clips = align(segs, H)
    assert [c.hem.id for c in clips] == [h.id for h in H]
    assert clips[1].start == segs[2].start


def test_align_skips_noise_burst():
    H = hems(2)
    segs = layout([4.8, 0.2, 4.8], [0.4, 0.4])
    clips = align(segs, H)
    assert len(clips) == 2
    assert clips[1].start == segs[2].start


def test_speech_segments_tone_bursts():
    sr = 24000
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 200 * t)
    sil = np.zeros(int(0.6 * sr))
    wav = np.concatenate([sil, tone, sil, tone, sil, tone, sil]).astype(np.float32)
    segs = speech_segments(wav, sr)
    assert len(segs) == 3
    for s, expect in zip(segs, [0.6, 2.2, 3.8]):
        assert abs(s.start - expect) < 0.05 and abs(s.end - (expect + 1.0)) < 0.05


def test_align_session_stops_early():
    H = hems(40)                                   # sarga has 40 lines, only 6 were chanted
    segs = layout([4.8, 4.6, 5.0, 4.9, 4.7, 4.8], [0.4] * 5)
    clips = align(segs, H)
    assert [c.hem.id for c in clips] == [h.id for h in H[:6]]


def test_split_long_segment_at_dip():
    from mv.segment import split_long
    sr = 24000
    t = np.arange(int(4.5 * sr)) / sr
    line = 0.3 * np.sin(2 * np.pi * 200 * t)
    dip = 0.3 * 10 ** (-20 / 20) * np.sin(2 * np.pi * 200 * np.arange(int(0.15 * sr)) / sr)  # echo-filled daṇḍa
    sil = np.zeros(int(0.5 * sr))
    # a realistic session: mostly single lines, one pair run together through an echo-filled daṇḍa
    wav = np.concatenate([sil, line, sil, line, sil, line, sil, line, dip, line, sil]).astype(np.float32)
    segs = speech_segments(wav, sr)
    assert len(segs) == 4 and segs[3].dur > 9.0
    out = split_long(segs, wav, sr)
    assert len(out) == 5
    assert abs(out[3].end - (4 * 0.5 + 4 * 4.5 + 0.075)) < 0.1
