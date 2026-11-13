import numpy as np
import soundfile as sf

from mv.bank import bank_entry, pick_reference

SR = 24000
ANUSHTUBH = ("सन्ततं चिन्तयेत् कण्ठं भास्वत्कौस्तुभभासकम् । "
             "वैकुण्ठस्याखिला वेदा उद्गीर्यन्तेऽनिशं यतः ॥")


def _wav(path, seconds, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    y = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)   # chant with gaps
    sf.write(path, (y + noise * rng.standard_normal(len(t))).astype(np.float32), SR)
    return str(path)


def test_bank_entry_sec_per_syll(tmp_path):
    clip = {"id": "5.1.1.1", "deva": "ततो रावणनीतायाः सीतायाः शत्रुकर्शनः", "wav": _wav(tmp_path / "a.wav", 4.8)}
    e = bank_entry(clip, 16)
    assert e["sec_per_syll"] == 0.33
    assert e["ref_text"] == clip["deva"]
    assert e["mode"] == "half" and e["sr"] == 24000 and e["dur_s"] == 4.8


def test_pick_reference_prefers_clean_clip_of_the_right_length_and_meter(tmp_path):
    clips = [
        {"id": "a", "deva": "x", "full_verse": ANUSHTUBH, "wav": _wav(tmp_path / "a.wav", 5, noise=0.05)},
        {"id": "b", "deva": "x", "full_verse": ANUSHTUBH, "wav": _wav(tmp_path / "b.wav", 5, noise=0.001)},
        {"id": "c", "deva": "x", "full_verse": ANUSHTUBH, "wav": _wav(tmp_path / "c.wav", 3, noise=0.0)},
        {"id": "d", "deva": "x", "full_verse": "न", "wav": _wav(tmp_path / "d.wav", 5, noise=0.0)},
    ]
    assert pick_reference(clips, "anuṣṭubh")["id"] == "b"


def test_confident_needs_the_content_check_to_agree_clearly():
    from mv.bank import confident
    checks = {"5.1.2.1": {"best": "5.1.2.1", "margin": 0.12},
              "5.1.3.1": {"best": "5.1.3.1", "margin": 0.02},
              "5.1.4.1": {"best": "5.1.5.1", "margin": 0.20}}
    assert confident({"id": "5.1.2.1"}, checks)
    assert not confident({"id": "5.1.3.1"}, checks)                                  # near tie
    assert confident({"id": "5.1.5.1", "relabelled_from": "5.1.4.1"}, checks)        # relabel agrees
    assert not confident({"id": "5.1.9.1"}, checks)                                  # never checked
