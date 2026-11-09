import re

from mv.text import clean_line, load_sarga


def test_clean_line_basic():
    src = "ततो रावणनीतायाः सीतायाः शत्रुकर्शनः ।इयेष पदमन्वेष्टुं चारणाचरिते पथि ॥५-१-१॥"
    assert clean_line(src) == ["ततो रावणनीतायाः सीतायाः शत्रुकर्शनः", "इयेष पदमन्वेष्टुं चारणाचरिते पथि"]


def test_clean_line_three_hemistichs():
    assert len(clean_line("क ख ग घ ।ङ च छ ज ।झ ञ ट ठ ॥५-२-९॥")) == 3


def test_clean_line_alt_numbering():
    src = "अथाहमुत्तरं देव्या पुनरुक्तः ससम्भ्रमः ।तव स्नेहान्नरव्याघ्र सौहार्दादनुमान्य वै ।। ५.६८.१ ।।।।"
    out = clean_line(src)
    assert len(out) == 2
    assert not any(re.search(r"[0-9०-९.]", h) for h in out)


def test_load_sarga_ids():
    v = load_sarga(1)[0]
    assert v.hemistichs[1].id == "5.1.1.2"
    assert [h.ends_verse for h in v.hemistichs] == [False, True]


def test_anushtubh_syllables():
    assert load_sarga(1)[0].hemistichs[0].n_syll == 16


def test_load_sarga_start():
    assert load_sarga(1, start=20)[0].shloka == 20


def test_all_sargas_parse():
    n_ok = 0
    for s in range(1, 69):
        for v in load_sarga(s):
            assert 1 <= len(v.hemistichs) <= 5, (v.sarga, v.shloka)
            for h in v.hemistichs:
                assert 8 <= h.n_syll <= 26, (h.id, h.deva)
                assert not re.search(r"[0-9०-९A-Za-z.]", h.deva), h.id
            n_ok += 1
    assert n_ok > 2700          # suspect verses are a small minority


def test_suspect_verse_excluded_by_default():
    # 5.1.92 has a Latin letter in the source ("पाताLअतल")
    assert 92 not in [v.shloka for v in load_sarga(1)]
    v = [v for v in load_sarga(1, include_suspect=True) if v.shloka == 92][0]
    assert v.suspect


def test_number_fragment_dropped():
    # source 5.38.18 ends "।। ५.३८.१८।।.३८.१७।।।।" — the stray ".३८.१७" must not become a hemistich
    assert clean_line("क ख ग घ ङ च छ ज ।झ ञ ट ठ ड ढ ण त ।। ५.३८.१८।।.३८.१७।।।।") == [
        "क ख ग घ ङ च छ ज", "झ ञ ट ठ ड ढ ण त"]
