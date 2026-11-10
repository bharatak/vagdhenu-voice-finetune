import json

from mv.review import kept_clips, render_html


def _session(tmp_path, drop=None):
    rows = [{"id": f"5.1.1.{k}", "deva": "x", "n_syll": 16, "start": 0, "end": 4.8, "cost": 0.05,
             "wav": str(tmp_path / f"5.1.1.{k}.wav")} for k in (1, 2, 3)]
    (tmp_path / "clips.json").write_text(json.dumps(rows))
    if drop is not None:
        (tmp_path / "drop.txt").write_text(drop)
    return tmp_path


def test_kept_clips_respects_drop(tmp_path):
    assert [r["id"] for r in kept_clips(_session(tmp_path, "5.1.1.2\n"))] == ["5.1.1.1", "5.1.1.3"]


def test_kept_clips_without_drop_file(tmp_path):
    assert len(kept_clips(_session(tmp_path))) == 3


def test_review_html_lists_clips(tmp_path):
    html = render_html(_session(tmp_path))
    assert html.count("<audio") == 3 and "5.1.1.2" in html


def _relabel(tmp_path, text):
    (tmp_path / "relabel.txt").write_text(text)
    return tmp_path


def test_relabel_takes_text_of_true_line(tmp_path):
    from mv.text import load_sarga
    true = {h.id: h for v in load_sarga(1) for h in v.hemistichs}["5.1.2.1"]
    d = _relabel(_session(tmp_path), "5.1.1.1 5.1.2.1\n")
    r = kept_clips(d)[0]
    assert (r["id"], r["deva"], r["n_syll"]) == ("5.1.2.1", true.deva, true.n_syll)
    assert r["wav"].endswith("5.1.1.1.wav")          # same audio, new label


def test_drop_uses_original_ids_before_relabel(tmp_path):
    d = _relabel(_session(tmp_path, "5.1.1.2\n"), "5.1.1.1 5.1.1.2\n")
    assert [r["id"] for r in kept_clips(d)] == ["5.1.1.2", "5.1.1.3"]


def test_relabel_collision_is_an_error(tmp_path):
    import pytest
    d = _relabel(_session(tmp_path), "5.1.1.1 5.1.1.2\n")
    with pytest.raises(ValueError, match="5.1.1.2"):
        kept_clips(d)


def test_relabel_to_unknown_line_is_an_error(tmp_path):
    import pytest
    d = _relabel(_session(tmp_path), "5.1.1.1 5.1.1.9\n")
    with pytest.raises(ValueError, match="no such line 5.1.1.9"):
        kept_clips(d)


def test_review_shows_relabel(tmp_path):
    html = render_html(_relabel(_session(tmp_path), "5.1.1.1 5.1.2.1\n"))
    assert "5.1.1.1 → 5.1.2.1" in html
