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
