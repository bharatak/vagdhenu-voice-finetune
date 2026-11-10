import json

import numpy as np

from mv.check import dtw_cost, features, proposal, suggest

SR = 16000


def _tones(freqs, dur=0.3):
    t = np.arange(int(SR * dur)) / SR
    return np.concatenate([np.sin(2 * np.pi * f * t) * 0.5 for f in freqs]).astype(np.float32)


def test_same_sequence_slower_is_closer_than_a_different_sequence():
    a = features(_tones([300, 600, 900, 450]), SR)
    slow = features(_tones([300, 600, 900, 450], dur=0.4), SR)
    other = features(_tones([900, 450, 300, 600]), SR)
    assert dtw_cost(a, slow) < dtw_cost(a, other)


class _Hem:
    def __init__(self, id):
        self.id = id


def _setup(truth_of_clip):
    """Each line's 'features' is a distinct random matrix; a clip's features are its true line's."""
    rng = np.random.default_rng(0)
    hems = [_Hem(f"5.1.{v}.{k}") for v in range(1, 6) for k in (1, 2)]
    line_feat = {h.id: rng.normal(size=(8, 30)) for h in hems}
    rows = [{"id": lab} for lab in truth_of_clip]
    return rows, hems, (lambda r: line_feat[truth_of_clip[r["id"]]]), (lambda h: line_feat[h.id])


def test_suggest_keeps_correct_labels():
    rows, hems, fc, fl = _setup({"5.1.2.1": "5.1.2.1", "5.1.2.2": "5.1.2.2"})
    out = suggest(rows, hems, fc, fl)
    assert [(o["id"], o["best"], o["shifted"]) for o in out] == [("5.1.2.1", "5.1.2.1", False),
                                                                ("5.1.2.2", "5.1.2.2", False)]


def test_suggest_finds_line_one_behind_its_label():
    rows, hems, fc, fl = _setup({"5.1.3.2": "5.1.3.1"})
    o = suggest(rows, hems, fc, fl)[0]
    assert o["best"] == "5.1.3.1" and o["shifted"] and o["margin"] > 0


def test_suggest_window_at_start_of_sarga():
    rows, hems, fc, fl = _setup({"5.1.1.1": "5.1.1.2"})
    assert suggest(rows, hems, fc, fl, window=3)[0]["best"] == "5.1.1.2"


def test_review_shows_check_suggestion(tmp_path):
    from mv.review import render_html
    rows = [{"id": "5.1.1.1", "deva": "x", "n_syll": 16, "start": 0, "end": 4, "cost": 0.05,
             "wav": str(tmp_path / "a.wav")}]
    (tmp_path / "clips.json").write_text(json.dumps(rows))
    (tmp_path / "check.json").write_text(json.dumps(
        [{"id": "5.1.1.1", "best": "5.1.1.2", "shifted": True, "margin": 0.08, "label_cost": 0.8,
          "best_cost": 0.72}]))
    assert "sounds like 5.1.1.2" in render_html(tmp_path)


def _c(id, best, margin):
    return {"id": id, "best": best, "shifted": id != best, "margin": margin}


def test_proposal_relabels_confident_shifts():
    relabel, drop = proposal([_c("5.1.2.1", "5.1.1.2", 0.2), _c("5.1.2.2", "5.1.2.2", 0.3)], set())
    assert relabel == {"5.1.2.1": "5.1.1.2"} and drop == []


def test_proposal_drops_weak_margin():
    assert proposal([_c("5.1.2.1", "5.1.1.2", 0.01)], set()) == ({}, ["5.1.2.1"])


def test_proposal_drops_damaged_target():
    assert proposal([_c("5.1.2.1", "5.1.1.2", 0.2)], {"5.1.1.2"}) == ({}, ["5.1.2.1"])


def test_proposal_collision_keeps_stronger():
    relabel, drop = proposal([_c("5.1.3.1", "5.1.2.1", 0.1), _c("5.1.3.2", "5.1.2.1", 0.3)], set())
    assert relabel == {"5.1.3.2": "5.1.2.1"} and drop == ["5.1.3.1"]


def test_proposal_unshifted_clip_holding_a_claimed_line_loses_to_stronger():
    # clip 5.1.2.1 sounds like itself weakly; clip 5.1.2.2 claims 5.1.2.1 strongly
    relabel, drop = proposal([_c("5.1.2.1", "5.1.2.1", 0.06), _c("5.1.2.2", "5.1.2.1", 0.2)], set())
    assert relabel == {"5.1.2.2": "5.1.2.1"} and drop == ["5.1.2.1"]
