import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets")  # runs in .venv-train (the Mac inference venv has no `datasets`)
import soundfile as sf

from mv.dataset import rebase, rows, write_f5_dataset

SR = 24000


def _session(tmp_path, clips, name="s"):
    """clips: [(id, deva, seconds)] -> a session dir like mv.segment writes."""
    d = tmp_path / "sessions" / name
    (d / "clips").mkdir(parents=True)
    out = []
    for cid, deva, sec in clips:
        f = d / "clips" / f"{cid}.wav"
        sf.write(f, (0.1 * np.sin(np.arange(int(SR * sec)) / 10)).astype(np.float32), SR)
        out.append({"id": cid, "deva": deva, "n_syll": 16, "start": 0, "end": sec, "cost": 0.05, "wav": str(f)})
    (d / "clips.json").write_text(json.dumps(out, ensure_ascii=False))
    return d


LINE = "समुद्रस्य परं पारं दुष्प्रापं प्राप्तुमिच्छति"


def test_rows_text_is_kannada(tmp_path):
    r = rows([_session(tmp_path, [("5.1.30.2", LINE, 4.0)])], heldout=set())
    assert r and all(0xC80 <= ord(c) <= 0xCFF or c == " " for c in r[0]["text"])


def test_rows_duration_filter(tmp_path):
    d = _session(tmp_path, [("5.1.30.1", LINE, 2.0), ("5.1.30.2", LINE, 4.0), ("5.1.31.1", LINE, 16.0)])
    assert [round(x["duration"], 1) for x in rows([d], heldout=set())] == [4.0]


def test_rows_excludes_heldout(tmp_path):
    d = _session(tmp_path, [("5.68.1.1", LINE, 4.0), ("5.68.2.1", LINE, 4.0)])
    assert len(rows([d], heldout={"5.68.1"})) == 1


def test_write_and_rebase(tmp_path):
    from datasets import Dataset
    r = rows([_session(tmp_path, [("5.1.30.2", LINE, 4.0)])], heldout=set())
    out = tmp_path / "dataset"
    write_f5_dataset(r, out)
    ds = Dataset.from_file(str(out / "raw.arrow"))
    assert ds.column_names == ["audio_path", "text", "duration"]
    assert ds[0]["audio_path"] == str(out / "wavs" / "s__5.1.30.2.wav") and (out / "wavs" / "s__5.1.30.2.wav").exists()
    assert json.loads((out / "duration.json").read_text())["duration"] == pytest.approx([4.0], abs=0.01)
    assert json.loads((out / "stats.json").read_text())["n"] == 1
    rebase(out, "/workspace/dataset")
    assert Dataset.from_file(str(out / "raw.arrow"))[0]["audio_path"] == "/workspace/dataset/wavs/s__5.1.30.2.wav"


def test_two_takes_of_one_line_in_two_sessions_are_both_kept(tmp_path):
    from datasets import Dataset
    a = _session(tmp_path, [("5.1.30.2", LINE, 4.0)], name="sarga_01")
    b = _session(tmp_path, [("5.1.30.2", LINE, 4.5)], name="sarga_01_v30")
    out = tmp_path / "dataset"
    write_f5_dataset(rows([a, b], heldout=set()), out)
    paths = [r["audio_path"] for r in Dataset.from_file(str(out / "raw.arrow"))]
    assert len(set(paths)) == 2 and all(Path(p).exists() for p in paths)
    assert sorted(round(sf.info(p).duration, 1) for p in paths) == [4.0, 4.5]


def test_write_removes_wavs_left_from_an_earlier_build(tmp_path):
    out = tmp_path / "dataset"
    (out / "wavs").mkdir(parents=True)
    (out / "wavs" / "dropped_cough.wav").write_bytes(b"x")
    write_f5_dataset(rows([_session(tmp_path, [("5.1.30.2", LINE, 4.0)])], heldout=set()), out)
    assert sorted(f.name for f in (out / "wavs").iterdir()) == ["s__5.1.30.2.wav"]
