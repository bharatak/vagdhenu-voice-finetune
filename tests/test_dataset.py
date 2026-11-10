import json

import numpy as np
import pytest

pytest.importorskip("datasets")  # runs in .venv-train (the Mac inference venv has no `datasets`)
import soundfile as sf

from mv.dataset import rebase, rows, write_f5_dataset

SR = 24000


def _session(tmp_path, clips):
    """clips: [(id, deva, seconds)] -> a session dir like mv.segment writes."""
    d = tmp_path / "sessions" / "s"
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
    assert ds[0]["audio_path"] == str(out / "wavs" / "5.1.30.2.wav") and (out / "wavs" / "5.1.30.2.wav").exists()
    assert json.loads((out / "duration.json").read_text())["duration"] == pytest.approx([4.0], abs=0.01)
    assert json.loads((out / "stats.json").read_text())["n"] == 1
    rebase(out, "/workspace/dataset")
    assert Dataset.from_file(str(out / "raw.arrow"))[0]["audio_path"] == "/workspace/dataset/wavs/5.1.30.2.wav"
