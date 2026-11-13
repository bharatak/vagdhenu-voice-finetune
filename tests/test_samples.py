import json

from mv import paths


def test_samples_json_has_every_field_upstream_render_reads():
    rows = json.loads((paths.ROOT / "train" / "samples.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["5.68.1", "5.68.2", "5.68.3"]
    for r in rows:
        assert {"id", "meter", "padas", "seed", "no_sandhi"} <= set(r)   # src/render.py: clip["no_sandhi"]
        assert r["no_sandhi"] is True and len(r["padas"]) == 2
