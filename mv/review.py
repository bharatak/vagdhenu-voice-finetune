"""Step 3: listen to the clips with their text, and throw out bad ones.

    python -m mv.review sarga_01 --open

Any clip that is wrong (wrong words, a stumble, a noise) -> put its id on its own line in
data/sessions/<session>/drop.txt. Dropped clips never reach training.
A clip that is a *different* line than its label -> "<clip id> <true line id>" on its own line in
relabel.txt; it keeps its audio and takes that line's text. Both files use the clip ids as cut.
"""
import argparse
import html
import json
import subprocess
from pathlib import Path

from mv import paths

CSS = """
:root{--bg:#fbf8f1;--fg:#1d1a14;--muted:#8a8172;--rule:#e7dfcf;--warn:#b0451a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15130f;--fg:#efe8da;--muted:#8f8676;--rule:#2a261f;--warn:#e88a5c}}
:root[data-theme="dark"]{--bg:#15130f;--fg:#efe8da;--muted:#8f8676;--rule:#2a261f;--warn:#e88a5c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:"Noto Sans Devanagari",system-ui,sans-serif}
header{padding:16px;border-bottom:1px solid var(--rule)}h1{margin:0 0 4px;font-size:20px}
header p{margin:0;color:var(--muted);font-size:14px}main{max-width:1000px;margin:0 auto;padding:8px 16px 64px}
.row{display:grid;grid-template-columns:7.5em 1fr;gap:4px 12px;padding:12px 0;border-bottom:1px solid var(--rule);align-items:center}
.id{font-family:ui-monospace,monospace;font-size:13px;color:var(--muted)}.id b{color:var(--fg);font-weight:600}
.deva{font-size:24px;margin:0 0 6px}audio{width:100%;max-width:520px;height:36px}
.check{font-size:13px;color:var(--warn);margin:4px 0 0}.check.ok{color:var(--muted)}
.dropped{opacity:.4}.dropped .id::after{content:" dropped";color:var(--warn)}
button{font:inherit;font-size:14px;padding:6px 12px;border-radius:6px;border:1px solid var(--rule);background:transparent;color:var(--fg);cursor:pointer;margin-top:8px}
"""

JS = """
const main=document.querySelector('main');
document.getElementById('shuffle').onclick=()=>{const r=[...main.children];
 for(let i=r.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[r[i],r[j]]=[r[j],r[i]]}
 r.forEach(x=>main.appendChild(x))};
"""


def _rows(session_dir: Path) -> list:
    return json.loads((Path(session_dir) / "clips.json").read_text(encoding="utf-8"))


def _dropped(session_dir: Path) -> set:
    f = Path(session_dir) / "drop.txt"
    return {l.strip() for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()


def _relabels(session_dir: Path) -> dict:
    f = Path(session_dir) / "relabel.txt"
    if not f.exists():
        return {}
    return dict(l.split() for l in f.read_text().splitlines() if l.strip())


def _hemistich(hid: str):
    from mv.text import load_sarga
    sarga = int(hid.split(".")[1])
    hems = {h.id: h for v in load_sarga(sarga, include_suspect=True) for h in v.hemistichs}
    if hid not in hems:
        raise ValueError(f"relabel.txt: no such line {hid}")
    return hems[hid]


def _checks(session_dir: Path) -> dict:
    f = Path(session_dir) / "check.json"
    return {c["id"]: c for c in json.loads(f.read_text(encoding="utf-8"))} if f.exists() else {}


def _check_note(c, relabelled_to):
    if not c or not c["shifted"]:
        return ""
    if relabelled_to == c["best"]:
        return '<p class="check ok">✓ content check agrees</p>'
    return f'<p class="check">⚠ sounds like {c["best"]} (margin {c["margin"]:.2f})</p>'


def kept_clips(session_dir: Path) -> list:
    drop, relabel = _dropped(session_dir), _relabels(session_dir)
    out = []
    for r in _rows(session_dir):
        if r["id"] in drop:
            continue
        if r["id"] in relabel:
            h = _hemistich(relabel[r["id"]])
            r = {**r, "id": h.id, "deva": h.deva, "n_syll": h.n_syll, "relabelled_from": r["id"]}
        out.append(r)
    ids = [r["id"] for r in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"two kept clips share a label after relabel: {', '.join(dup)}")
    return out


def render_html(session_dir: Path) -> str:
    session_dir = Path(session_dir)
    drop, relabel, checks = _dropped(session_dir), _relabels(session_dir), _checks(session_dir)
    rows = _rows(session_dir)
    items = []
    for r in rows:
        note = "" if r["id"] in drop else _check_note(checks.get(r["id"]), relabel.get(r["id"]))
        if r["id"] in relabel and r["id"] not in drop:
            h = _hemistich(relabel[r["id"]])
            r = {**r, "id": f'{r["id"]} → {h.id}', "deva": h.deva}
        rel = Path(r["wav"]).relative_to(session_dir) if Path(r["wav"]).is_relative_to(session_dir) else r["wav"]
        cls = "row dropped" if r["id"] in drop else "row"
        items.append(f'<div class="{cls}"><div class="id"><b>{r["id"]}</b><br>{r["end"]-r["start"]:.1f}s · '
                     f'fit {r["cost"]:.2f}</div><div><p class="deva">{html.escape(r["deva"])}</p>'
                     f'<audio controls preload="none" src="{html.escape(str(rel))}"></audio>{note}</div></div>')
    summary = {}
    if (session_dir / "summary.json").exists():
        summary = json.loads((session_dir / "summary.json").read_text())
    missing = summary.get("not_kept", [])
    note = (f"Not kept: {', '.join(missing)}" if missing else "Every line was matched.")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Clip review {session_dir.name}</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header><h1>{session_dir.name} — {len(rows)} clips</h1>
<p>Play a clip, read along. Wrong? Add its id to drop.txt in this folder. {html.escape(note)}</p>
<button id="shuffle">Shuffle (spot-check a random 20)</button></header>
<main>{"".join(items)}</main><script>{JS}</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    d = paths.DATA / "sessions" / a.session
    out = d / "review.html"
    out.write_text(render_html(d), encoding="utf-8")
    print(out)
    if a.open:
        subprocess.run(["open", str(out)])
