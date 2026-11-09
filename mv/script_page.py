"""The on-screen recording script: one sarga per page, one hemistich per line, in order.

    python -m mv.script_page 5            # -> data/script/sarga_05.html, then open it
    python -m mv.script_page 5 --start 40 # resume at verse 40 (name that recording sarga_05_v40.m4a)

Because you chant exactly these lines in exactly this order, every clip's text is known in advance.
Greyed "skip" verses have damaged source text — don't chant them.
"""
import argparse
import html
import subprocess

from mv import paths
from mv.text import load_sarga

CSS = """
:root{--bg:#fbf8f1;--fg:#1d1a14;--muted:#8a8172;--skip:#c9c1b2;--accent:#9a4b16;--rule:#e7dfcf}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15130f;--fg:#efe8da;--muted:#8f8676;--skip:#4a453c;--accent:#e39a5c;--rule:#2a261f}}
:root[data-theme="dark"]{--bg:#15130f;--fg:#efe8da;--muted:#8f8676;--skip:#4a453c;--accent:#e39a5c;--rule:#2a261f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Noto Sans Devanagari",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--rule);padding:12px 16px;
  display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline}
header h1{font-size:20px;margin:0;color:var(--accent)}
header p{margin:0;font-size:14px;color:var(--muted)}
main{max-width:1100px;margin:0 auto;padding:24px 16px 40vh}
.verse{display:grid;grid-template-columns:3.2em 1fr;gap:0 12px;padding:10px 0;border-bottom:1px solid var(--rule)}
.num{font-size:16px;color:var(--muted);padding-top:12px;text-align:right;font-variant-numeric:tabular-nums}
.line{font-size:clamp(26px,3.6vw,40px);line-height:1.55;margin:0}
.verse.skip .line{color:var(--skip);font-size:20px}
.verse.skip .num::after{content:" ⚠"}
.skipnote{font-size:14px;color:var(--muted)}
"""


def render_html(sarga: int, start: int = 1) -> str:
    verses = load_sarga(sarga, start=start, include_suspect=True)
    rows = []
    for v in verses:
        cls = "verse skip" if v.suspect else "verse"
        lines = []
        for h in v.hemistichs:
            danda = "॥" if h.ends_verse else "।"
            lines.append(f'<p class="line">{html.escape(h.deva)} {danda}</p>')
        if v.suspect:
            lines.append('<p class="skipnote">skip — source text damaged</p>')
        rows.append(f'<section class="{cls}" data-verse="{v.shloka}"><div class="num">{v.shloka}</div>'
                    f'<div>{"".join(lines)}</div></section>')
    title = f"Sarga {sarga}" + (f" · from verse {start}" if start > 1 else "")
    return f"""<!doctype html><html lang="sa"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sundarakāṇḍa {title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<header><h1>{title}</h1>
<p>3 s silence to start · say nothing but the verses · stumble? stay silent 3 s, then restart that line</p></header>
<main>{"".join(rows)}</main></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sarga", type=int)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    a = ap.parse_args()
    out = paths.DATA / "script" / (f"sarga_{a.sarga:02d}" + (f"_v{a.start}" if a.start > 1 else "") + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(a.sarga, a.start), encoding="utf-8")
    print(out)
    if a.open:
        subprocess.run(["open", str(out)])
