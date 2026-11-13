"""Chant any text with a model + a reference bank (upstream's MLX Renderer, unchanged).

    python -m mv.render --text "<verse>" --out data/renders/x.wav                  # Prathosh's model, YOUR bank
    python -m mv.render --text "<verse>" --bank upstream --out data/renders/y.wav  # Prathosh's model, his bank
    python -m mv.render --text "<verse>" --model data/mlx_ft --out z.wav           # your fine-tuned model
    python -m mv.render --text "<verse>" --model models/mlx-myvoice --out p.wav   # filter polish is the default; --polish none for raw

Swapping only the bank shows what in-context cloning alone does; swapping the model shows
what fine-tuning adds.
"""
import argparse
from pathlib import Path

import soundfile as sf

from mv import paths
from mv.polish import polish

BANK = paths.DATA / "bank" / "bank.json"
UP_BANK = paths.UPSTREAM_SRC / "reference_bank" / "bank.json"


def render(text: str, out: Path, model=paths.MLX_UP, bank=BANK, seed: int = 60, nfe: int = 64,
           meter_hint: str = "", polish_mode: str = "filter") -> Path:
    from chant_text import detect_meter_key
    from vagdhenu_mlx import Renderer
    r = Renderer(str(model), bank_path=str(bank), nfe=nfe, log=lambda *a, **k: None)
    found = meter_hint.lower() or detect_meter_key(text)   # the detector needs pādas it can split
    stems = {r._lut[m.lower()]["wav"].removesuffix(".wav"): m for m in r.meters()}
    meter = stems.get(found) or next((m for m in r.meters() if m.lower() == found), None) or r.meters()[0]
    print(f"meter: {found or '?'} -> reference '{meter}'" + ("" if meter.lower() == found or found in stems else " (fallback)"))
    sr, y = r.render_one(text, meter, seed=seed)
    y = polish(y, sr, polish_mode)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr)
    print(f"{out}  ({len(y) / sr:.1f} s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--model", default=str(paths.MLX_UP))
    ap.add_argument("--bank", default=str(BANK), help="path to bank.json, or 'upstream' for Prathosh's")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--nfe", type=int, default=64, help="denoising steps (upstream default; 32 is ~2x faster)")
    ap.add_argument("--meter", default="", help="force the meter, e.g. drutavilambita (bank key or wav stem)")
    ap.add_argument("--polish", default="filter", choices=["none", "filter", "opus"],
                    help="tame the vocoder's residual grrr (see mv/polish.py)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    render(a.text, a.out, a.model, UP_BANK if a.bank == "upstream" else a.bank, a.seed, a.nfe, a.meter, a.polish)
