"""Step 5: your own reference bank — one short clip of YOU per meter, for in-context cloning.

    python -m mv.bank          # -> data/bank/bank.json + data/bank/<meter>.wav

F5-TTS doesn't "know" a voice; at inference it is handed a reference clip plus that clip's exact
text, and it continues the audio as if the new text came next. The reference therefore sets the
melody, pace and (as much as the model allows) the voice. Two rules from Prathosh's bank:
  * half-reference: one hemistich, and ref_text must match the audio EXACTLY — any mismatch and
    the model "reads" the wrong syllables against the sound and drifts;
  * sec_per_syll: your pace (+10 % breathing room) fixes each line's length.
The bank is in upstream's format, so upstream's Renderer can use it unchanged.
"""
import json
import shutil
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from mv import paths
from mv.intake import level_report

OUT = paths.DATA / "bank"
UP_BANK = paths.UPSTREAM_SRC / "reference_bank" / "bank.json"
MIN_S, MAX_S = 4.0, 8.0


@lru_cache(maxsize=1)
def _upstream_bank() -> dict:
    return json.loads(UP_BANK.read_text(encoding="utf-8"))


def meter_stem(meter: str) -> str:
    """Upstream key 'anuṣṭubh' -> 'anushtubh': the name detect_meter_key returns and the wav is called."""
    return _upstream_bank()[meter]["wav"].removesuffix(".wav")


def _meters() -> list:
    return [k for k, v in _upstream_bank().items()
            if not k.startswith("_") and isinstance(v, dict) and "wav" in v]


@lru_cache(maxsize=None)
def _verse_meter(full_verse: str) -> str:
    from chant_text import detect_meter_key
    return detect_meter_key(full_verse)


def pick_reference(clips: list, meter: str = "anuṣṭubh"):
    """Cleanest (highest SNR) 4–8 s clip whose whole verse is in `meter`; None if there is none."""
    best, best_snr = None, None
    for c in clips:
        if _verse_meter(c["full_verse"]) != meter_stem(meter):
            continue
        y, sr = sf.read(c["wav"])
        if not MIN_S <= len(y) / sr <= MAX_S:
            continue
        snr = level_report(y, sr)["snr_db"]
        if best_snr is None or snr > best_snr:
            best, best_snr = c, snr
    return best


def confident(clip: dict, checks: dict, min_margin: float = 0.05) -> bool:
    """The content check (mv.check) heard this clip as its (possibly relabelled) line, clearly."""
    ck = checks.get(clip.get("relabelled_from", clip["id"]))
    return bool(ck) and ck["best"] == clip["id"] and ck["margin"] >= min_margin


def bank_entry(clip: dict, n_syll: int, full_verse: str = "") -> dict:
    info = sf.info(clip["wav"])
    dur = round(info.frames / info.samplerate, 2)
    return {"wav": Path(clip["wav"]).name, "clip_id": clip["id"], "mode": "half", "dur_s": dur,
            "sr": 24000, "sec_per_syll": round(dur / n_syll * 1.10, 3),
            "ref_text": clip["deva"], "full_verse": full_verse}


def build(out: Path = OUT) -> dict:
    from mv.review import kept_clips
    from mv.text import load_sarga
    sessions = sorted(p.parent for p in (paths.DATA / "sessions").glob("*/clips.json"))
    clips, verse_text, loaded = [], {}, set()
    for d in sessions:
        sarga = int(d.name.split("_")[1])
        if sarga not in loaded:
            loaded.add(sarga)
            for v in load_sarga(sarga, include_suspect=True):
                verse_text[f"5.{sarga}.{v.shloka}"] = " । ".join(h.deva for h in v.hemistichs) + " ॥"
        cj = d / "check.json"
        checks = {c["id"]: c for c in json.loads(cj.read_text(encoding="utf-8"))} if cj.exists() else {}
        for c in kept_clips(d):
            if not confident(c, checks):
                continue
            clips.append({**c, "full_verse": verse_text[".".join(c["id"].split(".")[:3])]})
    out.mkdir(parents=True, exist_ok=True)
    bank = {"_doc": "Own-voice reference bank (upstream format). ref_text matches wav exactly (half-reference rule)."}
    for m in _meters():
        c = pick_reference(clips, m)
        if c is None:
            continue
        dst = out / f"{meter_stem(m)}.wav"
        shutil.copyfile(c["wav"], dst)
        bank[m] = bank_entry({**c, "wav": str(dst)}, c["n_syll"], c["full_verse"])
    (out / "bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    return bank


if __name__ == "__main__":
    b = build()
    for k, e in b.items():
        if not k.startswith("_"):
            print(f"{k:16} {e['clip_id']:12} {e['dur_s']:.2f} s  {e['sec_per_syll']} s/syll  {e['ref_text']}")
    print(OUT / "bank.json")
