"""The Sundarakāṇḍa text, as the units we record and train on.

A *hemistich* is one half-verse (two pādas, ending at a daṇḍa) — one line on the recording screen,
one training clip. Source: DharmicData's Vālmīki Rāmāyaṇa (2810 numbered verses, 68 sargas).
Corrections to source typos go in data/text/fixes.json as {"5.1.3": "<corrected verse text>"}.

The source has scrape damage in ~2–3 % of verses (Latin letters like "पाताLअतल", merged lines).
Those verses are marked `suspect` and left out of recording/training unless fixed in fixes.json —
a wrong label teaches the model a wrong pronunciation.
"""
import json
import re
import urllib.request
from dataclasses import dataclass

from mv import paths
from chant_text import n_aksharas  # upstream syllable (akṣara) counter

SOURCE_URL = ("https://raw.githubusercontent.com/bhavykhatri/DharmicData/main/"
              "ValmikiRamayana/5_sundarakanda.json")
TEXT_JSON = paths.DATA / "text" / "5_sundarakanda.json"
FIXES_JSON = paths.DATA / "text" / "fixes.json"

_NUMBER = re.compile(r"[0-9०-९]+(?:\s*[-.]\s*[0-9०-९]+)*")


@dataclass
class Hemistich:
    id: str          # "5.<sarga>.<shloka>.<k>", k from 1
    deva: str        # clean Devanagari, no daṇḍa
    n_syll: int
    ends_verse: bool


@dataclass
class Verse:
    sarga: int
    shloka: int
    hemistichs: list
    suspect: bool = False


MIN_SYLL, MAX_SYLL = 8, 26   # a real hemistich/pāda line in this text; outside = damaged text


def clean_line(text: str) -> list:
    """One verse's raw text -> its hemistichs. Drops verse numbers (५-१-१ / ५.६८.१),
    treats ॥ and ।। alike as daṇḍas, splits on them."""
    t = _NUMBER.sub(" ", text).replace("॥", "।").replace(".", " ")
    parts = (re.sub(r"\s+", " ", p).strip() for p in t.split("।"))
    return [p for p in parts if n_aksharas(p) >= 4]   # drops number/letter scraps like "णम्", "ः"


def _records() -> list:
    if not TEXT_JSON.exists():
        TEXT_JSON.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, TEXT_JSON)
    return json.loads(TEXT_JSON.read_text(encoding="utf-8"))


def _is_suspect(raw: str, hems: list) -> bool:
    return bool(re.search(r"[A-Za-z]", raw)) or not hems or any(
        not MIN_SYLL <= h.n_syll <= MAX_SYLL for h in hems)


def load_sarga(sarga: int, start: int = 1, include_suspect: bool = False) -> list:
    fixes = json.loads(FIXES_JSON.read_text(encoding="utf-8")) if FIXES_JSON.exists() else {}
    verses = []
    for r in _records():
        if r["sarg"] != sarga or r["shloka"] < start:
            continue
        vid = f"5.{sarga}.{r['shloka']}"
        raw = fixes.get(vid, r["text"])
        lines = clean_line(raw)
        hems = [Hemistich(f"{vid}.{k}", d, n_aksharas(d), k == len(lines))
                for k, d in enumerate(lines, 1)]
        suspect = _is_suspect(raw, hems)
        if include_suspect or not suspect:
            verses.append(Verse(sarga, r["shloka"], hems, suspect))
    return sorted(verses, key=lambda v: v.shloka)
