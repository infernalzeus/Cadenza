"""Play/recommend helper — what YOU can play over a generated project.

Turns an arrangement's key + mood into human-readable guidance: the scale to stay
in, the chord progression (roman numerals + note names) and a "safe" pentatonic
for soloing. Bridges Cadenza -> your MIDI keyboard / LMMS.
"""
from __future__ import annotations

from .generate import CHORD_FAMILY, NOTE_BASE, PROGRESSIONS, SCALES

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ROMAN_MAJ = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
_ROMAN_MIN = ["i", "ii°", "III", "iv", "v", "VI", "VII"]


def _pc(root: str) -> int:
    """Pitch class 0-11 of a root name (LMMS key 48 = C)."""
    return NOTE_BASE.get(root.upper(), 48) % 12


def _family(mood: str) -> str:
    return CHORD_FAMILY.get(mood, "minor")


def scale_notes(root: str, mood: str) -> list[str]:
    scale = SCALES["major" if _family(mood) == "major" else "minor"]
    base = _pc(root)
    return [NAMES[(base + iv) % 12] for iv in scale]


def solo_scale(root: str, mood: str) -> list[str]:
    scale = SCALES["pent_major" if _family(mood) == "major" else "pent_minor"]
    base = _pc(root)
    return [NAMES[(base + iv) % 12] for iv in scale]


def chords(root: str, mood: str) -> list[dict]:
    fam = _family(mood)
    scale = SCALES["major" if fam == "major" else "minor"]
    roman = _ROMAN_MAJ if fam == "major" else _ROMAN_MIN
    base = _pc(root)
    out = []
    for deg in PROGRESSIONS[fam]:
        triad = [(base + scale[(deg + i) % 7]) % 12 for i in (0, 2, 4)]
        quality = "" if (triad[1] - triad[0]) % 12 == 4 else "m"
        out.append({"roman": roman[deg], "name": NAMES[triad[0]] + quality,
                    "notes": [NAMES[t] for t in triad]})
    return out


def recommend(arr: dict) -> dict:
    root, mood = arr["root"], arr["mood"]
    return {"key": f"{root} {'major' if _family(mood) == 'major' else 'minor'}",
            "scale": scale_notes(root, mood),
            "solo": solo_scale(root, mood),
            "chords": chords(root, mood)}
