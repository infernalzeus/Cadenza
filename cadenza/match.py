"""DJ-style matching over the ingested library (Rekordbox-like).

Given songs analysed by Phase C (bpm + key + mode), find pairs that are BPM- and
harmonically compatible (Camelot wheel), and blend a pair into generation targets
so Cadenza can create NEW music that mixes with them.
"""
from __future__ import annotations

# Camelot wheel: major = B ring, minor = A ring.
MAJOR = {"C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
         "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B"}
MINOR = {"A": "8A", "A#": "3A", "B": "10A", "C": "5A", "C#": "12A", "D": "7A",
         "D#": "2A", "E": "9A", "F": "4A", "F#": "11A", "G": "6A", "G#": "1A"}


def camelot(key: str, mode: str) -> str:
    table = MAJOR if str(mode).lower().startswith("maj") else MINOR
    return table.get(key, "?")


def _parse(code: str):
    if not code or code == "?" or code[-1] not in "AB":
        return None
    try:
        return int(code[:-1]), code[-1]
    except ValueError:
        return None


def key_compatible(c1: str, c2: str):
    """Return a reason string if harmonically compatible, else None."""
    a, b = _parse(c1), _parse(c2)
    if not a or not b:
        return None
    if c1 == c2:
        return "same key"
    if a[0] == b[0] and a[1] != b[1]:
        return "relative major/minor"
    if a[1] == b[1] and (abs(a[0] - b[0]) in (1, 11)):
        return "adjacent (+/-1)"
    return None


def bpm_compatible(b1: float, b2: float, tol: float = 0.08):
    """Return a reason string if tempos mix, else None (allows half/double time)."""
    b1, b2 = float(b1), float(b2)
    if not b1 or not b2:
        return None
    for factor, label in ((1.0, "same tempo"), (2.0, "double-time"), (0.5, "half-time")):
        if abs(b1 / b2 - factor) <= tol * factor:
            return label
    return None


def find_pairs(library: list[dict]) -> list[dict]:
    """All BPM+harmonically compatible pairs in the library, best first."""
    pairs = []
    for i in range(len(library)):
        for j in range(i + 1, len(library)):
            a, b = library[i], library[j]
            ca = camelot(a.get("key", ""), a.get("mode", ""))
            cb = camelot(b.get("key", ""), b.get("mode", ""))
            kr = key_compatible(ca, cb)
            br = bpm_compatible(a.get("bpm", 0), b.get("bpm", 0))
            if kr and br:
                spread = abs(float(a.get("bpm", 0)) - float(b.get("bpm", 0)))
                pairs.append({"a": a, "b": b, "camelot_a": ca, "camelot_b": cb,
                              "key_reason": kr, "bpm_reason": br,
                              "score": (0 if kr == "same key" else 1) + spread / 100})
    pairs.sort(key=lambda p: p["score"])
    return pairs


def blend(a: dict, b: dict) -> dict:
    """Generation target that mixes with both songs: shared key, tween BPM, mood."""
    return {
        "root": a.get("key", "C"),
        "mode": a.get("mode", "minor"),
        "bpm": int(round((float(a.get("bpm", 120)) + float(b.get("bpm", 120))) / 2)),
        "mood": a.get("mood", "calm"),
    }
