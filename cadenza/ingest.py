"""Phase C — reverse ingest: analyse a song you like -> features -> library.

Drop in an audio file (mp3/wav/flac/ogg/m4a); librosa extracts tempo, key/mode,
duration, energy and brightness; a heuristic maps those to mood + genre; the row
is appended to data/library.csv. This is the input-side mirror of the render step
(no Audacity needed). Full instrument/chord/stem extraction is a later sub-phase.
"""
from __future__ import annotations
import csv
import os
import time

import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

LIB_COLUMNS = ["id", "file", "title", "genre", "mood", "key", "mode", "bpm",
               "duration_s", "energy", "brightness", "added"]


def _detect_key(chroma_mean):
    best = None
    for i in range(12):
        for mode, prof in (("major", _MAJ), ("minor", _MIN)):
            score = np.corrcoef(np.roll(prof, i), chroma_mean)[0, 1]
            if best is None or score > best[2]:
                best = (NOTES[i], mode, score)
    return best[0], best[1]


def _mood(tempo, mode):
    fast, slow = tempo >= 115, tempo < 90
    if mode == "minor":
        return "sad" if slow else ("energetic" if fast else "calm")
    return "calm" if slow else ("energetic" if fast else "happy")


def _genre(tempo):
    for lim, name in ((90, "lofi"), (108, "hiphop"), (126, "house"), (140, "techno")):
        if tempo < lim:
            return name
    return "dnb"


def analyze(path: str) -> dict:
    import librosa
    y, sr = librosa.load(path, mono=True, duration=90)  # cap length for speed
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = _detect_key(chroma.mean(axis=1))
    return {
        "key": key, "mode": mode, "bpm": round(tempo, 1),
        "duration_s": round(float(librosa.get_duration(y=y, sr=sr)), 1),
        "energy": round(float(np.mean(librosa.feature.rms(y=y))), 4),
        "brightness": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))),
        "mood": _mood(tempo, mode), "genre": _genre(tempo),
    }


def read_library(lib_csv: str) -> list[dict]:
    if not os.path.exists(lib_csv):
        return []
    with open(lib_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_to_library(analysis: dict, src_path: str, lib_csv: str) -> dict:
    os.makedirs(os.path.dirname(lib_csv), exist_ok=True)
    existing = read_library(lib_csv)
    base = os.path.basename(src_path)
    row = {
        "id": f"song_{len(existing) + 1:03d}", "file": base,
        "title": os.path.splitext(base)[0], "genre": analysis["genre"],
        "mood": analysis["mood"], "key": analysis["key"], "mode": analysis["mode"],
        "bpm": analysis["bpm"], "duration_s": analysis["duration_s"],
        "energy": analysis["energy"], "brightness": analysis["brightness"],
        "added": time.strftime("%Y-%m-%d"),
    }
    write_header = not os.path.exists(lib_csv)
    with open(lib_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIB_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row
