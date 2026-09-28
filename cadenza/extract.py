"""MIDI corpus -> feature rows -> catalog.csv.

Walks data/corpus/, extracts per-loop features (role, key, bpm, note density,
mean pitch, syncopation) and appends categorised rows to data/catalog.csv.
Genre/mood are supplied by the curator (you) — the ML learns from those labels.
"""
from __future__ import annotations

# import pretty_midi, music21, pandas as pd


def extract_features(midi_path: str) -> dict:
    """Return a feature dict for one MIDI file (pitch stats, density, key...)."""
    raise NotImplementedError


def build_catalog(corpus_dir: str = "data/corpus", out_csv: str = "data/catalog.csv") -> None:
    """Scan corpus, extract features, write/merge catalog.csv."""
    raise NotImplementedError
