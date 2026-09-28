"""Central config — paths for LMMS, ffmpeg, and where output lands.

Everything Cadenza produces goes under ``output_dir`` (default: the project's
``out/`` on N:), so the pipeline never writes to C:. Edit ``config.json`` to
repoint these (e.g. move outputs onto an N: resources folder later).
"""
from __future__ import annotations
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "lmms_bin": r"Z:\Programs\LMMS\lmms.exe",
    "ffmpeg_bin": "ffmpeg",
    "output_dir": "out",
    "corpus_dir": "data/corpus",
    "library_csv": "data/library.csv",
    "uploads_dir": "data/library",
    "model_path": "models/mood_clf.pkl",
}


def _abs(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = os.path.join(ROOT, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    for key in ("output_dir", "corpus_dir", "library_csv", "uploads_dir", "model_path"):
        cfg[key] = _abs(ROOT, cfg[key])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["uploads_dir"], exist_ok=True)
    return cfg
