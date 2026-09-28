"""Vocal removal / karaoke via Demucs (stem separation).

Splits a song into vocals + instrumental (two-stem mode = fast). The instrumental
is the karaoke track and can be fed back into the library (ingest) so Cadenza can
generate NEW music that matches it. Runs on CPU here (AMD GPU has no CUDA/DirectML
path for Demucs) — expect ~1-2 min per song.
"""
from __future__ import annotations
import os
import subprocess
import sys


def available() -> bool:
    """True if Demucs is installed. find_spec avoids importing torch at page load."""
    import importlib.util
    return importlib.util.find_spec("demucs") is not None


def separate(audio_path: str, out_dir: str, model: str = "htdemucs",
             timeout: int = 1800) -> dict:
    """Run Demucs two-stem separation. Returns {'vocals', 'instrumental'} (wav)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "--two-stems", "vocals",
           "-o", out_dir, audio_path]
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")  # avoid OpenMP DLL clash
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    base = os.path.join(out_dir, model, stem)
    vocals = os.path.join(base, "vocals.wav")
    instrumental = os.path.join(base, "no_vocals.wav")
    if not (os.path.exists(vocals) and os.path.exists(instrumental)):
        raise RuntimeError("Demucs produced no stems.\n"
                           f"stdout:\n{proc.stdout[-800:]}\nstderr:\n{proc.stderr[-800:]}")
    return {"vocals": vocals, "instrumental": instrumental, "model": model, "name": stem}
