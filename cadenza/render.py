"""Render + master.

1. **LMMS headless render** — the CLI syntax differs slightly across builds, so
   we try the known variants and use whichever the installed binary accepts.
2. **Master** — pydub + ffmpeg (already on this system): normalize, fades,
   export MP3. Fully headless; no Audacity required. An optional Audacity hook
   is left for manual polish only (needs the GUI app running + mod-script-pipe).
"""
from __future__ import annotations
import os
import subprocess

# LMMS CLI has changed over versions; try each until one renders a file.
_RENDER_VARIANTS = [
    lambda b, mmp, wav: [b, "render", mmp, "-o", wav, "-f", "wav"],
    lambda b, mmp, wav: [b, "--render", mmp, "-o", wav, "-f", "wav"],
    lambda b, mmp, wav: [b, "-r", mmp, "-o", wav, "-f", "wav"],
]


def render_wav(mmp: str, wav: str, lmms_bin: str = "lmms", timeout: int = 300) -> str:
    last = None
    for build in _RENDER_VARIANTS:
        cmd = build(lmms_bin, mmp, wav)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise RuntimeError(f"LMMS binary not found: {lmms_bin}") from e
        if os.path.exists(wav) and os.path.getsize(wav) > 0:
            return wav
        last = f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    raise RuntimeError(f"LMMS render produced no WAV. Last attempt:\n{last}")


# Phase E — mastering FX applied via ffmpeg audio filters (aecho ~ reverb).
_FX = {
    "reverb": "aecho=0.8:0.9:50:0.3",
    "hall": "aecho=0.8:0.9:60|120:0.4|0.3",
}


def master(wav: str, mp3: str, ffmpeg_bin: str = "ffmpeg", fx: str = "none") -> bool:
    """Normalize + fade (+ optional FX) + export MP3. False (keeps WAV) on failure."""
    try:
        from pydub import AudioSegment, effects
        if ffmpeg_bin and ffmpeg_bin != "ffmpeg" and os.path.exists(ffmpeg_bin):
            AudioSegment.converter = ffmpeg_bin
        seg = effects.normalize(AudioSegment.from_wav(wav)).fade_in(300).fade_out(800)
        if fx in _FX:
            tmp = wav[:-4] + "_norm.wav"
            seg.export(tmp, format="wav")
            subprocess.run([ffmpeg_bin, "-y", "-i", tmp, "-af", _FX[fx],
                            "-b:a", "192k", mp3], capture_output=True, text=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return os.path.exists(mp3) and os.path.getsize(mp3) > 0
        seg.export(mp3, format="mp3", bitrate="192k")
        return True
    except Exception as e:  # pydub/ffmpeg missing or decode error
        print(f"[master] skipped ({e}); WAV kept at {wav}")
        return False
