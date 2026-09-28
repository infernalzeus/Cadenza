"""Orchestrates the full run: generate -> score -> .mmp -> WAV -> MP3."""
from __future__ import annotations
import os

from . import classify, config, generate, lmms_writer, render


def run(genre="lofi", mood="calm", root="C", bars=8, bpm=None, seed=None,
        n_candidates=8, name=None, opts=None, do_render=True, do_master=True):
    cfg = config.load()

    candidates = generate.propose(genre, mood, root, bars,
                                  n_candidates=n_candidates, seed=seed,
                                  corpus_dir=cfg["corpus_dir"], opts=opts)
    model = classify.load_model(cfg["model_path"])
    arr = classify.pick_best(candidates, mood, model)
    if bpm:
        arr["bpm"] = int(bpm)

    # name after the WINNER's seed so the file is reproducible from the UI
    stem = name or f"{genre}_{mood}_{root}_{arr['seed']}"
    stem = "".join(c for c in stem if c.isalnum() or c in "_-").strip("_")
    out = cfg["output_dir"]
    result = {"arrangement": arr, "stem": stem,
              "mmp": os.path.join(out, stem + ".mmp"),
              "wav": None, "mp3": None}

    lmms_writer.write_project(arr, result["mmp"])
    if not do_render:
        return result

    wav = os.path.join(out, stem + ".wav")
    result["wav"] = render.render_wav(result["mmp"], wav, cfg["lmms_bin"])

    if do_master:
        mp3 = os.path.join(out, stem + ".mp3")
        fx = (opts or {}).get("fx", arr.get("fx", "none"))
        if render.master(result["wav"], mp3, cfg["ffmpeg_bin"], fx=fx):
            result["mp3"] = mp3
    return result
