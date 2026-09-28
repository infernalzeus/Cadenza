"""CLI:  python -m cadenza --genre lofi --mood calm --bars 8 --seed 1"""
from __future__ import annotations
import argparse

from . import generate, pipeline


def main():
    ap = argparse.ArgumentParser(prog="cadenza", description="Generate music.")
    ap.add_argument("--genre", default="lofi")
    ap.add_argument("--mood", default="calm", choices=list(generate.MOOD))
    ap.add_argument("--root", default="C", choices=list(generate.NOTE_BASE))
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--bpm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--structure", default="loop", choices=generate.STRUCTURE_CHOICES)
    ap.add_argument("--generator", default="markov", choices=generate.GENERATOR_CHOICES)
    ap.add_argument("--fx", default="none", choices=generate.FX_CHOICES)
    ap.add_argument("--no-render", action="store_true", help="write .mmp only")
    ap.add_argument("--no-master", action="store_true", help="skip MP3 export")
    a = ap.parse_args()

    opts = {"structure": a.structure, "generator": a.generator, "fx": a.fx}
    res = pipeline.run(a.genre, a.mood, a.root, a.bars, a.bpm, a.seed,
                       a.candidates, a.name, opts=opts,
                       do_render=not a.no_render, do_master=not a.no_master)
    print("mmp:", res["mmp"])
    print("wav:", res["wav"])
    print("mp3:", res["mp3"])
    print("stats:", res["arrangement"]["stats"])


if __name__ == "__main__":
    main()
