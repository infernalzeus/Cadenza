"""Generator — proposes musical arrangements (Phase 0: Markov).

Two modes:
  * **Corpus-trained** — if data/corpus/*.mid exists, learn an order-1 Markov
    chain over pitch transitions (via mido) and sample from it.
  * **Seeded walk** — with no corpus, do a scale-constrained weighted random
    walk (an order-1 Markov with a hand-set transition matrix biased to
    stepwise motion). Runs with zero input so the pipeline works today.

Emits *candidates*; classify.pick_best chooses one (generate-many -> score ->
pick-best). Everything is symbolic — see the wiki concept symbolic-music-ml.

Timing model: LMMS uses 192 ticks per 4/4 bar (48 = quarter note).
Pitch model: values are LMMS keys (57 = A4, 48 = C4, 60 = C5).
"""
from __future__ import annotations
import glob
import os
import random
from collections import Counter, defaultdict

try:
    import mido
except ImportError:  # optional — only needed for corpus training
    mido = None

TICKS_PER_BAR = 192
QUARTER = 48

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "pent_major": [0, 2, 4, 7, 9],
    "pent_minor": [0, 3, 5, 7, 10],
}

# LMMS key for each root in octave 4 (A4 = 57).
NOTE_BASE = {"C": 48, "C#": 49, "D": 50, "D#": 51, "E": 52, "F": 53,
             "F#": 54, "G": 55, "G#": 56, "A": 57, "A#": 58, "B": 59}

# mood -> musical defaults (scale, tempo, target note-density/bar, rhythm pool)
MOOD = {
    "calm":      {"scale": "dorian",     "bpm": 78,  "density": 6,  "durs": [96, 48, 48, 48, 24]},
    "sad":       {"scale": "minor",      "bpm": 70,  "density": 5,  "durs": [96, 96, 48, 48]},
    "happy":     {"scale": "major",      "bpm": 120, "density": 8,  "durs": [48, 48, 24, 24, 24]},
    "energetic": {"scale": "pent_major", "bpm": 132, "density": 10, "durs": [24, 24, 24, 48, 12]},
}

# order-1 transition weights over scale-degree steps (stepwise motion favoured)
STEP_WEIGHTS = {0: 0.10, 1: 0.24, -1: 0.24, 2: 0.12, -2: 0.12,
                3: 0.06, -3: 0.06, 4: 0.03, -4: 0.03}


def build_pool(root: str, scale: str, octaves: int = 2) -> list[int]:
    base = NOTE_BASE.get(root.upper(), 48)
    intervals = SCALES.get(scale, SCALES["minor"])
    pool = []
    for octv in range(octaves):
        for iv in intervals:
            pool.append(base + 12 * octv + iv)
    return sorted(set(pool))


def _weighted(rng: random.Random, weights: dict) -> int:
    keys, wts = zip(*weights.items())
    return rng.choices(keys, weights=wts, k=1)[0]


def corpus_model(corpus_dir: str):
    """Order-1 pitch-transition chain from MIDI, or None if unavailable."""
    if mido is None:
        return None
    files = glob.glob(os.path.join(corpus_dir, "*.mid")) + \
        glob.glob(os.path.join(corpus_dir, "*.midi"))
    if not files:
        return None
    trans: dict[int, Counter] = defaultdict(Counter)
    for path in files:
        try:
            mid = mido.MidiFile(path)
        except Exception:
            continue
        prev = None
        for msg in mid:
            if msg.type == "note_on" and msg.velocity > 0:
                key = msg.note - 12  # MIDI note -> LMMS key
                if prev is not None:
                    trans[prev][key] += 1
                prev = key
    return trans or None


def _melody(rng, pool, bars, durs, model=None):
    """Generate a melody: list of {key,pos,len,vol} for `bars` bars."""
    total = bars * TICKS_PER_BAR
    notes, pos = [], 0
    idx = len(pool) // 2
    key = pool[idx]
    while pos < total:
        length = rng.choice(durs)
        if pos + length > total:
            length = total - pos
        notes.append({"key": key, "pos": pos, "len": length,
                      "vol": rng.randint(75, 100)})
        pos += length
        # pick next pitch
        if model is not None and key in model and model[key]:
            nxt, wts = zip(*model[key].items())
            key = rng.choices(nxt, weights=wts, k=1)[0]
        else:
            idx = max(0, min(len(pool) - 1, idx + _weighted(rng, STEP_WEIGHTS)))
            key = pool[idx]
    return notes


# chord family + 4-bar progressions (0-based scale degrees)
CHORD_FAMILY = {"happy": "major", "energetic": "major", "calm": "minor", "sad": "minor"}
PROGRESSIONS = {"major": [0, 4, 5, 3], "minor": [0, 5, 2, 6]}  # I-V-vi-IV / i-VI-III-VII

# drum step positions within a 192-tick bar
BEATS = {
    "four_floor": {"drums/bassdrum01.ogg": [0, 48, 96, 144],
                   "drums/hihat_closed01.ogg": [24, 72, 120, 168],
                   "drums/clap01.ogg": [48, 144]},
    "backbeat": {"drums/bassdrum01.ogg": [0, 96],
                 "drums/hihat_closed01.ogg": [0, 24, 48, 72, 96, 120, 144, 168],
                 "drums/clap01.ogg": [48, 144]},
}
BEAT_STYLE = {"happy": "four_floor", "energetic": "four_floor",
              "calm": "backbeat", "sad": "backbeat"}

# Selectable instrument palette (friendly name -> lmms_writer spec).
INSTRUMENTS = {
    "E-Piano": {"kind": "bitinvader"},
    "Saw synth": {"kind": "tripleoscillator", "wavetype": 2},
    "Supersaw": {"kind": "tripleoscillator", "wavetype": 2,
                 "osc": {"stphdetun0": "20", "stphdetun1": "-20", "stphdetun2": "12"}},
    "Square synth": {"kind": "tripleoscillator", "wavetype": 3},
    "Triangle synth": {"kind": "tripleoscillator", "wavetype": 1},
    "Sine": {"kind": "tripleoscillator", "wavetype": 0},
}
DRUM_CHOICES = ["auto", "four_floor", "backbeat", "off"]
DRUM_LABEL = {"drums/bassdrum01.ogg": "Kick",
              "drums/hihat_closed01.ogg": "Hi-hat", "drums/clap01.ogg": "Clap"}
TONE_CHOICES = ["bright", "warm"]  # maps to a low-pass filter cutoff
_FCUT = {"bright": 14000, "warm": 1600}

# Genre presets — these ACTUALLY change the music (tempo + default beat).
# (mood still drives scale/feel; explicit UI bpm/drums selections override.)
GENRES = {
    "lofi":    {"bpm": 78,  "drums": "backbeat"},
    "hiphop":  {"bpm": 92,  "drums": "backbeat"},
    "pop":     {"bpm": 116, "drums": "four_floor"},
    "house":   {"bpm": 124, "drums": "four_floor"},
    "techno":  {"bpm": 132, "drums": "four_floor"},
    "dnb":     {"bpm": 172, "drums": "four_floor"},
    "ambient": {"bpm": 70,  "drums": "off"},
}

STRUCTURE_CHOICES = ["loop", "build"]      # loop = uniform; build = intro/verse/chorus/outro
GENERATOR_CHOICES = ["markov", "neural"]   # neural needs torch + a trained model
FX_CHOICES = ["none", "reverb", "hall"]    # applied in mastering (ffmpeg)

DEFAULT_OPTS = {"melody_inst": "E-Piano", "chords_inst": "E-Piano",
                "bass_inst": "Square synth", "drums": "auto", "chords": True,
                "tone": "bright", "structure": "loop", "generator": "markov",
                "fx": "none"}


def _triad(base: int, scale: list[int], deg: int) -> list[int]:
    """Diatonic triad (root, third, fifth) from scale degree `deg`."""
    out = []
    for i in (0, 2, 4):
        idx = deg + i
        out.append(base + scale[idx % 7] + 12 * (idx // 7))
    return out


def _chords(root, mood, bars):
    fam = CHORD_FAMILY.get(mood, "minor")
    scale = SCALES[fam]
    prog = PROGRESSIONS[fam]
    base = NOTE_BASE.get(root.upper(), 48)
    chord_notes, bass_notes, roots = [], [], []
    for b in range(bars):
        deg = prog[b % len(prog)]
        triad = _triad(base, scale, deg)
        roots.append(triad[0])
        p = b * TICKS_PER_BAR
        for k in triad:
            chord_notes.append({"key": k, "pos": p, "len": TICKS_PER_BAR, "vol": 60})
        bass_notes.append({"key": triad[0] - 12, "pos": p, "len": 96, "vol": 85})
        bass_notes.append({"key": triad[0] - 12, "pos": p + 96, "len": 96, "vol": 80})
    return chord_notes, bass_notes


def _drums(style_name, bars):
    style = BEATS[style_name]
    vols = {"drums/bassdrum01.ogg": 100, "drums/hihat_closed01.ogg": 65,
            "drums/clap01.ogg": 90}
    tracks = []
    for src, steps in style.items():
        notes = [{"key": 57, "pos": b * TICKS_PER_BAR + s, "len": 24,
                  "vol": vols[src]} for b in range(bars) for s in steps]
        tracks.append({"name": src.split("/")[-1].split(".")[0],
                       "label": DRUM_LABEL[src], "basenote": 57, "vol": 55,
                       "instrument": {"kind": "audiofileprocessor", "src": src},
                       "notes": notes})
    return tracks


def _gen_melody(rng, pool, bars, durs, model, opts):
    """Pick the generator. Neural if requested + available, else Markov."""
    if opts.get("generator") == "neural":
        try:
            from . import neural
            notes = neural.generate_melody(bars, durs, rng)
            if notes:
                return notes, "neural"
        except Exception:
            pass
    return [dict(n, key=n["key"] + 12) for n in _melody(rng, pool, bars, durs, model)], "markov"


def _role(track: dict) -> str:
    if track["instrument"]["kind"] == "audiofileprocessor":
        return "drums"
    return track["name"].lower()


def _sectionize(bars: int, structure: str):
    """Return [(name, start_bar, end_bar, active_roles, vol_scale)] or None."""
    if structure != "build" or bars < 4:
        return None
    intro = max(1, bars // 8)
    outro = max(1, bars // 8)
    mid = bars - intro - outro
    verse = mid // 2
    full = {"chords", "melody", "bass", "drums"}
    return [
        ("intro", 0, intro, {"chords", "drums"}, 0.7),
        ("verse", intro, intro + verse, full, 0.9),
        ("chorus", intro + verse, bars - outro, full, 1.0),
        ("outro", bars - outro, bars, {"chords"}, 0.6),
    ]


def _apply_structure(tracks, bars, structure):
    secs = _sectionize(bars, structure)
    if not secs:
        return tracks

    def lookup(bar):
        for _name, s, e, roles, vol in secs:
            if s <= bar < e:
                return roles, vol
        return None, 1.0

    out = []
    for t in tracks:
        role = _role(t)
        notes = []
        for n in t["notes"]:
            roles, vol = lookup(n["pos"] // TICKS_PER_BAR)
            if roles is None or role in roles:
                notes.append(dict(n, vol=max(1, min(100, int(n.get("vol", 100) * vol)))))
        out.append(dict(t, notes=notes))
    return out


def _arrangement(rng, genre, mood, root, bars, model, opts, seed_val):
    m = MOOD.get(mood, MOOD["calm"])
    pool = build_pool(root, m["scale"])
    raw_mel, gen_used = _gen_melody(rng, pool, bars, m["durs"], model, opts)
    mel = raw_mel
    chord_notes, bass_notes = _chords(root, mood, bars)
    fam = CHORD_FAMILY.get(mood, "minor")
    prog_label = "I–V–vi–IV" if fam == "major" else "i–VI–III–VII"

    fcut = _FCUT.get(opts.get("tone", "bright"), 14000)
    # Volumes kept low for headroom — summed voices must stay < 0 dBFS or LMMS
    # hard-clips the mix (the "glitch"); mastering then normalizes cleanly.
    tracks = []
    if opts["chords"]:
        tracks.append({"name": "Chords", "label": opts["chords_inst"], "fcut": fcut,
                       "instrument": INSTRUMENTS[opts["chords_inst"]],
                       "basenote": 57, "vol": 32, "notes": chord_notes})
    tracks.append({"name": "Melody", "label": opts["melody_inst"], "fcut": fcut,
                   "instrument": INSTRUMENTS[opts["melody_inst"]], "midi_in": True,
                   "basenote": 57, "vol": 48, "notes": mel})
    tracks.append({"name": "Bass", "label": opts["bass_inst"], "fcut": fcut,
                   "instrument": INSTRUMENTS[opts["bass_inst"]],
                   "basenote": 57, "vol": 40, "notes": bass_notes})
    dstyle = opts["drums"]
    if dstyle == "auto":
        dstyle = GENRES[genre]["drums"] if genre in GENRES else BEAT_STYLE.get(mood, "backbeat")
    if dstyle != "off":
        tracks += _drums(dstyle, bars)

    structure = opts.get("structure", "loop")
    tracks = _apply_structure(tracks, bars, structure)

    bpm = GENRES[genre]["bpm"] if genre in GENRES else m["bpm"]
    keys = [n["key"] for n in mel]
    return {
        "genre": genre, "mood": mood, "root": root, "bars": bars, "bpm": bpm,
        "seed": seed_val, "progression": prog_label, "drum_style": dstyle,
        "tone": opts.get("tone", "bright"), "structure": structure,
        "generator": gen_used, "fx": opts.get("fx", "none"), "tracks": tracks,
        "stats": {
            "density": round(len(mel) / bars, 2),
            "range": (max(keys) - min(keys)) if keys else 0,
            "notes": len(mel), "instruments": len(tracks),
        },
    }


def propose(genre="lofi", mood="calm", root="C", bars=8, n_candidates=8,
            seed=None, corpus_dir=None, opts=None):
    """Return N candidate arrangements (dicts). Each is tagged with the exact
    ``seed`` that reproduces it (re-run with that seed + candidates=1)."""
    opts = {**DEFAULT_OPTS, **(opts or {})}
    model = corpus_model(corpus_dir) if corpus_dir else None
    base = seed if seed is not None else random.randint(0, 1_000_000)
    return [_arrangement(random.Random(base + i), genre, mood, root, bars,
                         model, opts, base + i)
            for i in range(n_candidates)]
