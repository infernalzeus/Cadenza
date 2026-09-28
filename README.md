# Cadenza

An ML-driven generative music pipeline. A **categorised MIDI corpus** feeds a
**generator + classifier** pair that composes arrangements, which are written
out as **LMMS project files**, rendered headless, and polished in **Audacity**.

> Core insight: a *classifier* sorts, it does not *create*. Cadenza uses a
> generator to propose musical candidates and the classifier to **score/filter**
> them against a requested category (genre/mood). The classifier's real job is
> taste, not composition.

## Pipeline

```
MIDI corpus ──> extract features ──> catalog.csv ──┬─> generator (Markov/LSTM) ─┐
                                                    │                            ├─> best pick
                                                    └─> classifier (sklearn) <───┘  (score/filter)
                                                                                     │
                        base.mmp template ──> lmms_writer (.mmp XML) <───────────────┘
                                                    │
                                              LMMS --render ──> WAV ──> Audacity (mod-script-pipe) ──> MP3
```

See [`docs/architecture.canvas`](docs/architecture.canvas) — open the `Cadenza`
folder as an Obsidian vault to view it as a diagram.

## Run it (works today)

```
pip install -r requirements.txt         # mido, pydub, flask (+ ML later)
python -m cadenza --mood calm --bars 8 --seed 1      # CLI -> out/*.mmp/.wav/.mp3
python -m webui.app                                  # web UI -> http://127.0.0.1:5005
```

Requires LMMS installed (path in `config.json`) and ffmpeg on PATH. No corpus
needed — with an empty `data/corpus/` the generator uses a scale-constrained
seeded walk. Drop `.mid` files in `data/corpus/` to train the Markov chain on them.

Moods: `calm`, `sad`, `happy`, `energetic` (drive scale/feel). **Genres**
(`lofi`, `hiphop`, `pop`, `house`, `techno`, `dnb`, `ambient`) set tempo + default
beat. Instruments: E-Piano, Saw, Supersaw, Square, Triangle, Sine. Tone: `bright`/
`warm` (low-pass filter). Every output is reproducible from its `--seed`.

**The web UI is the control surface** — genre/mood/key/bars/bpm/seed, per-part
instrument pickers, drums, tone, candidate count, chord toggle, plus an **Ingest**
tab to analyse songs into the library. All work is driveable from the UI.

### Beyond the roadmap
- **Play/recommend** — each result shows the key, scale, pentatonic and chord
  progression (roman + note names) to play over the track, and a **piano-roll** of the
  generated notes. Generated `.mmp` sets the Melody track MIDI-readable, so a MIDI
  controller plays the lead over the backing in LMMS.
- **Harmonic + BPM matching** (`match.py`) — Camelot-wheel + tempo compatibility over
  the library; "Generate matched" blends a compatible pair into a new track.
- **Vocal removal / karaoke** (`stems.py`, Demucs) — upload a song → vocals +
  instrumental; the instrumental is auto-filed into the library for matching/regeneration.
  Needs `pip install torch demucs` (CPU here — AMD GPU has no Demucs path). Runs ~1–2 min/song.
- **LMMS setup:** a starter `~/lmmsrc.xml` (SDL audio, 512-frame buffer, WinMM MIDI) ships
  to reduce live-play glitches. Raise the buffer to 1024 if you still get crackle, or lower
  to 256 for tighter live response.

**Note (Windows + Anaconda + torch):** `KMP_DUPLICATE_LIB_OK=TRUE` is set in
`cadenza/__init__.py` to avoid the torch/MKL OpenMP `Error #15` crash.

## Why this stack works

| Piece | Role | Automation path |
|-------|------|-----------------|
| **MIDI corpus** | Raw material — loops, progressions, drum patterns | `mido` / `pretty_midi` / `music21` |
| **catalog.csv** | Categorised feature database (the "brain's memory") | pandas |
| **Generator** | Proposes note/loop sequences | Markov chain first; LSTM/Transformer later |
| **Classifier** | Scores candidates vs. requested genre/mood | scikit-learn on the CSV features |
| **LMMS** | Renders the arrangement to audio | `.mmp` is XML → generate it; `lmms render` CLI is headless |
| **Audacity** | Final master — normalize, EQ, fades, export | `mod-script-pipe` (Python over a named pipe) |

**Keep the ML in symbolic (MIDI) space** — it is far more tractable than raw
audio. Audacity/LMMS are the *render + polish* endpoints, not the brains.

## Roadmap

- **Phase 0 — MVP:** ✅ Markov generator → `.mmp` → `lmms render` → ffmpeg master.
- **Phase A — Instrumentation & arrangement:** ✅ *(the sound fix)* real instruments
  (BitInvader e-piano for melody + chords, synth bass), a **drum track** from factory
  `.ogg` samples (kick/hihat/clap, genre-aware beat), and a **chord-progression
  engine** (I–V–vi–IV / i–VI–III–VII). 6-track arrangement instead of a bare synth.
- **Phase B — Classifier / critic:** ✅ RandomForest (scikit-learn) trained from a
  self-generated labelled dataset (arrangement features → mood). Replaces the density
  heuristic: *generate-many → score by mood-fit → pick-best*. Train/retrain from the
  web UI ("Classifier" card); the winning candidate shows the critic's predicted mood
  + confidence. Falls back to the heuristic if untrained. Next: fold Phase C's
  ingested library into the same feature space.
- **Phase C — Reverse ingest:** ✅ *(MVP)* upload a song in the web UI → librosa
  extracts tempo/key/mode/energy/brightness → heuristic → mood + genre → filed into
  `data/library.csv` (shown in the Library table). **No Audacity needed.** Next:
  instrument/chord/stem extraction, and using library features to drive generation.
- **Phase D — Neural generator:** ✅ *(optional, gated)* `cadenza/neural.py` — a
  PyTorch LSTM over MIDI pitch sequences from `data/corpus/`. Select Generator =
  `neural` in the UI; **falls back to Markov** if torch isn't installed or no model is
  trained (non-breaking). Train from the UI "Neural generator" card. Needs
  `pip install torch` + a MIDI corpus.
- **Phase E — Control & polish:** ✅ **song structure** (`build` = intro/verse/chorus/
  outro — tracks enter/exit and dynamics scale per section), **master FX** (`reverb`/
  `hall` via ffmpeg), plus the existing genre/mood/key/bpm/instrument/tone controls
  and full seed reproducibility. All in the web UI.

**For true piano/orchestral realism:** swap the bundled BitInvader preset for
**Sf2Player + a General-MIDI SoundFont** (a `.sf2` on N:, referenced by absolute
path). One file yields piano/bass/strings/drums via GM program numbers, which map
cleanly onto instrument classification. Needs a one-time ~30–150 MB download.

## LMMS .mmp gotchas (learned the hard way, encoded in lmms_writer.py)

- The `<pattern>` (notes) must be a **child of `<track>`, a sibling of
  `<instrumenttrack>`** — not nested inside it. Nesting it parses fine but LMMS's
  loader finds no track content → song length 0 → *"project is empty, aborting!"*.
- TripleOscillator `modalgo` must be **2 (SignalMix / additive)**. Value 1 is
  Amplitude Modulation, which mutes the oscillators against each other → silent render.
- An `<instrumenttrack>` needs its `<eldata>` envelope block + `<chordcreator>` /
  `<arpeggiator>` to load and sound correctly.
- Render CLI (LMMS 1.2.2): `lmms render song.mmp -o song.wav -f wav`.

## Layout

```
Cadenza/
├── data/
│   ├── corpus/         # raw MIDI (gitignored)
│   └── catalog.csv     # the categorised database
├── templates/
│   └── base.mmp        # hand-built LMMS project to reverse-engineer the XML
├── cadenza/
│   ├── extract.py      # MIDI -> features -> catalog.csv
│   ├── generate.py     # Markov / (later) neural generator
│   ├── classify.py     # train + score candidates
│   ├── lmms_writer.py  # arrangement -> .mmp XML
│   └── render.py       # LMMS CLI render + Audacity post
└── docs/
    └── architecture.canvas
```

## Where files live (and the LMMS working-dir question)

- **Cadenza output** → always `out/` on N: (set in `config.json`). No LMMS setting
  affects this.
- **LMMS factory assets** (drum `.ogg`s, presets) ship in the **install on Z:**
  (`Z:\Programs\LMMS\data\...`). LMMS does not download them. Cadenza references them
  by paths relative to the factory samples dir (e.g. `drums/bassdrum01.ogg`), which
  resolve from the Z: install.
- **Changing LMMS's user working dir does NOT affect Cadenza.** LMMS resolves a
  sample path as: absolute → factory dir (Z:) → user working dir. We only use factory
  paths and absolute paths, so renders are independent of that setting. (The user
  working dir defaults to `Documents\LMMS` on C: and is created on first GUI launch —
  it did not exist at project start.)

## Requirements

See [`requirements.txt`](requirements.txt). External tools: **LMMS** (path in
`config.json`) and **ffmpeg** on PATH. **Audacity is not required** — mastering runs
on ffmpeg/pydub; Audacity's script pipe needs the running GUI app, so it's optional
and only useful for manual hand-editing.
