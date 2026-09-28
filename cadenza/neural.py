"""Phase D — optional neural generator (PyTorch LSTM over MIDI pitches).

**Gated + non-breaking.** If PyTorch isn't installed or no model is trained, callers
fall back to the Markov generator, so the app always runs. Train from a corpus of
MIDI files in ``data/corpus/`` via ``train_model`` (UI: neural "Train" button).

The model predicts a pitch sequence (LMMS keys, C4..C6); note durations are sampled
from the mood's rhythm pool, same as Markov. This is experimental — timbre/quality
still come from Phase A instrumentation.
"""
from __future__ import annotations
import glob
import os

SEQ = 16                 # context window
LOW, HIGH = 48, 84       # LMMS keys C4..C6
VOCAB = HIGH - LOW + 1

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_ROOT, "models", "neural.pt")


def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception:
        return None, None


def available() -> bool:
    """True if PyTorch is installed. Uses find_spec so we don't import torch (and
    trigger its OpenMP init) just to render a page."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def trained(path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(path)


def _build(torch, nn):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 32)
            self.lstm = nn.LSTM(32, 128, batch_first=True)
            self.fc = nn.Linear(128, VOCAB)

        def forward(self, x, h=None):
            out, h = self.lstm(self.emb(x), h)
            return self.fc(out), h

    return Net()


def _corpus_sequences(corpus_dir: str):
    try:
        import mido
    except Exception:
        return []
    seqs = []
    files = glob.glob(os.path.join(corpus_dir, "*.mid")) + \
        glob.glob(os.path.join(corpus_dir, "*.midi"))
    for path in files:
        try:
            mid = mido.MidiFile(path)
        except Exception:
            continue
        s = []
        for msg in mid:
            if msg.type == "note_on" and msg.velocity > 0:
                key = msg.note - 12  # MIDI -> LMMS
                if LOW <= key <= HIGH:
                    s.append(key - LOW)
        if len(s) > SEQ + 1:
            seqs.append(s)
    return seqs


def train_model(corpus_dir: str, path: str = DEFAULT_PATH, epochs: int = 20) -> dict:
    torch, nn = _torch()
    if torch is None:
        raise RuntimeError("PyTorch not installed (pip install torch)")
    import random
    seqs = _corpus_sequences(corpus_dir)
    if not seqs:
        raise RuntimeError(f"No trainable MIDI found in {corpus_dir}")
    windows = [(s[i:i + SEQ], s[i + 1:i + SEQ + 1])
               for s in seqs for i in range(len(s) - SEQ)]
    if not windows:
        raise RuntimeError("Corpus sequences too short to train")
    model = _build(torch, nn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        random.shuffle(windows)
        for i in range(0, len(windows), 64):
            batch = windows[i:i + 64]
            X = torch.tensor([b[0] for b in batch])
            Y = torch.tensor([b[1] for b in batch])
            opt.zero_grad()
            out, _h = model(X)
            loss = lossf(out.reshape(-1, VOCAB), Y.reshape(-1))
            loss.backward()
            opt.step()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return {"sequences": len(seqs), "windows": len(windows), "epochs": epochs}


def generate_melody(bars: int, durs: list[int], rng, path: str = DEFAULT_PATH):
    """Return melody notes [{key,pos,len,vol}] or None (→ caller falls back)."""
    torch, nn = _torch()
    if torch is None or not os.path.exists(path):
        return None
    try:
        model = _build(torch, nn)
        model.load_state_dict(torch.load(path))
        model.eval()
    except Exception:
        return None
    import numpy as np

    cur = [rng.randint(0, VOCAB - 1) for _ in range(SEQ)]
    notes, pos, total = [], 0, bars * 192
    with torch.no_grad():
        while pos < total:
            x = torch.tensor([cur[-SEQ:]])
            out, _h = model(x)
            probs = torch.softmax(out[0, -1], dim=0).numpy()
            nxt = int(rng.choices(range(VOCAB), weights=probs.tolist())[0])
            cur.append(nxt)
            length = rng.choice(durs)
            if pos + length > total:
                length = total - pos
            notes.append({"key": LOW + nxt, "pos": pos, "len": length,
                          "vol": rng.randint(75, 100)})
            pos += length
    return notes
