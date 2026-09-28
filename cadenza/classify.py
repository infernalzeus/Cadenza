"""Classifier / critic (Phase B) — the *taste*, not the composer.

Turns the discriminative model into a creative *selector*: the generator proposes
N candidate arrangements, this scores each one's fit to the requested mood, and the
best is kept (generate-many -> score -> pick-best). See the wiki concept
[[generative-vs-discriminative-models]].

The model is a RandomForest trained on a self-generated, labelled dataset: we
render feature vectors for many arrangements across every mood/genre and let the
classifier learn which feature shapes read as which mood. No external data needed;
Phase C's ingested library can later be folded into the same feature space.

Falls back to a density heuristic when scikit-learn or a trained model is absent,
so the pipeline always runs.
"""
from __future__ import annotations
import os
from statistics import mean

from .generate import GENRES, MOOD, propose

# Fixed feature order — MUST match between training and scoring.
FEATURE_KEYS = ["bpm_norm", "density", "prange", "notes", "mean_pitch_norm",
                "chord_minorness", "drum_code", "n_instruments", "has_chords"]
_DRUM_CODE = {"off": 0, "backbeat": 1, "four_floor": 2}


def _melody_track(arr):
    return next((t for t in arr["tracks"] if t["name"] == "Melody"), None)


def _chord_minorness(arr):
    """1 = minor triad, 0 = major, 0.5 = unknown — from the first chord's third."""
    for t in arr["tracks"]:
        if t["name"] == "Chords" and t["notes"]:
            first_pos = t["notes"][0]["pos"]
            chord = sorted(n["key"] for n in t["notes"] if n["pos"] == first_pos)
            if len(chord) >= 2:
                third = chord[1] - chord[0]
                return 1.0 if third == 3 else (0.0 if third == 4 else 0.5)
    return 0.5


def featvec(arr) -> list[float]:
    mel = _melody_track(arr)
    keys = [n["key"] for n in (mel["notes"] if mel else [])]
    return [
        arr["bpm"] / 200.0,
        float(arr["stats"]["density"]),
        float(arr["stats"]["range"]),
        float(arr["stats"]["notes"]),
        (mean(keys) if keys else 60) / 127.0,
        _chord_minorness(arr),
        float(_DRUM_CODE.get(arr.get("drum_style", "off"), 0)),
        float(arr["stats"]["instruments"]),
        1.0 if any(t["name"] == "Chords" for t in arr["tracks"]) else 0.0,
    ]


def build_training_set(per_class: int = 25):
    """Generate a labelled dataset (features -> mood) from our own generator."""
    roots = ["C", "E", "G", "A"]
    genres = list(GENRES)
    X, y = [], []
    for mood in MOOD:
        for i in range(per_class):
            arr = propose(genres[i % len(genres)], mood, roots[i % len(roots)],
                          bars=8, n_candidates=1, seed=1000 + i)[0]
            X.append(featvec(arr))
            y.append(mood)
    return X, y


def train_model(path: str, per_class: int = 25) -> dict:
    """Train + persist the mood classifier. Returns a summary (n, test accuracy)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    X, y = build_training_set(per_class)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=0)
    clf.fit(Xtr, ytr)
    acc = round(float(clf.score(Xte, yte)), 3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_KEYS, "n": len(y),
                 "classes": list(clf.classes_), "accuracy": acc}, path)
    return {"n": len(y), "accuracy": acc, "classes": list(clf.classes_)}


def load_model(path: str):
    try:
        import joblib
        if path and os.path.exists(path):
            return joblib.load(path)
    except Exception:
        return None
    return None


def _heuristic(arr, target_mood: str) -> float:
    target = MOOD.get(target_mood, MOOD["calm"])["density"]
    return -abs(arr["stats"]["density"] - target) + 0.02 * arr["stats"]["range"]


def score(arr, target_mood: str, model=None) -> float:
    """Fit of `arr` to `target_mood`: model probability, or heuristic fallback."""
    if not model:
        return _heuristic(arr, target_mood)
    classes = list(model["model"].classes_)
    if target_mood not in classes:
        return 0.0
    proba = model["model"].predict_proba([featvec(arr)])[0]
    return float(proba[classes.index(target_mood)])


def pick_best(candidates, target_mood: str, model=None):
    best = max(candidates, key=lambda a: score(a, target_mood, model))
    if model:
        classes = list(model["model"].classes_)
        proba = model["model"].predict_proba([featvec(best)])[0]
        best["classifier_used"] = True
        best["classifier_pred"] = classes[int(proba.argmax())]
        best["classifier_conf"] = round(
            float(proba[classes.index(target_mood)]) if target_mood in classes else 0.0, 3)
    else:
        best["classifier_used"] = False
    return best
