"""Cadenza control panel — the app's front end.

Run:  python -m webui.app   ->  http://127.0.0.1:5005
Generate music with full control, and (Phase C) ingest songs you like into the
library. Every capability is surfaced here.
"""
from __future__ import annotations
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, send_from_directory, render_template_string  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from cadenza import (classify, config, generate, ingest, match, neural,  # noqa: E402
                     pipeline, recommend, stems)

app = Flask(__name__)
CFG = config.load()
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff"}

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>Cadenza</title>
<style>
 :root{--bg:#0e0e14;--card:#17171f;--card2:#1e1e28;--line:#2a2a3a;--txt:#e8e8f0;--dim:#a9a9c0;--acc:#8b7cf6;--gr:#4ade80}
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--txt);margin:0;padding:2rem}
 .wrap{max-width:860px;margin:0 auto}
 h1{font-weight:700;letter-spacing:.02em;margin:0}h1 span{color:var(--acc)}
 .sub{color:var(--dim);margin:.35rem 0 1.5rem}
 form{background:var(--card);padding:1.5rem;border-radius:14px}
 .sec{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--acc);margin:1.1rem 0 .6rem;font-weight:700}
 .sec:first-child{margin-top:0}
 .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.9rem}
 label{display:flex;flex-direction:column;font-size:.78rem;color:var(--dim);gap:.35rem}
 input,select{background:var(--bg);color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:.55rem;font-size:.92rem}
 input[type=file]{padding:.4rem}
 .row-inline{display:flex;align-items:center;gap:.5rem;font-size:.85rem;color:var(--dim);margin-top:1.7rem}
 .row-inline input{width:auto}
 button{width:100%;margin-top:1.3rem;background:var(--acc);color:#fff;border:0;border-radius:10px;padding:.85rem;font-size:1.02rem;font-weight:700;cursor:pointer}
 button:hover{filter:brightness(1.08)}
 .card{margin-top:1.5rem;background:var(--card);padding:1.4rem;border-radius:14px}
 .card h2{margin:0 0 .2rem;font-size:1.15rem}
 audio{width:100%;margin:1rem 0 .5rem}
 a{color:var(--acc)}
 .err{color:#ff6b6b;white-space:pre-wrap;font-size:.78rem;font-family:monospace}
 .chips{margin:.5rem 0}
 .chip{display:inline-block;background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:.2rem .55rem;margin:.15rem;font-size:.76rem}
 .chip b{color:var(--acc)}
 table{width:100%;border-collapse:collapse;margin:.6rem 0;font-size:.84rem}
 th,td{text-align:left;padding:.4rem .5rem;border-bottom:1px solid var(--line)}
 th{color:var(--dim);font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em}
 td.n{color:var(--dim);text-align:right}
 .badge{background:var(--card2);border-radius:5px;padding:.1rem .45rem;font-size:.74rem;color:var(--acc)}
 .tabs{display:flex;gap:.5rem;margin:1.3rem 0 0}
 .tab{background:var(--card);border:1px solid var(--line);border-bottom:0;border-radius:10px 10px 0 0;padding:.55rem 1rem;color:var(--dim);cursor:pointer;font-weight:600;font-size:.9rem}
 .tab.active{color:var(--acc);background:var(--card)}
 .pane{display:none}.pane.active{display:block}
 .recent ul,.lib ul{list-style:none;padding:0;margin:.3rem 0 0}
 .recent li{padding:.4rem 0;border-bottom:1px solid var(--line);font-size:.86rem;display:flex;justify-content:space-between}
 .hint{color:var(--dim);font-size:.78rem}
 .ok{color:var(--gr)}
</style></head><body><div class="wrap">
<h1>Cadenza <span>&#9835;</span></h1>
<p class="sub">ML-driven generative music &mdash; generate tracks or ingest songs you like into the library.</p>

<div class="tabs">
 <div class="tab active" onclick="tab('gen',this)">Generate</div>
 <div class="tab" onclick="tab('ing',this)">Ingest a song</div>
</div>

<div id="gen" class="pane active">
<form method="post" action="/generate">
 <div class="sec">Track</div>
 <div class="grid">
  <label>Genre<select name="genre">{% for g in genres %}<option {{'selected' if g==f.genre}}>{{g}}</option>{% endfor %}</select></label>
  <label>Mood<select name="mood">{% for m in moods %}<option {{'selected' if m==f.mood}}>{{m}}</option>{% endfor %}</select></label>
  <label>Key<select name="root">{% for k in roots %}<option {{'selected' if k==f.root}}>{{k}}</option>{% endfor %}</select></label>
  <label>Bars<input name="bars" type="number" min="1" max="64" value="{{f.bars}}"></label>
  <label>BPM (blank=genre)<input name="bpm" type="number" value="{{f.bpm}}"></label>
  <label>Seed (blank=random)<input name="seed" type="number" value="{{f.seed}}"></label>
 </div>
 <div class="sec">Instruments</div>
 <div class="grid">
  <label>Melody<select name="melody_inst">{% for i in instruments %}<option {{'selected' if i==f.melody_inst}}>{{i}}</option>{% endfor %}</select></label>
  <label>Chords<select name="chords_inst">{% for i in instruments %}<option {{'selected' if i==f.chords_inst}}>{{i}}</option>{% endfor %}</select></label>
  <label>Bass<select name="bass_inst">{% for i in instruments %}<option {{'selected' if i==f.bass_inst}}>{{i}}</option>{% endfor %}</select></label>
 </div>
 <div class="sec">Arrangement</div>
 <div class="grid">
  <label>Drums<select name="drums">{% for d in drum_choices %}<option {{'selected' if d==f.drums}}>{{d}}</option>{% endfor %}</select></label>
  <label>Tone<select name="tone">{% for t in tones %}<option {{'selected' if t==f.tone}}>{{t}}</option>{% endfor %}</select></label>
  <label>Structure<select name="structure">{% for s in structures %}<option {{'selected' if s==f.structure}}>{{s}}</option>{% endfor %}</select></label>
  <label>Generator<select name="generator">{% for g in generators %}<option {{'selected' if g==f.generator}}>{{g}}</option>{% endfor %}</select></label>
  <label>FX (master)<select name="fx">{% for x in fxs %}<option {{'selected' if x==f.fx}}>{{x}}</option>{% endfor %}</select></label>
  <label>Candidates (classifier picks best)<input name="candidates" type="number" min="1" max="32" value="{{f.candidates}}"></label>
 </div>
 <div class="row-inline"><input type="checkbox" name="chords" id="ch" {{'checked' if f.chords=='on'}}><label for="ch" style="flex-direction:row">Include chord track</label></div>
 <button>Generate</button>
</form>

{% if error %}<div class="card err">{{error}}</div>{% endif %}
{% if result %}
<div class="card">
 <h2>{{result.stem}}</h2>
 <div class="chips">
  <span class="chip">genre <b>{{result.arrangement.genre}}</b></span>
  <span class="chip"><b>{{result.arrangement.bpm}}</b> bpm</span>
  <span class="chip">key <b>{{result.arrangement.root}} {{result.arrangement.mood}}</b></span>
  <span class="chip"><b>{{result.arrangement.bars}}</b> bars</span>
  <span class="chip">seed <b>{{result.arrangement.seed}}</b></span>
  <span class="chip">chords <b>{{result.arrangement.progression}}</b></span>
  <span class="chip">beat <b>{{result.arrangement.drum_style}}</b></span>
  <span class="chip">tone <b>{{result.arrangement.tone}}</b></span>
  {% if result.arrangement.classifier_used %}<span class="chip">critic <b>{{result.arrangement.classifier_pred}} ({{result.arrangement.classifier_conf}})</b></span>{% endif %}
  <span class="chip">gen <b>{{result.arrangement.generator}}</b></span>
  <span class="chip">structure <b>{{result.arrangement.structure}}</b></span>
  {% if result.arrangement.fx != 'none' %}<span class="chip">fx <b>{{result.arrangement.fx}}</b></span>{% endif %}
 </div>
 <table>
  <tr><th>Track</th><th>Instrument</th><th>Notes</th></tr>
  {% for t in result.arrangement.tracks %}
  <tr><td>{{t.name}}</td><td><span class="badge">{{t.label}}</span></td><td class="n">{{t.notes|length}}</td></tr>
  {% endfor %}
 </table>
 {% if audio %}<audio controls src="/out/{{audio}}"></audio>{% endif %}
 <p>
  {% if result.mp3 %}<a href="/out/{{result.stem}}.mp3" download>MP3</a> &middot; {% endif %}
  {% if result.wav %}<a href="/out/{{result.stem}}.wav" download>WAV</a> &middot; {% endif %}
  <a href="/out/{{result.stem}}.mmp" download>LMMS project (.mmp)</a>
 </p>
 <p class="hint">Edit it in a DAW: open the .mmp in LMMS. &middot; Reproduce: re-enter seed <b>{{result.arrangement.seed}}</b> with Candidates = 1.</p>
 {% if pianoroll %}<div class="sec">Piano-roll</div>{{pianoroll|safe}}
 <p class="hint" style="margin-top:.3rem"><span style="color:#8b7cf6">&#9632;</span> melody &nbsp;<span style="color:#4ade80">&#9632;</span> bass &nbsp;<span style="color:#5a5a75">&#9632;</span> chords</p>{% endif %}
 {% if rec %}
 <div class="sec">Play over this (your MIDI keyboard / LMMS)</div>
 <div class="chips">
  <span class="chip">key <b>{{rec.key}}</b></span>
  <span class="chip">scale <b>{{rec.scale|join(' ')}}</b></span>
  <span class="chip">solo (pentatonic) <b>{{rec.solo|join(' ')}}</b></span>
 </div>
 <table>
  <tr><th>Deg</th><th>Chord</th><th>Notes</th></tr>
  {% for c in rec.chords %}<tr><td>{{c.roman}}</td><td><span class="badge">{{c.name}}</span></td><td>{{c.notes|join(' &middot; ')|safe}}</td></tr>{% endfor %}
 </table>
 <p class="hint">The Melody track already accepts live MIDI &mdash; plug in your controller, open the .mmp in LMMS, and play the lead in these notes over the backing.</p>
 {% endif %}
</div>{% endif %}

<div class="card">
 <div class="sec" style="margin-top:0">Classifier (mood critic)</div>
 {% if clf %}<p class="hint ok">Trained &#10003; RandomForest &middot; {{clf.n}} samples &middot; test accuracy {{clf.accuracy}} &middot; picks best of your candidates by mood fit.</p>
 {% else %}<p class="hint">Not trained &mdash; candidates ranked by a density heuristic. Train a model to score them by mood.</p>{% endif %}
 <form method="post" action="/train" style="padding:0;background:none;margin:0">
  <button style="margin-top:.4rem">{{ 'Retrain classifier' if clf else 'Train classifier' }}</button>
 </form>
 {% if train_result %}<p class="hint ok" style="margin-top:.6rem">&#10003; Trained on {{train_result.n}} samples &mdash; test accuracy {{train_result.accuracy}} (moods: {{train_result.classes|join(', ')}}).</p>{% endif %}
 {% if train_error %}<div class="err" style="margin-top:.6rem">{{train_error}}</div>{% endif %}
</div>

<div class="card">
 <div class="sec" style="margin-top:0">Neural generator (Phase D)</div>
 {% if not nn.available %}<p class="hint">PyTorch not installed &mdash; the "neural" generator falls back to Markov. Run <code>pip install torch</code> to enable, then add MIDI to <code>data/corpus/</code> and train.</p>
 {% elif nn.trained %}<p class="hint ok">Trained &#10003; &mdash; select Generator = <b>neural</b> above to use it.</p>
 {% else %}<p class="hint">PyTorch ready. Add MIDI to <code>data/corpus/</code>, then train an LSTM on it.</p>{% endif %}
 {% if nn.available %}<form method="post" action="/train_neural" style="padding:0;background:none;margin:0"><button style="margin-top:.4rem">{{ 'Retrain neural' if nn.trained else 'Train neural on corpus' }}</button></form>{% endif %}
 {% if nn_result %}<p class="hint ok" style="margin-top:.6rem">&#10003; Trained on {{nn_result.sequences}} MIDI sequences ({{nn_result.windows}} windows, {{nn_result.epochs}} epochs).</p>{% endif %}
 {% if nn_error %}<div class="err" style="margin-top:.6rem">{{nn_error}}</div>{% endif %}
</div>

{% if recent %}
<div class="card recent">
 <div class="sec" style="margin-top:0">Recent outputs</div>
 <ul>{% for r in recent %}<li><span>{{r}}</span><a href="/out/{{r}}" download>&#8681; mp3</a></li>{% endfor %}</ul>
</div>{% endif %}
</div>

<div id="ing" class="pane">
<form method="post" action="/ingest" enctype="multipart/form-data">
 <div class="sec">Ingest a song you like &rarr; analyse &rarr; library</div>
 <label>Audio file (mp3 / wav / flac / ogg / m4a)<input type="file" name="song" accept="audio/*"></label>
 <p class="hint">Extracts tempo, key, mood, energy &amp; brightness (librosa) and files it into <code>data/library.csv</code>. No Audacity needed.</p>
 <button>Analyse &amp; add to library</button>
</form>

{% if ingest_error %}<div class="card err">{{ingest_error}}</div>{% endif %}
{% if ingest_result %}
<div class="card">
 <h2 class="ok">&#10003; Added: {{ingest_result.title}}</h2>
 <div class="chips">
  <span class="chip">key <b>{{ingest_result.key}} {{ingest_result.mode}}</b></span>
  <span class="chip"><b>{{ingest_result.bpm}}</b> bpm</span>
  <span class="chip">mood <b>{{ingest_result.mood}}</b></span>
  <span class="chip">genre <b>{{ingest_result.genre}}</b></span>
  <span class="chip"><b>{{ingest_result.duration_s}}</b>s</span>
  <span class="chip">energy <b>{{ingest_result.energy}}</b></span>
  <span class="chip">brightness <b>{{ingest_result.brightness}}</b> Hz</span>
 </div>
</div>{% endif %}

<form method="post" action="/stems" enctype="multipart/form-data">
 <div class="sec" style="margin-top:0">Vocal removal / karaoke (stem separation)</div>
 {% if stems_available %}
 <label>Song to split<input type="file" name="song" accept="audio/*"></label>
 <p class="hint">Demucs splits into <b>vocals + instrumental</b> (karaoke). Runs on CPU (~1&ndash;2 min/song). The instrumental is auto-added to the library so you can generate a track that mixes with it.</p>
 <button>Split into vocals + instrumental</button>
 {% else %}
 <p class="hint">Demucs not installed &mdash; run <code>pip install demucs torch</code> to enable karaoke / stem separation.</p>
 {% endif %}
</form>
{% if stems_error %}<div class="card err">{{stems_error}}</div>{% endif %}
{% if stems_result %}
<div class="card">
 <h2 class="ok">&#10003; Split: {{stems_result.name}}</h2>
 <audio controls src="/out/{{stems_result.inst_url}}"></audio>
 <p><a href="/out/{{stems_result.inst_url}}" download>&#8681; Instrumental (karaoke)</a> &nbsp;&middot;&nbsp; <a href="/out/{{stems_result.voc_url}}" download>&#8681; Vocals</a></p>
 <p class="hint ok">Instrumental added to the library &mdash; see Harmonic &amp; BPM matches below to generate a track that mixes with it.</p>
</div>{% endif %}

<div class="card lib">
 <div class="sec" style="margin-top:0">Library ({{library|length}} songs)</div>
 {% if library %}
 <table>
  <tr><th>Title</th><th>Genre</th><th>Mood</th><th>Key</th><th>Camelot</th><th>BPM</th><th>Added</th></tr>
  {% for s in library %}
  <tr><td>{{s.title}}</td><td><span class="badge">{{s.genre}}</span></td><td>{{s.mood}}</td><td>{{s.key}} {{s.mode}}</td><td><span class="badge">{{s.camelot}}</span></td><td class="n">{{s.bpm}}</td><td>{{s.added}}</td></tr>
  {% endfor %}
 </table>
 {% else %}<p class="hint">Empty &mdash; ingest a song to start building the library.</p>{% endif %}
</div>

{% if pairs %}
<div class="card lib">
 <div class="sec" style="margin-top:0">Harmonic &amp; BPM matches ({{pairs|length}}) &mdash; Rekordbox-style</div>
 <p class="hint">Library pairs that mix (Camelot wheel + tempo). Generate a NEW track that sits between them.</p>
 <table>
  <tr><th>Track A</th><th>Track B</th><th>Key match</th><th>Tempo</th><th></th></tr>
  {% for p in pairs %}
  <tr>
   <td>{{p.a.title}} <span class="badge">{{p.camelot_a}}</span></td>
   <td>{{p.b.title}} <span class="badge">{{p.camelot_b}}</span></td>
   <td>{{p.key_reason}}</td>
   <td>{{p.bpm_reason}}</td>
   <td><form method="post" action="/generate" style="padding:0;background:none;margin:0">
    <input type="hidden" name="genre" value="pop"><input type="hidden" name="mood" value="{{p.blend.mood}}">
    <input type="hidden" name="root" value="{{p.blend.root}}"><input type="hidden" name="bpm" value="{{p.blend.bpm}}">
    <input type="hidden" name="chords" value="on"><input type="hidden" name="bars" value="8"><input type="hidden" name="candidates" value="6">
    <button style="margin:0;width:auto;padding:.4rem .7rem;font-size:.8rem">Generate matched &rarr; {{p.blend.root}} {{p.blend.mood}} {{p.blend.bpm}}bpm</button>
   </form></td>
  </tr>
  {% endfor %}
 </table>
</div>{% endif %}
</div>

<script>
function tab(id,el){
 document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
 document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
 document.getElementById(id).classList.add('active');
 el.classList.add('active');
}
{% if ingest_result or ingest_error %}tab('ing', document.querySelectorAll('.tab')[1]);{% endif %}
</script>
</div></body></html>
"""


def _form(src, posted=False):
    g = src.get
    chords = "on" if g("chords") == "on" else ("" if posted else "on")
    return {"genre": g("genre", "lofi"), "mood": g("mood", "calm"),
            "root": g("root", "C"), "bars": g("bars", "8"), "bpm": g("bpm", ""),
            "seed": g("seed", ""), "candidates": g("candidates", "8"),
            "melody_inst": g("melody_inst", "E-Piano"),
            "chords_inst": g("chords_inst", "E-Piano"),
            "bass_inst": g("bass_inst", "Square synth"),
            "drums": g("drums", "auto"), "tone": g("tone", "bright"),
            "structure": g("structure", "loop"), "generator": g("generator", "markov"),
            "fx": g("fx", "none"), "chords": chords}


def _recent(n=8):
    out = CFG["output_dir"]
    mp3s = [f for f in os.listdir(out) if f.endswith(".mp3")] if os.path.isdir(out) else []
    mp3s.sort(key=lambda f: os.path.getmtime(os.path.join(out, f)), reverse=True)
    return mp3s[:n]


def _clf_status():
    m = classify.load_model(CFG["model_path"])
    if not m:
        return None
    return {"n": m.get("n"), "accuracy": m.get("accuracy"), "classes": m.get("classes", [])}


def render_pianoroll(arr):
    """Inline SVG piano-roll of the melodic notes (melody/chords/bass)."""
    melodic = [t for t in arr["tracks"] if t["instrument"]["kind"] != "audiofileprocessor"]
    notes = [(n, t["name"]) for t in melodic for n in t["notes"]]
    if not notes:
        return ""
    total = max(1, arr["bars"] * 192)
    keys = [n["key"] for n, _ in notes]
    lo, hi = min(keys), max(keys)
    span = max(1, hi - lo)
    W, H, pad = 760, 190, 6
    def x(p): return pad + (p / total) * (W - 2 * pad)
    def wdt(l): return max(2.0, (l / total) * (W - 2 * pad))
    def y(k): return pad + (1 - (k - lo) / span) * (H - 2 * pad - 8)
    colors = {"Melody": "#8b7cf6", "Chords": "#3a3a55", "Bass": "#4ade80"}
    grid = "".join(
        f'<line x1="{x(b*192):.1f}" y1="0" x2="{x(b*192):.1f}" y2="{H}" stroke="#24243250" stroke-width="1"/>'
        for b in range(arr["bars"] + 1))
    rects = "".join(
        f'<rect x="{x(n["pos"]):.1f}" y="{y(n["key"]):.1f}" width="{wdt(n["len"]):.1f}" '
        f'height="6" rx="2" fill="{colors.get(name, "#888")}" opacity="0.92"/>'
        for n, name in notes)
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" '
            f'style="background:#0b0b11;border-radius:8px;border:1px solid #2a2a3a">'
            f'{grid}{rects}</svg>')


def _library_matches():
    lib = ingest.read_library(CFG["library_csv"])
    for s in lib:
        s["camelot"] = match.camelot(s.get("key", ""), s.get("mode", ""))
    pairs = match.find_pairs(lib)
    for p in pairs:
        p["blend"] = match.blend(p["a"], p["b"])
    return lib, pairs


def _ctx(**kw):
    library, pairs = _library_matches()
    base = dict(library=library, pairs=pairs,
                moods=list(generate.MOOD), roots=list(generate.NOTE_BASE),
                instruments=list(generate.INSTRUMENTS), drum_choices=generate.DRUM_CHOICES,
                genres=list(generate.GENRES), tones=generate.TONE_CHOICES,
                structures=generate.STRUCTURE_CHOICES, generators=generate.GENERATOR_CHOICES,
                fxs=generate.FX_CHOICES,
                recent=_recent(), clf=_clf_status(),
                nn={"available": neural.available(), "trained": neural.trained()},
                stems_available=stems.available(),
                result=None, error=None, audio=None, rec=None, pianoroll=None,
                ingest_result=None, ingest_error=None,
                train_result=None, train_error=None,
                nn_result=None, nn_error=None,
                stems_result=None, stems_error=None)
    base.update(kw)
    return base


@app.route("/")
def index():
    return render_template_string(PAGE, **_ctx(f=_form({})))


@app.route("/generate", methods=["POST"])
def gen():
    f = _form(request.form, posted=True)
    try:
        opts = {"melody_inst": f["melody_inst"], "chords_inst": f["chords_inst"],
                "bass_inst": f["bass_inst"], "drums": f["drums"], "tone": f["tone"],
                "chords": f["chords"] == "on"}
        res = pipeline.run(
            genre=f["genre"] or "lofi", mood=f["mood"], root=f["root"],
            bars=int(f["bars"] or 8), bpm=int(f["bpm"]) if f["bpm"] else None,
            seed=int(f["seed"]) if f["seed"] else None,
            n_candidates=int(f["candidates"] or 8), opts=opts)
        audio = os.path.basename(res["mp3"] or res["wav"]) if (res["mp3"] or res["wav"]) else None
        rec = recommend.recommend(res["arrangement"])
        pr = render_pianoroll(res["arrangement"])
        return render_template_string(PAGE, **_ctx(f=f, result=res, audio=audio,
                                                   rec=rec, pianoroll=pr))
    except Exception:
        return render_template_string(PAGE, **_ctx(f=f, error=traceback.format_exc()))


@app.route("/ingest", methods=["POST"])
def ing():
    f = _form({})
    file = request.files.get("song")
    if not file or not file.filename:
        return render_template_string(PAGE, **_ctx(f=f, ingest_error="No file selected."))
    name = secure_filename(file.filename)
    if os.path.splitext(name)[1].lower() not in AUDIO_EXT:
        return render_template_string(PAGE, **_ctx(f=f, ingest_error=f"Unsupported type: {name}"))
    dest = os.path.join(CFG["uploads_dir"], name)
    file.save(dest)
    try:
        analysis = ingest.analyze(dest)
        row = ingest.add_to_library(analysis, dest, CFG["library_csv"])
        return render_template_string(PAGE, **_ctx(f=f, ingest_result=row))
    except Exception:
        return render_template_string(PAGE, **_ctx(f=f, ingest_error=traceback.format_exc()))


@app.route("/train", methods=["POST"])
def train():
    try:
        res = classify.train_model(CFG["model_path"])
        return render_template_string(PAGE, **_ctx(f=_form({}), train_result=res))
    except Exception:
        return render_template_string(PAGE, **_ctx(f=_form({}), train_error=traceback.format_exc()))


@app.route("/train_neural", methods=["POST"])
def train_neural():
    try:
        res = neural.train_model(CFG["corpus_dir"])
        return render_template_string(PAGE, **_ctx(f=_form({}), nn_result=res))
    except Exception as e:
        return render_template_string(PAGE, **_ctx(f=_form({}), nn_error=str(e)))


@app.route("/stems", methods=["POST"])
def stems_route():
    f = _form({})
    file = request.files.get("song")
    if not file or not file.filename:
        return render_template_string(PAGE, **_ctx(f=f, stems_error="No file selected."))
    name = secure_filename(file.filename)
    if os.path.splitext(name)[1].lower() not in AUDIO_EXT:
        return render_template_string(PAGE, **_ctx(f=f, stems_error=f"Unsupported type: {name}"))
    dest = os.path.join(CFG["uploads_dir"], name)
    file.save(dest)
    try:
        res = stems.separate(dest, os.path.join(CFG["output_dir"], "stems"))
        out_dir = CFG["output_dir"]
        sr = {"name": res["name"],
              "inst_url": os.path.relpath(res["instrumental"], out_dir).replace("\\", "/"),
              "voc_url": os.path.relpath(res["vocals"], out_dir).replace("\\", "/")}
        try:  # feed the karaoke instrumental back into the library
            ingest.add_to_library(ingest.analyze(res["instrumental"]),
                                  res["instrumental"], CFG["library_csv"])
        except Exception:
            pass
        return render_template_string(PAGE, **_ctx(f=f, stems_result=sr))
    except Exception:
        return render_template_string(PAGE, **_ctx(f=f, stems_error=traceback.format_exc()))


@app.route("/out/<path:fn>")
def out(fn):
    return send_from_directory(CFG["output_dir"], fn)


if __name__ == "__main__":
    print("Cadenza UI -> http://127.0.0.1:5005")
    app.run(port=5005, debug=False)
