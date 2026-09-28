"""Arrangement -> LMMS project (.mmp) XML.

Builds a valid LMMS 1.2 project from scratch. Supports three instrument kinds so
we can move past raw synth:
  * ``bitinvader``       — wavetable e-piano (bundled preset) for melody/chords
  * ``audiofileprocessor`` — plays a factory drum sample (drums/*.ogg) for beats
  * ``tripleoscillator`` — synth, used for bass

Sample paths are given RELATIVE to LMMS's factory samples dir (e.g.
``drums/bassdrum01.ogg``), so they resolve from the Z: install and are
independent of LMMS's user working-dir setting.

Note timing is in LMMS ticks (192 per 4/4 bar). Keys are LMMS key numbers.
Gotchas (do not "fix"): <pattern> must be a sibling of <instrumenttrack>;
tripleoscillator modalgo must be 2 (SignalMix) or the render is silent.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET

TICKS_PER_BAR = 192

# BitInvader e-piano wavetable, lifted verbatim from the bundled epiano.xpf.
_BITINVADER = {
    "sampleLength": "30", "version": "0.1", "normalize": "1", "interpolation": "1",
    "sampleShape": ("AAAAAPMEtT56glo/eoJaP/IEtT72BLW+eoJav3iCWr/vBLW+"
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    "AAAAAAAAAAAAAAAA"),
}

_AUDIOFILE = {"eframe": "1", "stutter": "0", "amp": "100", "interp": "1",
              "reversed": "0", "looped": "0", "sframe": "0", "lframe": "0"}

_MIDIPORT = {
    "inputchannel": "0", "outputchannel": "1", "fixedinputvelocity": "-1",
    "fixedoutputvelocity": "-1", "outputprogram": "1", "fixedoutputnote": "-1",
    "basevelocity": "127", "readable": "0", "writable": "0",
    "inputcontroller": "0", "outputcontroller": "0",
}

_ELTARGET = {
    "rel": "0.1", "dec": "0.5", "lspd": "0.1", "pdel": "0", "att": "0",
    "hold": "0.5", "syncmode": "0", "amt": "0", "latt": "0", "lpdel": "0",
    "lamt": "0", "userwavefile": "", "lspd_numerator": "4", "lshp": "0",
    "sustain": "0.5", "x100": "0", "lspd_denominator": "4", "ctlenvamt": "0",
}


def _osc(wavetype: int) -> dict:
    return {
        "wavetype0": str(wavetype), "wavetype1": str(wavetype), "wavetype2": "0",
        "vol0": "33", "vol1": "33", "vol2": "0",
        "coarse0": "0", "coarse1": "-12", "coarse2": "0",
        "finel0": "0", "finel1": "0", "finel2": "0",
        "finer0": "0", "finer1": "0", "finer2": "0",
        "pan0": "0", "pan1": "0", "pan2": "0",
        "phoffset0": "0", "phoffset1": "0", "phoffset2": "0",
        "stphdetun0": "0", "stphdetun1": "0", "stphdetun2": "0",
        "modalgo1": "2", "modalgo2": "2", "modalgo3": "2",  # 2 = SignalMix
        "useWaveTable0": "1", "useWaveTable1": "1", "useWaveTable2": "1",
    }


def _make_instrument(it: ET.Element, spec: dict) -> None:
    kind = spec["kind"]
    instr = ET.SubElement(it, "instrument", name=kind)
    if kind == "tripleoscillator":
        osc = _osc(spec.get("wavetype", 3))
        osc.update(spec.get("osc", {}))  # per-instrument overrides (e.g. Supersaw)
        ET.SubElement(instr, "tripleoscillator", osc)
    elif kind == "bitinvader":
        ET.SubElement(instr, "bitinvader", _BITINVADER)
    elif kind == "audiofileprocessor":
        ET.SubElement(instr, "audiofileprocessor", dict(_AUDIOFILE, src=spec["src"]))
    else:
        raise ValueError(f"unknown instrument kind: {kind}")


def _instr_track(track: dict, pat_len: int) -> ET.Element:
    t = ET.Element("track", type="0", muted="0", solo="0", name=track["name"])
    it = ET.SubElement(t, "instrumenttrack", {
        "pan": "0", "fxch": "0", "pitchrange": "1",
        "basenote": str(track.get("basenote", 57)),
        "vol": str(track.get("vol", 100)), "pitch": "0", "usemasterpitch": "1"})
    _make_instrument(it, track["instrument"])
    eld = ET.SubElement(it, "eldata", {"ftype": "0", "fcut": str(track.get("fcut", 14000)),
                                       "fwet": "0", "fres": "0.5"})
    for tag in ("elvol", "elcut", "elres"):
        ET.SubElement(eld, tag, _ELTARGET)
    ET.SubElement(it, "chordcreator", {"chord": "0", "chord-enabled": "0",
                                       "chordrange": "1"})
    ET.SubElement(it, "arpeggiator", {
        "syncmode": "0", "arptime_numerator": "4", "arptime_denominator": "4",
        "arpmode": "0", "arprange": "1", "arptime": "100", "arp": "0",
        "arp-enabled": "0", "arpgate": "100", "arpdir": "0"})
    midiport = dict(_MIDIPORT)
    if track.get("midi_in"):
        midiport["readable"] = "1"  # accept live MIDI (play this track from a controller)
    ET.SubElement(it, "midiport", midiport)
    ET.SubElement(it, "fxchain", {"enabled": "0", "numofeffects": "0"})
    # pattern is a SIBLING of <instrumenttrack>, child of <track>
    pat = ET.SubElement(t, "pattern", {
        "pos": "0", "len": str(pat_len), "name": track["name"],
        "type": "1", "muted": "0", "steps": "16"})
    for n in track["notes"]:
        ET.SubElement(pat, "note", {
            "pan": "0", "key": str(n["key"]), "vol": str(n.get("vol", 100)),
            "pos": str(n["pos"]), "len": str(n["len"])})
    return t


def write_project(arr: dict, out_path: str, mastervol: int = 70) -> str:
    root = ET.Element("lmms-project", {
        "type": "song", "creator": "Cadenza", "creatorversion": "1.0", "version": "1.0"})
    ET.SubElement(root, "head", {
        "bpm": str(arr["bpm"]), "mastervol": str(mastervol), "masterpitch": "0",
        "timesig_numerator": "4", "timesig_denominator": "4"})
    song = ET.SubElement(root, "song")
    tc = ET.SubElement(song, "trackcontainer", {
        "type": "song", "x": "5", "y": "5", "width": "700", "height": "400",
        "maximized": "0", "visible": "1"})
    pat_len = arr["bars"] * TICKS_PER_BAR
    for track in arr["tracks"]:
        tc.append(_instr_track(track, pat_len))
    fx = ET.SubElement(song, "fxmixer")
    ch = ET.SubElement(fx, "fxchannel", {"num": "0", "name": "Master",
                                         "volume": "1", "muted": "0"})
    ET.SubElement(ch, "fxchain", {"enabled": "0", "numofeffects": "0"})
    ET.SubElement(song, "timeline", {"lp0pos": "0", "lpstate": "0", "lp1pos": "192"})
    ET.SubElement(song, "controllers")

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<!DOCTYPE lmms-project>\n' + body)
    return out_path
