"""Builds the interactive 2D control plane (market x era). Click = discrete cell, drag = continuous blend of the neighbouring clips.
usage (from topic folder): python src/164_plane_page.py public   -> ../docs/era_market/plane.html (accompaniment-only clips, committed)
                           python src/164_plane_page.py internal -> data_local/plane/plane.html (vocal-containing clips, git-ignored, never published)"""
import sys, json, os
mode = sys.argv[1] if len(sys.argv) > 1 else "public"
if mode == "public":
    eras = ["a0", "a0.2", "a0.4", "a0.6", "a0.8", "a1"]
    rows = [{"label": f"era dose {e[1:] or '0'}", "tick": {"a0": "1960s", "a1": "2010s"}.get(e, "")} for e in eras]
    cols = [{"label": "original language (accompaniment only)", "short": "original"},
            {"label": "Spanish (sung; internal only)", "short": "es", "off": True},
            {"label": "German (sung; internal only)", "short": "de", "off": True}]
    cells = [[f"audio_dose/adapter/A_007e1d02fd77_{e}_s0.ogg", None, None] for e in eras]
    out, title = "../docs/era_market/plane.html", "Time Traveler: control plane (accompaniment only)"
else:
    eras = ["1960s", "1980s", "2010s"]
    rows = [{"label": e, "tick": ""} for e in eras]
    cols = [{"label": "original language", "short": "original"}, {"label": "Spanish", "short": "es"}, {"label": "German", "short": "de"}]
    cells = [[f"audio/A_007e1d02fd77_{e}_{l}.ogg" for l in ["orig", "es", "de"]] for e in eras]
    out, title = "data_local/plane/plane.html", "INTERNAL - control plane with sung vocals, do not publish"
data = {"rows": rows, "cols": cols, "cells": cells, "mode": mode}
html = open("src/plane_template.html").read().replace("__DATA__", json.dumps(data)).replace("__TITLE__", title)
open(out, "w").write(html); print("wrote", out)
