"""Builds docs/era_market/index.html (project page of topic 2) from results/six_decades.json and the numbers of the report. usage (from the topic folder): python src/92_make_page.py ; the audio section uses the 32 base-model demo clips (docs/era_market/audio) and the adapter-system clips of src/93_build_demo_clips.py (docs/era_market/audio_lora)."""
import json, re
P = "../docs/era_market/index.html"
songs = json.load(open("../docs/era_market/audio_index.json"))
sd = json.load(open("results/six_decades.json")); D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
keep = ["base_ge7", "lora_ge7", "lora_ge10", "lora_ge15", "lora_ge20"]
six = {enc: {k: {"E": [sd[k][f"{enc}_{d}"]["expected_decade"] for d in D], "dP": [sd[k][f"{enc}_{d}"]["delta"] for d in D]} for k in keep} for enc in ("mert", "muq")}
data = {"six": six, "songs": songs, "lora": json.load(open("../docs/era_market/audio_lora_index.json"))}
html = open("src/page_template.html").read().replace("/*DATA*/", "const D=" + json.dumps(data) + ";")
open(P, "w").write(html); print("wrote", P, len(html))
