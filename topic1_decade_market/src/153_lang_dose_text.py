"""Language-dose text: code-switched lyrics for the 16 dataset-A evaluation songs. For target language L in (es, de) and fraction f in (0.25, 0.5, 0.75) the lyrics are the first f of the translated text (words) followed by the last 1-f of the source text; conditions are named L25, L50, L75 (results/lyrics_lang_dose_A.json, same keys as lyrics_scale_text_A.json). Full-language (f = 1) and source (f = 0) conditions are the existing cover-route outputs. usage (from ~/era_market): python src/153_lang_dose_text.py"""
import json, os
H = os.path.expanduser("~/era_market"); T = json.load(open(f"{H}/results/lyrics_scale_text_A.json")); out = {}
for sid, d in T.items():
    src = d.get("text_src", d.get("text_en", "")).split(); o = {k: d[k] for k in ("market", "src_lang", "text_src") if k in d}
    for L in ("es", "de"):
        if L not in d: continue
        tr = d[L].split()
        for f in (25, 50, 75): o[f"{L}{f}"] = " ".join(tr[:int(len(tr) * f / 100)] + src[int(len(src) * f / 100):])
    out[sid] = o
json.dump(out, open(f"{H}/results/lyrics_lang_dose_A.json", "w"), ensure_ascii=False); print(len(out), "songs")
