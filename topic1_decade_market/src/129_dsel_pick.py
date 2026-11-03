"""T-4p stage 2: select the style words per market. Discogs-EffNet tags (venv_ess, CPU) of every candidate clip of 128_dsel_gen.py; target score of market m = mean activation over the five highest-lift styles of (m, 1980s) present among the 400 tags (as 115_market_style_eval.py); for each candidate word: mean over the 6 TRAIN clips of score(word prompt) - score(neutral prompt) of the same clip. The three words with the highest mean form the 'dt_' descriptor of the market (written to results/dsel_selected.json together with all candidate scores). Selection uses TRAIN clips only; the 24 evaluation clips are not used.
usage (from ~/era_market with venv_ess): GENDIR=gen_dsel python src/129_dsel_pick.py"""
import os, sys, json, re, numpy as np, soundfile as sf, librosa
sys.path.insert(0, "src")
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from common import ROOT, SR, manifest
M = os.path.join(ROOT, "ess_models"); emb_m = TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(M, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"); head = TensorflowPredict2D(graphFilename=os.path.join(M, "genre_discogs400-discogs-effnet-1.pb"), input="serving_default_model_Placeholder", output="PartitionedCall:0")
tags = json.load(open(os.path.join(M, "genre_discogs400-discogs-effnet-1.json")))["classes"]; MS = json.load(open(f"{ROOT}/results/market_styles.json")); MK = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
tg = {m: [tags.index(x["style"]) for x in MS[f"{m}|1980s"]["top_lift"] if x["style"] in tags][:5] for m in MK}
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen_dsel")); CAND = json.load(open(os.path.join(G, "candidates.json"))); rows = manifest("B"); pick = [sorted([r for r in rows if r["split"] == "train" and r["label"] == m], key=lambda r: r["sample_id"])[0]["sample_id"] for m in MK]
def tagp(y): return head(emb_m(librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=SR, target_sr=16000))).mean(0)
neu = {s: tagp(sf.read(os.path.join(G, f"{s}_neutral_neutral_s0_acc.wav"), dtype="float32")[0]) for s in pick}; out = {"selected": {}, "scores": {}}
for m in MK:
    sc = {}
    for w in CAND[m]:
        d = [tagp(sf.read(os.path.join(G, f"{s}_{m}_{re.sub(r'[^A-Za-z0-9]', '', w)}_s0_acc.wav"), dtype="float32")[0])[tg[m]].mean() - neu[s][tg[m]].mean() for s in pick]; sc[w] = float(np.mean(d))
    out["scores"][m] = sc; out["selected"][m] = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:3]]; print(m, sorted(sc.items(), key=lambda kv: -kv[1])[:5], flush=True)
json.dump(out, open(f"{ROOT}/results/dsel_selected.json", "w"), indent=1, ensure_ascii=False)
