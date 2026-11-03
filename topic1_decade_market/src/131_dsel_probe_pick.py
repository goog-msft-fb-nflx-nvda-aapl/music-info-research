"""T-4q stage 2 (instruments swapped relative to 129_dsel_pick.py): select the style words per market with the codec-matched MARKET PROBES instead of the Discogs tagger. For every candidate clip of 128_dsel_gen.py (6 TRAIN clips, one per market, candidate words per market) the average of the MERT and MuQ probe probabilities of the market (probes as 37_eval_market.py: logistic regression C .01 on EnCodec-resynthesized dataset-B TRAIN accompaniment, layers from probe_codec_{mert,muq}_B.json) minus the neutral clip's; the three best words per market -> results/dsel_selected_probe.json. The source clips of the selection are TRAIN clips (their own labels are used in the probes' training set; the generated audio is new).
usage (from ~/era_market, main venv, one GPU): GENDIR=gen_dsel python src/131_dsel_probe_pick.py"""
import os, sys, json, re, numpy as np, soundfile as sf
sys.path.insert(0, "src")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
from eraprobe import embed_waves, predict
MK = LABELS["B"]; rows = manifest("B"); y = np.array([MK.index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.where(np.array([r["split"] for r in rows]) == "train")[0]
layer = {k: json.load(open(os.path.join(ROOT, "results", f"probe_codec_{k}_B.json")))["conds"]["acccodec_norm"]["layer"] for k in ("mert", "muq")}; probes = {}
for k, pre in (("mert", ""), ("muq", "muq_")):
    E = np.load(os.path.join(ROOT, "emb", f"B_{pre}acccodec_norm.npy")).astype(np.float32)[:, layer[k]]; probes[k] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(E[tr], y[tr])
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen_dsel")); CAND = json.load(open(os.path.join(G, "candidates.json"))); pick = [sorted([r for r in rows if r["split"] == "train" and r["label"] == m], key=lambda r: r["sample_id"])[0]["sample_id"] for m in MK]
clips = {}
for s in pick:
    clips[f"{s}|neutral|0"] = sf.read(os.path.join(G, f"{s}_neutral_neutral_s0_acc.wav"), dtype="float32")[0]
    for m in MK:
        for w in CAND[m]: clips[f"{s}|{m}|{w}"] = sf.read(os.path.join(G, f"{s}_{m}_{re.sub(r'[^A-Za-z0-9]', '', w)}_s0_acc.wav"), dtype="float32")[0]
P = predict(probes, embed_waves(clips, layer)); out = {"selected": {}, "scores": {}}
for m in MK:
    t = MK.index(m); sc = {}
    for w in CAND[m]: sc[w] = float(np.mean([0.5 * (P["mert"][f"{s}|{m}|{w}"][t] + P["muq"][f"{s}|{m}|{w}"][t]) - 0.5 * (P["mert"][f"{s}|neutral|0"][t] + P["muq"][f"{s}|neutral|0"][t]) for s in pick]))
    out["scores"][m] = sc; out["selected"][m] = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:3]]; print(m, sorted(sc.items(), key=lambda kv: -kv[1])[:4], flush=True)
json.dump(out, open(f"{ROOT}/results/dsel_selected_probe.json", "w"), indent=1, ensure_ascii=False)
