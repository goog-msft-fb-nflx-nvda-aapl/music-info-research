"""T-4e read-out: Discogs-EffNet style tags (400 Discogs styles; essentia models in ~/era_market/ess_models; venv_ess, CPU) of the generated accompaniment (GENDIR/<id>_<cond>_s<seed>_acc.wav). Target score of market m = mean tag activation over the five styles with the highest lift in (m, 1980s) (results/market_styles.json top_lift, styles present among the 400 tags) ; Delta = score(clip with prompt of m) - score(neutral clip of the same song and seed); per condition family hw_ (hand-written prompts), ds_ (metadata-derived, highest lift) and dk_ (metadata-derived, largest P x log lift), paired over songs and seeds, mean with song-bootstrap 95% CI (2000 resamples) and Wilcoxon p; all 24 songs (4 per source market) are used for every target market. Also reports the mean target score of the real accompaniment of the clips of market m (reference).
usage (from ~/era_market with venv_ess): GENDIR=gen_mstyle python src/115_market_style_eval.py -> results/market_style_eval.{json,txt}"""
import os, sys, json, numpy as np, soundfile as sf, librosa
from scipy.stats import wilcoxon
sys.path.insert(0, "src")
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from common import ROOT, SR
M = os.path.join(ROOT, "ess_models"); emb_m = TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(M, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"); head = TensorflowPredict2D(graphFilename=os.path.join(M, "genre_discogs400-discogs-effnet-1.pb"), input="serving_default_model_Placeholder", output="PartitionedCall:0")
tags = json.load(open(os.path.join(M, "genre_discogs400-discogs-effnet-1.json")))["classes"]; MS = json.load(open(f"{ROOT}/results/market_styles.json")); MK = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
tg = {}; cov = {}
for m in MK:
    want = [x["style"] for x in MS[f"{m}|1980s"]["top_lift"]]; have = [w for w in want if w in tags][:5]; tg[m] = [tags.index(w) for w in have]; cov[m] = {"wanted": want, "in_tagger": have}
def tagp(y): return head(emb_m(librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=SR, target_sr=16000))).mean(0)
GD = os.environ.get("GENDIR", "gen_mstyle"); G = os.path.join(ROOT, GD); ids = [l.split(",") for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; seeds = [0, 1]; rng = np.random.default_rng(0)
T = {}
for sid, lab in ids:
    for cond in ["neutral"] + [f"{f}_{m}" for f in ("hw", "ds", "dk") for m in MK]:
        for sd in seeds:
            fp = os.path.join(G, f"{sid}_{cond}_s{sd}_acc.wav")
            if os.path.exists(fp): T[(sid, cond, sd)] = tagp(sf.read(fp, dtype="float32")[0])
    T[(sid, "real", 0)] = tagp(np.load(os.path.join(ROOT, "stems", "B", sid + ".npz"))["accomp"]); print(sid, flush=True)
out = {"coverage": cov, "cells": {}}; txt = []
for fam in ("hw", "ds", "dk"):
    for m in MK:
        d = []
        for sid, lab in ids:
            for sd in seeds:
                a, b = T.get((sid, f"{fam}_{m}", sd)), T.get((sid, "neutral", sd))
                if a is not None and b is not None: d.append(a[tg[m]].mean() - b[tg[m]].mean())
        d = np.array(d); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; real = np.mean([T[(sid, "real", 0)][tg[m]].mean() for sid, lab in ids if lab == m]); neu = np.mean([T[(sid, "neutral", sd)][tg[m]].mean() for sid, lab in ids for sd in seeds if (sid, "neutral", sd) in T])
        out["cells"][f"{fam}_{m}"] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else None, "n": int(len(d)), "neutral_score": float(neu), "real_clips_of_market_score": float(real)}
        txt.append(f"{fam}_{m}: delta target-style activation {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}] n={len(d)} (neutral {neu:.4f}; real accompaniment of market-{m} clips {real:.4f}); styles {cov[m]['in_tagger']}")
json.dump(out, open(f"{ROOT}/results/market_style_eval.json", "w"), indent=1); open(f"{ROOT}/results/market_style_eval.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
