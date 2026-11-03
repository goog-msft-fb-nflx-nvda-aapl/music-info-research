"""T-9d additivity of the vocal chain and the era-axis accompaniment: for decade e in {1980s, 2010s} on the 24 evaluation songs (seeds 0-2 of the accompaniment): mix-level era probe (MERT layer 12, probe_A.json 'mix_norm'; refit on TRAIN, validation accuracy reproduced) P(decade e) in the four cells vocal in {dry, tuned chain of results/vocal_chain_tune2.json} x accompaniment in {neutral, era-axis `gen_ax_ctrlc_ge10`}; main effects (chain, accompaniment) and interaction = (chain effect with era accompaniment) - (chain effect with neutral accompaniment), song-bootstrap 95% CI (2000 resamples, seeds averaged per song). Remix = vocal (chain output RMS-matched to the dry vocal) + accompaniment, -23 LUFS.
usage (from ~/era_market, main venv, one GPU): python src/127_vocal_acc_additivity.py -> results/vocal_acc_additivity.{json,txt}"""
import os, sys, json, numpy as np, soundfile as sf, torch
sys.path.insert(0, "src")
src = open("src/126_vocal_chain_tune2.py").read(); head = src[:src.index("pj = json.load(")]; exec(compile(head, "chain_defs", "exec"))   # chain functions, DEC, rng, norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
rows = manifest("A"); y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
pm = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["mix_norm"]; Lm, Cm = pm["layer"], pm["C"]; Em = np.load(f"{ROOT}/emb/A_mix_norm.npy"); cm = make_pipeline(StandardScaler(), LogisticRegression(C=Cm, max_iter=2000)).fit(Em[tr, Lm].astype(np.float32), y[tr]); assert abs(float((cm.predict(Em[va, Lm].astype(np.float32)) == y[va]).mean()) - pm["val_acc"]) < 1e-6
proc, mert = load_mert()
def embm(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[Lm] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[Lm].mean(1)).squeeze(0).float().cpu().numpy()
TU = json.load(open(f"{ROOT}/results/vocal_chain_tune2.json"))["tuning"]; ids = [l.split(",")[0] for l in open(f"{ROOT}/gen/picked.txt").read().split("\n")]; out = {}; txt = []
def ci(d): bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
for e in ("1980s", "2010s"):
    k = DEC.index(e); p = TU[e]["best"]; C4 = {c: [] for c in ("dry_neutral", "chain_neutral", "dry_era", "chain_era")}
    for s in ids:
        v = np.load(f"{ROOT}/stems/A/{s}.npz")["vocals"].astype(np.float32); vp = chain(v.astype(np.float64), e, p).astype(np.float32); vp = vp * (np.sqrt((v ** 2).mean()) / (np.sqrt((vp ** 2).mean()) + 1e-9)); cell = {c: [] for c in C4}
        for sd in (0, 1, 2):
            for an, tag in (("neutral", "neutral"), (e, "era")):
                acc = sf.read(f"{ROOT}/gen_ax_ctrlc_ge10/{s}_{an}_s{sd}_acc.wav", dtype="float32")[0][:len(v)]
                for vn, vv in (("dry", v), ("chain", vp)): cell[f"{vn}_{tag}"].append(cm.predict_proba(embm(norm(np.clip(vv + acc, -1, 1)))[None].astype(np.float32))[0][k])
        for c in C4: C4[c].append(np.mean(cell[c]))
    A = {c: np.array(v) for c, v in C4.items()}; chain_main = (A["chain_neutral"] - A["dry_neutral"] + A["chain_era"] - A["dry_era"]) / 2; acc_main = (A["dry_era"] - A["dry_neutral"] + A["chain_era"] - A["chain_neutral"]) / 2; inter = (A["chain_era"] - A["dry_era"]) - (A["chain_neutral"] - A["dry_neutral"])
    out[e] = {"params": p, "cells": {c: float(v.mean()) for c, v in A.items()}, "chain_main": [float(chain_main.mean()), *ci(chain_main)], "accompaniment_main": [float(acc_main.mean()), *ci(acc_main)], "interaction": [float(inter.mean()), *ci(inter)]}
    txt.append(f"{e} (tuned chain {p}): P({e}) dry/neutral {A['dry_neutral'].mean():.3f}, chain/neutral {A['chain_neutral'].mean():.3f}, dry/era {A['dry_era'].mean():.3f}, chain/era {A['chain_era'].mean():.3f}; chain main effect {chain_main.mean():+.3f} [{ci(chain_main)[0]:+.3f},{ci(chain_main)[1]:+.3f}]; accompaniment main effect {acc_main.mean():+.3f} [{ci(acc_main)[0]:+.3f},{ci(acc_main)[1]:+.3f}]; interaction {inter.mean():+.3f} [{ci(inter)[0]:+.3f},{ci(inter)[1]:+.3f}]"); print(txt[-1], flush=True)
json.dump(out, open(f"{ROOT}/results/vocal_acc_additivity.json", "w"), indent=1); open(f"{ROOT}/results/vocal_acc_additivity.txt", "w").write("\n".join(txt) + "\n")
