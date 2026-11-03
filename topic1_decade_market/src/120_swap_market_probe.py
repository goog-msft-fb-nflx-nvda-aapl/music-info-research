"""T-4j market read-out of the lyric route: the vocal market probe (dataset-B vocal probe, MERT layer and C from probe_B.json 'voc_norm', StandardScaler + logistic regression fit on dataset-B TRAIN vocal embeddings; validation accuracy reproduced) applied to the separated generated vocal of the stem-swap route (ace_swap_voc/<id>_<cond>_s0.wav, 12 pilot clips: US/UK source, English; conditions orig = English control, es -> Spain, pt -> Brazil, de -> Germany, it -> Italy). Reported: P(target market | vocal of condition c) for c in es, pt, de, it; its difference to the same market's probability under the English control (orig) of the same clip (paired over 12 clips, song-bootstrap 95% CI, 2000 resamples) ; top-1 hit of the target market; and the same quantities for the ORIGINAL vocal stems (reference).
usage (from ~/era_market, main venv, one GPU): python src/120_swap_market_probe.py -> results/swap_market_probe.{json,txt}"""
import os, sys, json, numpy as np, soundfile as sf, torch, pyloudnorm as pyln
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
MK = LABELS["B"]; TG = {"es": "Spain", "pt": "Brazil", "de": "Germany", "it": "Italy"}; rng = np.random.default_rng(0)
pj = json.load(open(f"{ROOT}/results/probe_B.json"))["conds"]["voc_norm"]; L, C = pj["layer"], pj["C"]; rows = manifest("B"); y = np.array([MK.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
E_ = np.load(f"{ROOT}/emb/B_voc_norm.npy"); clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E_[tr, L].astype(np.float32), y[tr]); acc0 = float((clf.predict(E_[va, L].astype(np.float32)) == y[va]).mean()); assert abs(acc0 - pj["val_acc"]) < 1e-6
proc, mert = load_mert(); meter = pyln.Meter(SR)
def norm(w): l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) or np.abs(w).max() < 1e-4 else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def emb(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[L] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[L].mean(1)).squeeze(0).float().cpu().numpy()
T = json.load(open(f"{ROOT}/results/lyrics_route_text.json")); ids = list(T); P = {}
for s in ids:
    P[(s, "original")] = clf.predict_proba(emb(norm(np.load(f"{ROOT}/stems/B/{s}.npz")["vocals"].astype(np.float32)))[None].astype(np.float32))[0]
    for c in ("orig", "es", "pt", "de", "it"): P[(s, c)] = clf.predict_proba(emb(norm(sf.read(f"{ROOT}/ace_swap_voc/{s}_{c}_s0.wav", dtype="float32")[0]))[None].astype(np.float32))[0]
out = {"val_acc_vocal_probe": acc0, "layer": L, "C": C, "cells": {}}; txt = [f"vocal market probe (MERT layer {L}, C {C}) validation accuracy {acc0:.3f}; 12 pilot clips (US/UK source)"]
for c, m in TG.items():
    k = MK.index(m); pc = np.array([P[(s, c)][k] for s in ids]); p0 = np.array([P[(s, "orig")][k] for s in ids]); po = np.array([P[(s, "original")][k] for s in ids]); d = pc - p0; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    hit = float(np.mean([np.argmax(P[(s, c)]) == k for s in ids])); hit0 = float(np.mean([np.argmax(P[(s, "orig")]) == k for s in ids]))
    out["cells"][c] = {"target": m, "P_target": float(pc.mean()), "P_target_english_control": float(p0.mean()), "P_target_original_vocal": float(po.mean()), "delta_vs_control": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "top1_hit": hit, "top1_hit_control": hit0}
    txt.append(f"{c} -> {m}: P(target) {pc.mean():.3f} (English control {p0.mean():.3f}, original vocal {po.mean():.3f}); delta vs control {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]; target top-1 {hit:.2f} (control {hit0:.2f})")
json.dump(out, open(f"{ROOT}/results/swap_market_probe.json", "w"), indent=1); open(f"{ROOT}/results/swap_market_probe.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
