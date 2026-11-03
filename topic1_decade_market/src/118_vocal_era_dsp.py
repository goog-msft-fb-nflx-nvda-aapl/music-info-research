"""T-9: era-specific vocal processing. The ORIGINAL vocal stem (HTDemucs) of the 24 evaluation songs (gen/picked.txt, dataset A validation) is processed with a parametrized DSP chain per target decade (numpy/scipy only; synthetic reverb impulse response with fixed seed): 1960s = band-limit 120 Hz-7 kHz, tape-style tanh saturation, compression, short plate reverb; 1980s = high-shelf brightening, chorus, gated reverb (IR cut at 0.35 s), compression; 2010s = 90 Hz high-pass, heavy compression + soft limiting, brightening, tight reverb. Read-out: the dataset-A vocal era probe (MERT, layer and C from probe_A.json 'voc_norm', StandardScaler + logistic regression fit on TRAIN vocal embeddings emb/A_voc_norm.npy; validation accuracy must reproduce the json) applied to the loudness-normalized (-23 LUFS) processed vocal: Delta_e = P(decade e | vocal processed with chain e) - P(decade e | dry vocal), per song, mean with song-bootstrap 95% CI (2000 resamples) and Wilcoxon p; all chains on all songs (off-diagonal cells show specificity).
Second stage (MIX=1): mix-level probe (probe_A.json 'mix_norm', emb/A_mix_norm.npy) on remix = era-axis accompaniment (gen_ax_ctrlc_ge10, seeds 0-2) + vocal, dry vs era-processed vocal (RMS-matched to the dry vocal), both -23 LUFS: P(decade e) difference.
usage (from ~/era_market, main venv, one GPU): python src/118_vocal_era_dsp.py -> results/vocal_era_dsp.{json,txt}; processed vocals are written to vocal_dsp/<id>_<chain>.wav (24 kHz)"""
import os, sys, json, numpy as np, soundfile as sf, torch, pyloudnorm as pyln
from scipy import signal
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
DEC = LABELS["A"]; rng = np.random.default_rng(0)
def bp(x, lo, hi): sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos"); return signal.sosfilt(sos, x)
def hp(x, f): sos = signal.butter(2, f, "highpass", fs=SR, output="sos"); return signal.sosfilt(sos, x)
def sat(x, d): return np.tanh(d * x) / np.tanh(d)
def comp(x, thr_db, ratio, tc=0.02):
    a = np.exp(-1.0 / (tc * SR)); env = signal.lfilter([1 - a], [1, -a], np.abs(x)) + 1e-8; thr = 10 ** (thr_db / 20); g = np.where(env > thr, (env / thr) ** (1.0 / ratio - 1.0), 1.0); y = x * g
    return y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
def reverb(x, rt60, wet, gate=None, seed=0):
    r = np.random.default_rng(seed); n = int(rt60 * SR); ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    if gate: ir[int(gate * SR):] = 0
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9; y = signal.fftconvolve(x, ir)[:len(x)]; return (1 - wet) * x + wet * y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
def chorus(x, depth_ms, wet): t = np.arange(len(x)); d = (0.020 * SR) + depth_ms * 1e-3 * SR * np.sin(2 * np.pi * 0.4 * t / SR); idx = np.clip(t - d, 0, len(x) - 1); y = np.interp(idx, t, x); return (1 - wet) * x + wet * y
def chain(x, e):
    if e == "1960s": return reverb(comp(sat(bp(x, 120, 7000), 2.0), -24, 3.0), 0.5, 0.12)
    if e == "1980s": y = x + 0.6 * hp(x, 4000); return comp(reverb(chorus(y, 4, 0.25), 1.0, 0.25, gate=0.35), -26, 3.0)
    if e == "2010s": y = hp(x, 90); y = comp(y, -30, 6.0); y = np.tanh(1.5 * y) / np.tanh(1.5); y = y + 0.4 * hp(y, 6000); return reverb(y, 0.25, 0.08)
    return x
meter = pyln.Meter(SR)
def norm(w): l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) or np.abs(w).max() < 1e-4 else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
pj = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["voc_norm"]; L, C = pj["layer"], pj["C"]; rows = manifest("A"); y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
E_ = np.load(f"{ROOT}/emb/A_voc_norm.npy"); clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E_[tr, L].astype(np.float32), y[tr]); acc0 = float((clf.predict(E_[va, L].astype(np.float32)) == y[va]).mean()); assert abs(acc0 - pj["val_acc"]) < 1e-6, (acc0, pj["val_acc"])
proc, mert = load_mert()
def emb(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[L] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[L].mean(1)).squeeze(0).float().cpu().numpy()
ids = [l.split(",")[0] for l in open(f"{ROOT}/gen/picked.txt").read().split("\n")]; os.makedirs(f"{ROOT}/vocal_dsp", exist_ok=True); CH = ["dry", "1960s", "1980s", "2010s"]; P = {}
for s in ids:
    v = np.load(f"{ROOT}/stems/A/{s}.npz")["vocals"].astype(np.float32)
    for c in CH:
        w = v if c == "dry" else chain(v.astype(np.float64), c).astype(np.float32); w = norm(w)
        if c != "dry": sf.write(f"{ROOT}/vocal_dsp/{s}_{c}.wav", w, SR)
        P[(s, c)] = clf.predict_proba(emb(w)[None].astype(np.float32))[0]
    print(s, flush=True)
out = {"val_acc_dry_vocal_probe": acc0, "layer": L, "C": C, "cells": {}}; txt = [f"vocal era probe (MERT layer {L}, C {C}) validation accuracy {acc0:.3f}"]
for c in CH[1:]:
    for t in CH[1:]:
        d = np.array([P[(s, c)][DEC.index(t)] - P[(s, "dry")][DEC.index(t)] for s in ids]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out["cells"][f"chain_{c}|target_{t}"] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else None}
        txt.append(f"chain {c} -> P({t}) minus dry: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
if os.environ.get("MIX") == "1":   # second stage: mix-level probe on era-axis accompaniment (gen_ax_ctrlc_ge10, seeds 0-2) + dry or era-processed vocal
    pm = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["mix_norm"]; Lm, Cm = pm["layer"], pm["C"]; Em = np.load(f"{ROOT}/emb/A_mix_norm.npy"); cm = make_pipeline(StandardScaler(), LogisticRegression(C=Cm, max_iter=2000)).fit(Em[tr, Lm].astype(np.float32), y[tr]); assert abs(float((cm.predict(Em[va, Lm].astype(np.float32)) == y[va]).mean()) - pm["val_acc"]) < 1e-6
    def embm(w):
        inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
        with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
        m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
        return ((o.hidden_states[Lm] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[Lm].mean(1)).squeeze(0).float().cpu().numpy()
    out["mix_level"] = {"layer": Lm, "C": Cm, "val_acc_mix_probe": pm["val_acc"], "cells": {}}; txt.append(f"mix-level probe (MERT layer {Lm}, C {Cm}) validation accuracy {pm['val_acc']:.3f}; era-axis accompaniment gen_ax_ctrlc_ge10 seeds 0-2")
    for e in ("1960s", "1980s", "2010s"):
        D = []
        for s_ in ids:
            v = np.load(f"{ROOT}/stems/A/{s_}.npz")["vocals"].astype(np.float32); vp = chain(v.astype(np.float64), e).astype(np.float32); vp = vp * (np.sqrt((v ** 2).mean()) / (np.sqrt((vp ** 2).mean()) + 1e-9))
            for sd in (0, 1, 2):
                acc_ = sf.read(f"{ROOT}/gen_ax_ctrlc_ge10/{s_}_{e}_s{sd}_acc.wav", dtype="float32")[0][:len(v)]
                a_ = cm.predict_proba(embm(norm(np.clip(v + acc_, -1, 1)))[None].astype(np.float32))[0][DEC.index(e)]; b_ = cm.predict_proba(embm(norm(np.clip(vp + acc_, -1, 1)))[None].astype(np.float32))[0][DEC.index(e)]; D.append((a_, b_))
        D = np.array(D); d = D[:, 1] - D[:, 0]; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out["mix_level"]["cells"][e] = {"P_dry_vocal": float(D[:, 0].mean()), "P_era_vocal": float(D[:, 1].mean()), "delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d))}
        txt.append(f"mix-level {e}: P({e}) with dry vocal {D[:, 0].mean():.3f}, with era-processed vocal {D[:, 1].mean():.3f}, delta {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] n={len(d)}")
json.dump(out, open(f"{ROOT}/results/vocal_era_dsp.json", "w"), indent=1); open(f"{ROOT}/results/vocal_era_dsp.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
