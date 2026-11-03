"""T-4j market read-out of the lyric route: the vocal market probe (dataset-B vocal probe, MERT layer and C from probe_B.json 'voc_norm', StandardScaler + logistic regression fit on dataset-B TRAIN vocal embeddings; validation accuracy reproduced) applied to the separated generated vocal of the stem-swap route (ace_swap_voc/<id>_<cond>_s{SEED}.wav, 12 pilot clips: US/UK source, English; conditions orig = English control, es -> Spain, pt -> Brazil, de -> Germany, it -> Italy). Reported: P(target market | vocal of condition c) for c in es, pt, de, it; its difference to the same market's probability under the English control (orig) of the same clip (paired over 12 clips, song-bootstrap 95% CI, 2000 resamples) ; top-1 hit of the target market; and the same quantities for the ORIGINAL vocal stems (reference).
LEACE=1: the probe is refit after LEACE erasure of the Whisper large-v3 language group from the TRAIN vocal embeddings (file name suffix _leace; the validation accuracy of the erased probe is printed and stored). Env: TEXT, VOCDIR, OUTNAME select the text file, vocal directory and output name.
usage (from ~/era_market, main venv, one GPU): python src/120_swap_market_probe.py -> results/swap_market_probe.{json,txt}"""
import os, sys, json, numpy as np, soundfile as sf, torch, pyloudnorm as pyln
SEED = int(os.environ.get("SEED", "0"))   # generation seed of the files to read (default 0)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
MK = LABELS["B"]; TG = {"es": "Spain", "pt": "Brazil", "de": "Germany", "it": "Italy", "en": "US+UK"}; VOC = os.environ.get("VOCDIR", "ace_swap_voc"); OUTN = os.environ.get("OUTNAME", "swap_market_probe"); rng = np.random.default_rng(0)
pj = json.load(open(f"{ROOT}/results/probe_B.json"))["conds"]["voc_norm"]; L, C = pj["layer"], pj["C"]; rows = manifest("B"); y = np.array([MK.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
E_ = np.load(f"{ROOT}/emb/B_voc_norm.npy"); clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E_[tr, L].astype(np.float32), y[tr]); acc0 = float((clf.predict(E_[va, L].astype(np.float32)) == y[va]).mean()); assert abs(acc0 - pj["val_acc"]) < 1e-6; ACC_PLAIN = acc0
if os.environ.get("LEACE") == "1":   # language erased from the probe: LEACE (concept-erasure) fit on TRAIN vocal embeddings with the Whisper large-v3 language group (en, pt, es, it, de, other) as concept, then the same logistic regression
    from concept_erasure import LeaceEraser
    lid3 = np.load(f"{ROOT}/emb/B_lid3_voc.npy"); langs = json.load(open(f"{ROOT}/emb/lid_langs.json")); G = ["en", "pt", "es", "it", "de"]; gl = np.array([G.index(langs[int(np.argmax(lid3[i]))]) if langs[int(np.argmax(lid3[i]))] in G else 5 for i in range(len(rows))]); Z = np.eye(6)[gl]
    sc = StandardScaler().fit(E_[tr, L].astype(np.float32)); Xtr = sc.transform(E_[tr, L].astype(np.float32)); er = LeaceEraser.fit(torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Z[tr], dtype=torch.float64))
    def fe(X): return er(torch.tensor(sc.transform(X.astype(np.float32)), dtype=torch.float64)).numpy().astype(np.float32)
    lr = LogisticRegression(C=C, max_iter=2000).fit(fe(E_[tr, L]), y[tr]); acc0 = float((lr.predict(fe(E_[va, L])) == y[va]).mean()); print("language-erased vocal probe validation accuracy", acc0, flush=True)
    class _C:
        def predict_proba(self, X): return lr.predict_proba(fe(X))
    clf = _C(); OUTN = os.environ.get("OUTNAME", "swap_market_probe") + "_leace"
proc, mert = load_mert(); meter = pyln.Meter(SR)
def norm(w): l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) or np.abs(w).max() < 1e-4 else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def emb(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[L] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[L].mean(1)).squeeze(0).float().cpu().numpy()
TEXT = os.environ.get("TEXT", "lyrics_route_text.json"); TEXT = os.environ.get("TEXT", "lyrics_route_text.json"); T = json.load(open(f"{ROOT}/results/{TEXT}")); ids = list(T); P = {}
for s in ids:
    P[(s, "original")] = clf.predict_proba(emb(norm(np.load(f"{ROOT}/stems/B/{s}.npz")["vocals"].astype(np.float32)))[None].astype(np.float32))[0]
    for c in ("orig", "en", "es", "pt", "de", "it"):
        fp = f"{ROOT}/{VOC}/{s}_{c}_s{SEED}.wav"
        if os.path.exists(fp): P[(s, c)] = clf.predict_proba(emb(norm(sf.read(fp, dtype="float32")[0]))[None].astype(np.float32))[0]
    print(s, flush=True)
def tp(p, m): return p[MK.index("US")] + p[MK.index("UK")] if m == "US+UK" else p[MK.index(m)]
out = {"val_acc_vocal_probe": acc0, "layer": L, "C": C, "cells": {}}; txt = [f"vocal market probe (MERT layer {L}, C {C}) validation accuracy {acc0:.3f}; {len(ids)} clips ({TEXT})"]
for c, m in TG.items():
    S = [s for s in ids if (s, c) in P and (s, "orig") in P and T[s].get("src_lang", "en") != c]
    if len(S) < 3: continue
    pc = np.array([tp(P[(s, c)], m) for s in S]); p0 = np.array([tp(P[(s, "orig")], m) for s in S]); po = np.array([tp(P[(s, "original")], m) for s in S]); d = pc - p0; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    top = lambda p: (MK[int(np.argmax(p))] in ("US", "UK")) if m == "US+UK" else (MK[int(np.argmax(p))] == m); hit = float(np.mean([top(P[(s, c)]) for s in S])); hit0 = float(np.mean([top(P[(s, "orig")]) for s in S]))
    out["cells"][c] = {"target": m, "n": len(S), "P_target": float(pc.mean()), "P_target_source_language_control": float(p0.mean()), "P_target_original_vocal": float(po.mean()), "delta_vs_control": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "top1_hit": hit, "top1_hit_control": hit0}
    txt.append(f"{c} -> {m} (n={len(S)}): P(target) {pc.mean():.3f} (control {p0.mean():.3f}, original vocal {po.mean():.3f}); delta vs control {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]; target top-1 {hit:.2f} (control {hit0:.2f})")
json.dump(out, open(f"{ROOT}/results/{OUTN}.json", "w"), indent=1); open(f"{ROOT}/results/{OUTN}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
