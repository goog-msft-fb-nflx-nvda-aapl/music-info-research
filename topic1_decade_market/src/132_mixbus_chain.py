"""T-9f: era-specific processing of the whole MIX (mix-bus chain) instead of the vocal only. The chain families of 126_vocal_chain_tune2.py (1960s: band, hiss, wow, saturation, compression, plate reverb; 1980s: high shelf + chorus + gated reverb + compression; 2010s: compression, soft limiting, brightening, short reverb) are applied to the full remix. Tuning on 60 dataset-A TRAIN REAL mixes (10 per decade) with a mix-level probe (MERT layer 12, C .01, probe_A.json 'mix_norm') fit on the remaining TRAIN mixes: objective = mean P(target decade | processed mix) - P(target | dry mix); grids as 126 (1960s 18 settings, 1980s 9, 2010s fixed). Evaluation once on the 24 evaluation songs: remix = dry original vocal + era-axis accompaniment (gen_ax_ctrlc_ge10, seeds 0-2) vs the same remix with the tuned mix-bus chain, mix-level probe fit on all TRAIN; Delta P(e) with song-bootstrap 95% CI; also the chain on the neutral-accompaniment remix (additivity).
usage (from ~/era_market, main venv, one GPU): python src/132_mixbus_chain.py -> results/mixbus_chain.{json,txt}"""
import os, sys, json, itertools, numpy as np, soundfile as sf, torch
sys.path.insert(0, "src")
src = open("src/126_vocal_chain_tune2.py").read(); head = src[:src.index("pj = json.load(")]; exec(compile(head, "chain_defs", "exec"))   # chain(), GRID, DEFAULT, norm(), rng, DEC, imports
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
rows = manifest("A"); y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]
pm = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["mix_norm"]; Lm, Cm = pm["layer"], pm["C"]; Em = np.load(f"{ROOT}/emb/A_mix_norm.npy")
tune = np.concatenate([[i for i in tr if y[i] == k][:10] for k in range(6)]); fit_idx = np.array([i for i in tr if i not in set(tune)])
clf_t = make_pipeline(StandardScaler(), LogisticRegression(C=Cm, max_iter=2000)).fit(Em[fit_idx, Lm].astype(np.float32), y[fit_idx]); clf = make_pipeline(StandardScaler(), LogisticRegression(C=Cm, max_iter=2000)).fit(Em[tr, Lm].astype(np.float32), y[tr])
proc, mert = load_mert()
def emb(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[Lm] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[Lm].mean(1)).squeeze(0).float().cpu().numpy()
GR = {"1960s": GRID["1960s"], "1980s": [dict(shelf=a, wet=b) for a, b in itertools.product((0.6, 1.0, 2.0), (0.25, 0.4, 0.55))], "2010s": [dict(ratio=a, bright=b) for a, b in itertools.product((6.0, 10.0), (0.4, 0.8))]}
mixes = {i: sf.read(rows[i]["path"], dtype="float32")[0].astype(np.float64) for i in tune}; dry = {i: clf_t.predict_proba(Em[i, Lm][None].astype(np.float32))[0] for i in tune}; out = {"tuning": {}, "eval": {}}; txt = []; best = {}
for e in ("1960s", "1980s", "2010s"):
    k = DEC.index(e); sc = [float(np.mean([clf_t.predict_proba(emb(norm(chain(mixes[i], e, p).astype(np.float32)))[None].astype(np.float32))[0][k] - dry[i][k] for i in tune])) for p in GR[e]]; best[e] = GR[e][int(np.argmax(sc))]
    out["tuning"][e] = {"settings": GR[e], "objective": sc, "best": best[e]}; txt.append(f"tuning {e}: best {best[e]} objective {max(sc):+.3f} (range {min(sc):+.3f} to {max(sc):+.3f})")
ids = [l.split(",")[0] for l in open(f"{ROOT}/gen/picked.txt").read().split("\n")]
def ci(d): bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
for e in ("1960s", "1980s", "2010s"):
    k = DEC.index(e); D = {"era": [], "neutral": []}
    for s in ids:
        v = np.load(f"{ROOT}/stems/A/{s}.npz")["vocals"].astype(np.float32)
        for tag, an in (("era", e), ("neutral", "neutral")):
            vals = []
            for sd in (0, 1, 2):
                acc = sf.read(f"{ROOT}/gen_ax_ctrlc_ge10/{s}_{an}_s{sd}_acc.wav", dtype="float32")[0][:len(v)]; mix = np.clip(v + acc, -1, 1); mp = np.clip(chain(mix.astype(np.float64), e, best[e]), -1, 1).astype(np.float32)
                vals.append(clf.predict_proba(emb(norm(mp))[None].astype(np.float32))[0][k] - clf.predict_proba(emb(norm(mix))[None].astype(np.float32))[0][k])
            D[tag].append(np.mean(vals))
    de, dn = np.array(D["era"]), np.array(D["neutral"]); out["eval"][e] = {"params": best[e], "delta_with_era_accompaniment": [float(de.mean()), *ci(de)], "delta_with_neutral_accompaniment": [float(dn.mean()), *ci(dn)]}
    txt.append(f"eval {e} mix-bus chain {best[e]}: P({e}) gain on the era-axis remix {de.mean():+.3f} [{ci(de)[0]:+.3f},{ci(de)[1]:+.3f}]; on the neutral-accompaniment remix {dn.mean():+.3f} [{ci(dn)[0]:+.3f},{ci(dn)[1]:+.3f}]"); print(txt[-1], flush=True)
json.dump(out, open(f"{ROOT}/results/mixbus_chain.json", "w"), indent=1); open(f"{ROOT}/results/mixbus_chain.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
