"""T-9e: extended vocal chains and a proper random control (same protocol as 124_vocal_chain_tune.py: parameters tuned on 60 TRAIN clips against a probe fit on the remaining TRAIN clips, one evaluation on the 24 evaluation songs). New: 1980s grid beyond the previous edge (shelf {1.0,1.5,2.0} x reverb wet {.40,.55,.70}); 1960s chain with tape cues (band (120 Hz, 7 kHz) or telephone-like (300 Hz, 3.4 kHz); hiss at {-45,-35,-25} dB below the vocal RMS (white noise high-passed at 2 kHz); wow {0,1,3} ms delay modulation at 0.7 Hz; saturation drive 2, compression and plate reverb of 118); random control = 20 settings drawn uniformly from the union of the grid ranges per decade, vocal-probe Delta on the 24 songs reported as mean and range. Original text of 124 follows: tune the era-specific vocal chains of 118_vocal_era_dsp.py on dataset-A TRAIN vocals only, then evaluate the selected chains once on the 24 evaluation songs. Tuning: 60 TRAIN clips (10 per decade, first by sample id) are split from the probe fit: the vocal era probe (MERT layer 20, C .01) used for the objective is fit on the REMAINING TRAIN clips' dry vocal embeddings; objective of a setting = mean over the 60 tuning clips of P(target decade | processed vocal) - P(target decade | dry vocal). Grids: 1960s: low cut {80,150,300} Hz x high cut {4000,7000,10000} Hz x tape saturation drive {1,2,4} (compression -24 dB 3:1 and plate reverb RT60 .5 s wet .12 fixed); 1980s: high-shelf gain {.3,.6,1.0} x reverb wet {.15,.25,.40} (chorus and gated reverb as 118); 2010s: compression ratio {4,6,10} x brightening {.2,.4,.8} (high-pass 90 Hz, soft limiting, short reverb as 118). Evaluation: the 24 evaluation songs (gen/picked.txt): for each decade the default chain of 118, the tuned chain and a random-chain control (the setting of the grid with the median tuning objective); vocal-probe Delta P(target) vs dry (song-bootstrap 95% CI) and the mix-level Delta with the era-axis accompaniment (as 118 MIX=1).
usage (from ~/era_market, main venv, one GPU): python src/126_vocal_chain_tune2.py -> results/vocal_chain_tune2.{json,txt}"""
import os, sys, json, itertools, numpy as np, soundfile as sf, torch, pyloudnorm as pyln
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
DEC = LABELS["A"]; rng = np.random.default_rng(0)
def bp(x, lo, hi): return signal.sosfilt(signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x)
def hp(x, f): return signal.sosfilt(signal.butter(2, f, "highpass", fs=SR, output="sos"), x)
def sat(x, d): return np.tanh(d * x) / np.tanh(d)
def comp(x, thr_db, ratio, tc=0.02):
    a = np.exp(-1.0 / (tc * SR)); env = signal.lfilter([1 - a], [1, -a], np.abs(x)) + 1e-8; thr = 10 ** (thr_db / 20); g = np.where(env > thr, (env / thr) ** (1.0 / ratio - 1.0), 1.0); y = x * g
    return y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
def reverb(x, rt60, wet, gate=None, seed=0):
    r = np.random.default_rng(seed); n = int(rt60 * SR); ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    if gate: ir[int(gate * SR):] = 0
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9; y = signal.fftconvolve(x, ir)[:len(x)]; return (1 - wet) * x + wet * y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
def chorus(x, depth_ms, wet): t = np.arange(len(x)); d = (0.020 * SR) + depth_ms * 1e-3 * SR * np.sin(2 * np.pi * 0.4 * t / SR); return (1 - wet) * x + wet * np.interp(np.clip(t - d, 0, len(x) - 1), t, x)
def hiss(x, db, seed=1):
    n = np.random.default_rng(seed).standard_normal(len(x)); n = hp(n, 2000); return x + n * (np.sqrt((x ** 2).mean()) * 10 ** (db / 20) / (np.sqrt((n ** 2).mean()) + 1e-9))
def wow(x, ms):
    if ms <= 0: return x
    t = np.arange(len(x)); d = ms * 1e-3 * SR * np.sin(2 * np.pi * 0.7 * t / SR); return np.interp(np.clip(t - d, 0, len(x) - 1), t, x)
def chain(x, e, p):
    if e == "1960s": return hiss(wow(reverb(comp(sat(bp(x, p["lo"], p["hi"]), 2.0), -24, 3.0), 0.5, 0.12), p["wow"]), p["hiss"])
    if e == "1980s": y = x + p["shelf"] * hp(x, 4000); return comp(reverb(chorus(y, 4, 0.25), 1.0, p["wet"], gate=0.35), -26, 3.0)
    y = hp(x, 90); y = comp(y, -30, p["ratio"]); y = np.tanh(1.5 * y) / np.tanh(1.5); y = y + p["bright"] * hp(y, 6000); return reverb(y, 0.25, 0.08)
DEFAULT = {"1960s": {"lo": 120, "hi": 7000, "hiss": -90.0, "wow": 0.0}, "1980s": {"shelf": 0.6, "wet": 0.25}, "2010s": {"ratio": 6.0, "bright": 0.4}}
GRID = {"1960s": [dict(lo=a[0], hi=a[1], hiss=b, wow=c) for a, b, c in itertools.product(((120, 7000), (300, 3400)), (-45.0, -35.0, -25.0), (0.0, 1.0, 3.0))], "1980s": [dict(shelf=a, wet=b) for a, b in itertools.product((1.0, 1.5, 2.0), (0.40, 0.55, 0.70))], "2010s": [dict(ratio=10.0, bright=0.8)]}
meter = pyln.Meter(SR)
def norm(w): l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) or np.abs(w).max() < 1e-4 else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
pj = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["voc_norm"]; L, C = pj["layer"], pj["C"]; rows = manifest("A"); y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]
tune = np.concatenate([[i for i in tr if y[i] == k][:10] for k in range(6)]); fit_idx = np.array([i for i in tr if i not in set(tune)]); E_ = np.load(f"{ROOT}/emb/A_voc_norm.npy")
clf_t = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E_[fit_idx, L].astype(np.float32), y[fit_idx]); clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E_[tr, L].astype(np.float32), y[tr])
proc, mert = load_mert()
def emb(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[L] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[L].mean(1)).squeeze(0).float().cpu().numpy()
vocs = {i: np.load(f"{ROOT}/stems/A/{rows[i]['sample_id']}.npz")["vocals"].astype(np.float64) for i in tune}; dry = {i: clf_t.predict_proba(E_[i, L][None].astype(np.float32))[0] for i in tune}
out = {"tuning": {}, "eval": {}}; txt = []; best = {}; median = {}
for e in ("1960s", "1980s", "2010s"):
    k = DEC.index(e); sc = []
    for p in GRID[e]:
        d = [clf_t.predict_proba(emb(norm(chain(vocs[i], e, p).astype(np.float32)))[None].astype(np.float32))[0][k] - dry[i][k] for i in tune]; sc.append(float(np.mean(d)))
    order = np.argsort(sc); best[e] = GRID[e][int(order[-1])]; median[e] = GRID[e][int(order[len(order) // 2])]
    out["tuning"][e] = {"settings": GRID[e], "objective": sc, "best": best[e], "median": median[e], "default_objective_in_grid": sc[GRID[e].index(DEFAULT[e])] if DEFAULT[e] in GRID[e] else None}
    txt.append(f"tuning {e}: best {best[e]} objective {max(sc):+.3f}; median setting {median[e]} {sc[int(order[len(order) // 2])]:+.3f}; default {DEFAULT[e]} " + (f"{sc[GRID[e].index(DEFAULT[e])]:+.3f}" if DEFAULT[e] in GRID[e] else "not in grid"))
ids = [l.split(",")[0] for l in open(f"{ROOT}/gen/picked.txt").read().split("\n")]; rowsd = {r["sample_id"]: r for r in rows}; pm = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["mix_norm"]; Lm, Cm = pm["layer"], pm["C"]; Em = np.load(f"{ROOT}/emb/A_mix_norm.npy"); cm = make_pipeline(StandardScaler(), LogisticRegression(C=Cm, max_iter=2000)).fit(Em[tr, Lm].astype(np.float32), y[tr])
def embm(w):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[Lm] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[Lm].mean(1)).squeeze(0).float().cpu().numpy()
def ci(d): bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
for e in ("1960s", "1980s", "2010s"):
    k = DEC.index(e)
    for name, p in (("default", DEFAULT[e]), ("tuned", best[e])):
        dv, dm = [], []
        for s in ids:
            v = np.load(f"{ROOT}/stems/A/{s}.npz")["vocals"].astype(np.float32); vp = chain(v.astype(np.float64), e, p).astype(np.float32); dv.append(clf.predict_proba(emb(norm(vp))[None].astype(np.float32))[0][k] - clf.predict_proba(emb(norm(v))[None].astype(np.float32))[0][k]); vp = vp * (np.sqrt((v ** 2).mean()) / (np.sqrt((vp ** 2).mean()) + 1e-9))
            for sd in (0, 1, 2):
                acc = sf.read(f"{ROOT}/gen_ax_ctrlc_ge10/{s}_{e}_s{sd}_acc.wav", dtype="float32")[0][:len(v)]; dm.append(cm.predict_proba(embm(norm(np.clip(vp + acc, -1, 1)))[None].astype(np.float32))[0][k] - cm.predict_proba(embm(norm(np.clip(v + acc, -1, 1)))[None].astype(np.float32))[0][k])
        dv, dm = np.array(dv), np.array(dm); out["eval"][f"{e}|{name}"] = {"params": p, "vocal_delta": float(dv.mean()), "vocal_ci": ci(dv), "mix_delta": float(dm.mean()), "mix_ci": ci(dm)}
        txt.append(f"eval {e} {name} {p}: vocal probe {dv.mean():+.3f} [{ci(dv)[0]:+.3f},{ci(dv)[1]:+.3f}]; mix-level with era-axis accompaniment {dm.mean():+.3f} [{ci(dm)[0]:+.3f},{ci(dm)[1]:+.3f}]")
    print(e, flush=True)
rc = np.random.default_rng(5); out["random_control"] = {}
for e in ("1960s", "1980s", "2010s"):
    k = DEC.index(e); vals = []
    for _ in range(20):
        p = {"1960s": lambda: dict(lo=float(rc.choice([80, 120, 300])), hi=float(rc.choice([3400, 7000, 10000])), hiss=float(rc.uniform(-45, -25)), wow=float(rc.uniform(0, 3))), "1980s": lambda: dict(shelf=float(rc.uniform(0.3, 2.0)), wet=float(rc.uniform(0.15, 0.7))), "2010s": lambda: dict(ratio=float(rc.uniform(4, 10)), bright=float(rc.uniform(0.2, 0.8)))}[e]()
        d = []
        for s in ids:
            v = np.load(f"{ROOT}/stems/A/{s}.npz")["vocals"].astype(np.float32); vp = chain(v.astype(np.float64), e, p).astype(np.float32); d.append(clf.predict_proba(emb(norm(vp))[None].astype(np.float32))[0][k] - clf.predict_proba(emb(norm(v))[None].astype(np.float32))[0][k])
        vals.append(float(np.mean(d)))
    out["random_control"][e] = vals; txt.append(f"random chains {e} (20 draws): vocal-probe delta mean {np.mean(vals):+.3f}, min {np.min(vals):+.3f}, max {np.max(vals):+.3f}")
json.dump(out, open(f"{ROOT}/results/vocal_chain_tune2.json", "w"), indent=1); open(f"{ROOT}/results/vocal_chain_tune2.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
