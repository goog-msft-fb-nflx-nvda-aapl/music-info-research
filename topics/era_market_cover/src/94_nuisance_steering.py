"""Does the era-steering effect survive removal of nuisance information from the probe's feature space? (CPU, stored embeddings)
Nuisance concept Z, fitted on dataset-A TRAIN only: (style) top-10 principal components of the 400 Discogs-EffNet style-tag activations of the original mix (genre/style proxy); (bw) the 9 signal statistics of the original accompaniment stem (loudness, loudness range, crest factor, RMS, bandwidth, centroid, rolloff, zcr, noise floor; bandwidth / noise-floor proxy); (both) the concatenation.
LEACE (Belrose et al. 2023) eraser fitted on the codec-resynthesized TRAIN accompaniment embeddings (MERT / MuQ codec-probe layer) with Z; the era probe (logistic regression C=0.01) is retrained on the erased train embeddings; the same eraser is applied to the stored embeddings of generated sets (results/<set>_eval_emb.npz) and the Delta statistic is recomputed (per song P(target decade | era prompt) - P(. | neutral), 95% song-bootstrap CI). Control: random orthogonal projection of the same rank (20 draws, mean).
Also reports real-audio validation accuracy and Spearman(expected decade, true decade) of each erased probe (A: n=132).
usage: python 94_nuisance_steering.py set1 set2 ... -> results/nuisance_steering.{json,txt}"""
import sys, json, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from concept_erasure import LeaceEraser
from common import ROOT, LABELS, manifest
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; rows = manifest("A"); lab = LABELS["A"]; rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
PJ = {"mert": json.load(open(f"{ROOT}/results/probe_codec_mert_A.json")), "muq": json.load(open(f"{ROOT}/results/probe_codec_muq_A.json"))}; L = {k: PJ[k]["conds"]["acccodec_norm"]["layer"] for k in PJ}
E = {k: np.load(f"{ROOT}/emb/A_{pre}acccodec_norm.npy").astype(np.float32)[:, L[k]] for k, pre in (("mert", ""), ("muq", "muq_"))}
tags = np.load(f"{ROOT}/emb/A_dgtags.npy").astype(np.float64); st = np.load(f"{ROOT}/emb/A_stats_acc.npy").astype(np.float64); st = np.nan_to_num(st)
pca = PCA(10, random_state=0).fit(tags[tr]); Zs = pca.transform(tags); Zb = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-9)
Z = {"none": None, "style": Zs, "bw": Zb, "both": np.concatenate([Zs / Zs[tr].std(0), Zb], 1)}
def fit_eraser(X, z): return LeaceEraser.fit(torch.tensor(X, dtype=torch.float64), torch.tensor(z, dtype=torch.float64))
def apply(er, X): return er(torch.tensor(X, dtype=torch.float64)).numpy().astype(np.float32)
M = {}; out = {"layers": L, "real": {}, "sets": {}}; txt = []
for k in PJ:
    for name, z in Z.items():
        if z is None: f = lambda X: X; rank = 0
        else: er = fit_eraser(E[k][tr], z[tr]); f = (lambda er: lambda X: apply(er, X))(er); rank = int(np.linalg.matrix_rank(np.cov((er(torch.tensor(E[k][tr], dtype=torch.float64)).numpy() - E[k][tr]).T)))
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(f(E[k][tr]), y[tr]); M[(k, name)] = (f, m)
        pv = m.predict_proba(f(E[k][va])); out["real"][f"{k}_{name}"] = {"rank": rank, "val_acc": float((pv.argmax(1) == y[va]).mean()), "spearman_E_vs_true": float(spearmanr(pv @ np.arange(6), y[va])[0])}
    rk = max(out["real"][f"{k}_both"]["rank"], 1); R = []
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.standard_normal((E[k].shape[1], rk))); g = (lambda Q: lambda X: X - X @ Q @ Q.T)(Q)
        R.append((g, make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(g(E[k][tr]), y[tr])))
    M[(k, "random")] = R
txt.append("real-audio validation (A n=132): " + " | ".join(f"{key}: rank {v['rank']} acc {v['val_acc']:.3f} rho {v['spearman_E_vs_true']:.2f}" for key, v in out["real"].items()))
for s in sys.argv[1:]:
    Rr = json.load(open(f"{ROOT}/results/{s}_eval.json"))["rows"]; Zd = np.load(f"{ROOT}/results/{s}_eval_emb.npz"); songs = sorted({r["clip"] for r in Rr}); res = {}
    for k in PJ:
        X = Zd[k]; ok = np.isfinite(X).all(1)
        for name in list(Z) + ["random"]:
            pr = np.full((len(X), 6), np.nan)
            if name == "random": pr[ok] = np.mean([m.predict_proba(g(X[ok])) for g, m in M[(k, name)]], 0)
            else: f, m = M[(k, name)]; pr[ok] = m.predict_proba(f(X[ok]))
            cells = {}
            for e, t in TD.items():
                P = lambda c: np.array([np.mean([pr[i, t] for i, r in enumerate(Rr) if r["clip"] == x and r["cond"] == c]) for x in songs]); d = P(e) - P("neutral"); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                cells[e] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            res[f"{k}_{name}"] = cells
    out["sets"][s] = res; txt.append(s + f" (n_songs={len(songs)})\n" + "\n".join(f"  {key:12s} " + " | ".join(f"{e} {c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" for e, c in cells.items()) for key, cells in res.items()))
json.dump(out, open(f"{ROOT}/results/nuisance_steering.json", "w"), indent=1); open(f"{ROOT}/results/nuisance_steering.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
