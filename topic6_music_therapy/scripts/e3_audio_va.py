"""E3: audio -> valence/arousal evaluator from MERT-v1-95M layer features.
Train/validate on PMEmo static annotations (10-fold CV over songs, ridge), then apply to BCMI synthetic clips (ds002722) and compare
with (a) the clips' mean reported valence/arousal across listeners and (b) the targeted affect code. Usage: python -I e3_audio_va.py <data_dir> <out_json>"""
import sys, json, os, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
d, out = sys.argv[1], sys.argv[2]
P = np.load(f"{d}/feat/mert_pmemo.npz"); B = np.load(f"{d}/feat/mert_bcmi.npz")
ann = pd.read_csv(f"{d}/pmemo/x/PMEmo2019/annotations/static_annotations.csv").set_index("musicId")
allids = [int(f.split(".")[0]) for f in P["files"]]
keep = np.array([i in ann.index for i in allids]); ids = [i for i, k in zip(allids, keep) if k]
PF = P["F"][keep]
Y = ann.loc[ids][["Valence(mean)", "Arousal(mean)"]].values; print("PMEmo label ranges", Y.min(0), Y.max(0), len(ids))
def ccc(a, b): return 2*np.mean((a-a.mean())*(b-b.mean()))/(a.var()+b.var()+(a.mean()-b.mean())**2)
def ridge(A, y, B_, lam):
    mu, sd = A.mean(0), A.std(0)+1e-6; A=(A-mu)/sd; B_=(B_-mu)/sd; ym=y.mean(0)
    W = np.linalg.solve(A.T@A + lam*len(A)*np.eye(A.shape[1]), A.T@(y-ym)); return B_@W+ym
rng = np.random.default_rng(0); fold = rng.integers(0, 10, len(ids)); res = {"cv": {}}
for L in range(13):
    X = PF[:, L]
    for lam in (0.1, 1.0, 10.0):
        pr = np.zeros_like(Y)
        for k in range(10):
            tr, te = fold != k, fold == k; pr[te] = ridge(X[tr], Y[tr], X[te], lam)
        res["cv"][f"L{L}/lam{lam}"] = {t: dict(r=float(pearsonr(Y[:, j], pr[:, j])[0]), ccc=float(ccc(Y[:, j], pr[:, j])),
                                               rmse=float(np.sqrt(((Y[:, j]-pr[:, j])**2).mean()))) for j, t in enumerate(("valence", "arousal"))}
best = {t: max(res["cv"], key=lambda k: res["cv"][k][t]["ccc"]) for t in ("valence", "arousal")}; res["best"] = best; print(best)
# BCMI application
z = np.load(f"{d}/bcmi_win4s.npz"); clips = pd.DataFrame({"clip": z["song"], "v": z["valence"], "a": z["arousal"], "cat": z["cat"]}).drop_duplicates().groupby("clip").agg(v=("v", "mean"), a=("a", "mean"), cat=("cat", "first"), n=("v", "size"))
bm = {f: i for i, f in enumerate(B["files"])}; res["bcmi"] = {}
tv = {"L": 0, "N": 1, "H": 2}
for j, t in enumerate(("valence", "arousal")):
    L, lam = int(best[t].split("/")[0][1:]), float(best[t].split("lam")[1])
    pred = ridge(PF[:, L], Y[:, j], B["F"][:, L], lam)
    pc = np.array([pred[bm[c]] for c in clips.index]); rep = clips["v" if j == 0 else "a"].values
    code = np.array([tv[c[0 if j == 0 else 2]] for c in clips["cat"]])      # cat like HVLA: valence letter 0, arousal letter 2
    res["bcmi"][t] = dict(n_clips=len(clips), r_reported=float(pearsonr(pc, rep)[0]), rho_reported=float(spearmanr(pc, rep)[0]),
                          rho_target=float(spearmanr(pc, code)[0]), r_target_vs_reported=float(spearmanr(code, rep)[0]))
print(json.dumps(res["bcmi"], indent=1)); json.dump(res, open(out, "w"), indent=1)
for t in ("valence", "arousal"): print(t, best[t], res["cv"][best[t]][t])
