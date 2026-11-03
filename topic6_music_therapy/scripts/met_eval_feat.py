"""Ridge evaluation of precomputed window features with the met_eval.py protocols.
Usage: python -I met_eval_feat.py <feat.npz> <out_json> <key> [center=0|1]
center=1 subtracts each subject's mean feature (computed over all that subject's windows, no labels used)."""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import make_folds, ridge_fit_predict, trial_metrics, boot_ci
z = np.load(sys.argv[1]); key = sys.argv[3]; center = len(sys.argv) > 4 and sys.argv[4] == "1"
F = z[key].astype(np.float64); sub, song = z["sub"], z["song"]
if center:
    for s in np.unique(sub):
        F[sub == s] -= F[sub == s].mean(0)
Y = np.stack([z["valence"], z["arousal"]], 1)
tk = np.array([a + "|" + b for a, b in zip(sub, song)]); u, inv = np.unique(tk, return_inverse=True)
ty = np.stack([Y[inv == i][0] for i in range(len(u))]); ts = np.array([x.split("|")[0] for x in u])
res = {}
for proto in ("loso", "loso_song", "strict", "random"):
    for alpha in (1.0, 10.0, 100.0):
        P = np.full(Y.shape, np.nan)
        for tr, te in make_folds(proto, sub, song):
            P[te] = ridge_fit_predict(F[tr], Y[tr], F[te], alpha)
        tp = np.stack([P[inv == i].mean(0) for i in range(len(u))])
        for j, t in enumerate(("valence", "arousal")):
            m = trial_metrics(ty[:, j], tp[:, j], ts); m["ci"] = boot_ci(ty[:, j], tp[:, j], ts, n=300)
            res[f"{proto}/a{alpha}/{t}"] = m
json.dump(res, open(sys.argv[2], "w"), indent=1)
for k, v in res.items():
    print(k, round(v["r"], 3), round(v["ccc"], 3), round(v["rmse"], 2), [round(x, 2) for x in v["ci"]["ccc"]])
