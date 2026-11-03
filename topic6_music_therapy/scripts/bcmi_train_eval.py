"""ds002724 (BCMI training sessions): ridge on log band power (5 bands x 32 channels). Usage: python -I bcmi_train_eval.py <npz> <out_json>
Protocols (trial level, Pearson r / CCC / RMSE): loso (unseen participant); within_session_out (per participant, train on its other sessions, test on the held-out session;
participants with >= 2 sessions); within_trial5 (per participant, 5-fold over trials, same sessions: like ds002722 within-subject).
Participant-level mean and SD of r over participants for the within-subject protocols; subject-bootstrap CI for loso. Shuffled-label control for within_session_out."""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import band_features, ridge_fit_predict, trial_metrics, boot_ci
z = np.load(sys.argv[1]); F = band_features(np.ascontiguousarray(z["X"]), int(z["fs"]), "cuda:0")
sub, ses, trial = z["sub"], z["ses"], z["trial"]; tk = np.array([a + "|" + b for a, b in zip(sub, trial)]); Y = np.stack([z["valence"], z["arousal"]], 1)
u, inv = np.unique(tk, return_inverse=True); ts = np.array([x.split("|")[0] for x in u]); ty = np.stack([Y[inv == i][0] for i in range(len(u))]); res = {}
def agg(P): return np.stack([P[inv == i].mean(0) for i in range(len(u))])
for alpha in (1.0, 10.0, 100.0):
    P = np.zeros_like(Y)
    for s in np.unique(sub):
        tr, te = sub != s, sub == s; P[te] = ridge_fit_predict(F[tr], Y[tr], F[te], alpha)
    tp = agg(P)
    for j, t in enumerate(("valence", "arousal")):
        m = trial_metrics(ty[:, j], tp[:, j], ts); m["ci"] = boot_ci(ty[:, j], tp[:, j], ts, n=300); res[f"loso/a{alpha}/{t}"] = m
    for name in ("within_session_out", "within_session_out_shuf", "within_trial5"):
        rows = {"valence": [], "arousal": []}
        for s in np.unique(sub):
            ix = np.where(sub == s)[0]; sess = np.unique(ses[ix])
            if name.startswith("within_session_out") and len(sess) < 2: continue
            P = np.zeros((len(ix), 2)); tr_ids = np.unique(tk[ix])
            if name.startswith("within_session_out"):
                for q in sess:
                    te, tr = ses[ix] == q, ses[ix] != q; Ytr = Y[ix][tr]
                    if name.endswith("shuf"): Ytr = Ytr[np.random.default_rng(0).permutation(len(Ytr))]
                    P[te] = ridge_fit_predict(F[ix][tr], Ytr, F[ix][te], alpha)
            else:
                fold = dict(zip(tr_ids, np.random.default_rng(0).integers(0, 5, len(tr_ids)))); f = np.array([fold[t] for t in tk[ix]])
                for k in range(5): P[f == k] = ridge_fit_predict(F[ix][f != k], Y[ix][f != k], F[ix][f == k], alpha)
            tt = np.unique(tk[ix])
            for j, t in enumerate(("valence", "arousal")):
                tp_ = np.array([P[tk[ix] == q, j].mean() for q in tt]); ty_ = np.array([Y[ix][tk[ix] == q, j][0] for q in tt]); rows[t].append(trial_metrics(ty_, tp_, None))
        for t in rows:
            if rows[t]: res[f"{name}/a{alpha}/{t}"] = {k: [float(np.mean([r[k] for r in rows[t]])), float(np.std([r[k] for r in rows[t]]))] for k in ("r", "ccc", "rmse")} | {"n_participants": len(rows[t])}
json.dump(res, open(sys.argv[2], "w"), indent=1)
for k, v in res.items(): print(k, {a: ([round(x, 3) for x in b] if isinstance(b, list) and a != "ci" else (round(b, 3) if isinstance(b, float) else b)) for a, b in v.items() if a != "ci"}, v.get("ci", {}).get("ccc"))
