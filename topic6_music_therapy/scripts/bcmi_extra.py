"""Extra protocols on BCMI (ds002722) band-power features (ridge). Usage: python -I bcmi_extra.py <bcmi_win4s.npz> <out_json>
 within_subject : per participant, 5-fold over that participant's trials (windows of one trial never split), train on the same participant only
 loso_window    : leave-one-subject-out, window-level continuous targets (window means of the reported VA), predictions scored per window
 loso_trialfeat : leave-one-subject-out, features averaged over the 5 windows of a trial, trial-level targets
Metrics: Pearson r, CCC, RMSE; subject-level bootstrap CI in the main met_eval; here mean over participants (+ across-participant SD) for within_subject."""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import band_features, ridge_fit_predict, trial_metrics, boot_ci
z = np.load(sys.argv[1]); X = np.ascontiguousarray(z["X"]); F = band_features(X, int(z["fs"]), "cuda:0")
sub = z["sub"]; trial = np.array([a + "|" + b for a, b in zip(sub, z["trial"])])
tv, ta = z["valence"], z["arousal"]; wv, wa = z["val_win"], z["aro_win"]; res = {}
us = np.unique(sub)
for alpha in (1.0, 10.0, 100.0):
    # within subject, trial-blocked 5-fold
    rows = {"valence": [], "arousal": []}
    for s in us:
        idx = np.where(sub == s)[0]; ut = np.unique(trial[idx]); rng = np.random.default_rng(0); fold = dict(zip(ut, rng.integers(0, 5, len(ut))))
        f = np.array([fold[t] for t in trial[idx]]); P = np.zeros((len(idx), 2))
        for k in range(5):
            P[f == k] = ridge_fit_predict(F[idx][f != k], np.stack([tv[idx], ta[idx]], 1)[f != k], F[idx][f == k], alpha)
        for j, (name, y) in enumerate((("valence", tv), ("arousal", ta))):
            tk = np.unique(trial[idx]); tp = np.array([P[trial[idx] == t, j].mean() for t in tk]); ty = np.array([y[idx][trial[idx] == t][0] for t in tk])
            rows[name].append(trial_metrics(ty, tp, None))
    for name in rows:
        res[f"within_subject/a{alpha}/{name}"] = {k: [float(np.mean([r[k] for r in rows[name]])), float(np.std([r[k] for r in rows[name]]))] for k in ("r", "ccc", "rmse")}
    # LOSO, window targets
    P = np.zeros((len(sub), 2)); Pt = np.zeros((len(sub), 2))
    Ft = np.stack([F[trial == t].mean(0) for t in trial]) if False else None
    ut = np.unique(trial); tmean = {t: F[trial == t].mean(0) for t in ut}; Ft = np.stack([tmean[t] for t in trial])
    for s in us:
        tr, te = sub != s, sub == s
        P[te] = ridge_fit_predict(F[tr], np.stack([wv, wa], 1)[tr], F[te], alpha)
        Pt[te] = ridge_fit_predict(Ft[tr], np.stack([tv, ta], 1)[tr], Ft[te], alpha)
    for j, (name, yw, yt) in enumerate((("valence", wv, tv), ("arousal", wa, ta))):
        m = trial_metrics(yw, P[:, j], sub); m["ci"] = boot_ci(yw, P[:, j], sub, n=300); res[f"loso_window/a{alpha}/{name}"] = m
        tk = np.unique(trial); tp = np.array([Pt[trial == t, j].mean() for t in tk]); ty = np.array([yt[trial == t][0] for t in tk]); ts = np.array([t.split("|")[0] for t in tk])
        m = trial_metrics(ty, tp, ts); m["ci"] = boot_ci(ty, tp, ts, n=300); res[f"loso_trialfeat/a{alpha}/{name}"] = m
json.dump(res, open(sys.argv[2], "w"), indent=1)
for k, v in res.items(): print(k, {a: ([round(x, 3) for x in b] if isinstance(b, list) and a != "ci" else (round(b, 3) if isinstance(b, float) else b)) for a, b in v.items() if a != "ci"}, v.get("ci", {}).get("ccc"))
