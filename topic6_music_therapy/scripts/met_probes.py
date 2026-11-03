"""Probes on MET band-power features (same features as met_eval.py ridge).
Usage: python -I met_probes.py <windows.npz> <out_json>
 1. song identity: leave-one-subject-out multinomial logistic regression, window and trial accuracy (chance 1/20)
 2. residual rating: rating minus the song mean of the TRAINING subjects, regressed from EEG with leave-one-subject-out
    (tests subject-specific emotion information beyond the song); evaluated on residual of the test subject
    against the same training-subject song mean."""
import sys, json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import band_features, ridge_fit_predict, trial_metrics
z = np.load(sys.argv[1])
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
F = band_features(np.ascontiguousarray(z["X"][:, keep]), int(z["fs"]), "cuda:0")
sub, song = z["sub"], z["song"]
V, A = z["valence"], z["arousal"]
out = {}
# 1 song identity
songs = np.unique(song); sid = np.searchsorted(songs, song)
accw, acct = [], []
for s in np.unique(sub):
    tr, te = sub != s, sub == s
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=300, C=0.01).fit((F[tr] - mu) / sd, sid[tr])
    p = clf.predict_proba((F[te] - mu) / sd)
    accw.append((p.argmax(1) == sid[te]).mean())
    ts = sid[te]; pl = np.log(p + 1e-9)
    for g in np.unique(ts):
        m = ts == g
        # trial = one recording; windows of the test subject with this song
        acct.append(float(clf.classes_[pl[m].mean(0).argmax()] == g))
out["song_id_window_acc"] = float(np.mean(accw)); out["song_id_trial_acc"] = float(np.mean(acct)); out["chance"] = 1 / len(songs)
# 2 residual rating
for name, Y in (("valence", V), ("arousal", A)):
    P, R = np.full(len(sub), np.nan), np.full(len(sub), np.nan)
    for s in np.unique(sub):
        tr, te = sub != s, sub == s
        smean = {g: Y[tr & (song == g)].mean() for g in songs}
        res = Y - np.array([smean[g] for g in song])
        mu = ridge_fit_predict(F[tr], res[tr, None], F[te], 10.0)[:, 0]
        P[te], R[te] = mu, res[te]
    tk = np.array([a + "|" + b for a, b in zip(sub, song)]); u, inv = np.unique(tk, return_inverse=True)
    tp = np.array([P[inv == i].mean() for i in range(len(u))]); tr_ = np.array([R[inv == i].mean() for i in range(len(u))])
    out[f"residual_{name}"] = trial_metrics(tr_, tp, None)
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1))
