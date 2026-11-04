"""Diagnostics D-E on MET.
Usage: python -I met_diag_b.py <windows.npz> <out_json>
 D. stimulus-locking positive control (time-domain, leave-one-subject-out): correlated component analysis (CorrCA,
    Dmochowski et al.) filters learned on training subjects (maximise between-subject covariance relative to within-subject
    covariance, same song, same time); template per (song, window) = mean projected training-subject series; test trial is
    scored against the template of each of the 20 songs. Reports song identification accuracy (chance 0.05), the matched
    minus mismatched correlation, and the correlation of that per-trial value with subject-centred ratings.
 E. cross-subject alignment: covariance of each 4 s window in 5 bands (FFT cross-spectrum), trial mean, Euclidean/
    re-centring alignment per subject (uses unlabeled data of the test subject: transductive), log map at identity, ridge
    (dual form), leave-one-subject-out; with and without alignment; also song-category (5 classes) and song-id (20 classes)
    decoding with logistic regression.
"""
import sys, json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import trial_metrics, boot_ci
z = np.load(sys.argv[1])
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
X = np.ascontiguousarray(z["X"][:, keep]); fs = int(z["fs"])
sub, song, win, cat = z["sub"], z["song"], z["win"], z["cat"]
us, ug = np.unique(sub), np.unique(song)
S = np.searchsorted(us, sub); G = np.searchsorted(ug, song)
nS, nG, K = len(us), len(ug), int(win.max()) + 1
dev = "cuda:0"
out = {}
V = {k: z[k] for k in ("valence", "arousal", "like", "fam")}
Rm = {k: np.full((nS, nG), np.nan) for k in V}
catm = np.empty((nS, nG), dtype=object)
for i in range(len(sub)):
    for k in V:
        Rm[k][S[i], G[i]] = V[k][i]
    catm[S[i], G[i]] = cat[i]
okm = ~np.isnan(Rm["valence"])
# ---------------------------------------------------------------- D
T = torch.zeros(nS, nG, K, X.shape[1], X.shape[2], device=dev)       # zero-filled where window missing
P = torch.zeros(nS, nG, K, device=dev, dtype=torch.bool)
for i in range(len(sub)):
    T[S[i], G[i], win[i]] = torch.from_numpy(X[i]).to(dev); P[S[i], G[i], win[i]] = True
T = T - T.mean(-1, keepdim=True)
Cw = torch.einsum("sgkct,sgkdt->sgkcd", T, T)       # (nS,nG,K,C,C) within covariance per window
Cw_sum = Cw.sum((1, 2))                              # per subject
sumX = T.sum(0)                                      # (nG,K,C,T)
nt = X.shape[2]
res = {"ncomp": 3}
ncomp = 3
scores = np.full((nS, nG, nG), np.nan)               # test subject s, true song g, candidate song g2
for s in range(nS):
    tr = [i for i in range(nS) if i != s]
    Sx = sumX - T[s]
    Rb = torch.einsum("gkct,gkdt->cd", Sx, Sx) - Cw[tr].sum((0, 1, 2))
    Rw = Cw[tr].sum((0, 1, 2))
    Rb = (Rb + Rb.T) / 2
    Rwr = Rw / torch.trace(Rw) * Rw.shape[0]
    Rbr = Rb / torch.trace(Rw) * Rw.shape[0]
    Rwr = 0.9 * Rwr + 0.1 * torch.eye(Rw.shape[0], device=dev)
    ev, U = torch.linalg.eigh(Rwr)
    Wh = U @ torch.diag(ev ** -0.5) @ U.T
    e2, U2 = torch.linalg.eigh(Wh @ Rbr @ Wh)
    W = Wh @ U2[:, -ncomp:]                           # (C, ncomp)
    # templates: mean over training subjects of projected series, per (song, window)
    cnt = P[tr].sum(0).clamp(min=1)                   # (nG,K)
    tmpl = torch.einsum("cm,gkct->gkmt", W, Sx) / cnt[:, :, None, None]
    tmpl = tmpl - tmpl.mean(-1, keepdim=True); tmpl = tmpl / (tmpl.norm(dim=-1, keepdim=True) + 1e-9)
    y = torch.einsum("cm,gkct->gkmt", W, T[s])
    y = y - y.mean(-1, keepdim=True); y = y / (y.norm(dim=-1, keepdim=True) + 1e-9)
    # corr[g_true,k,g_cand,m]
    cc = torch.einsum("gkmt,hkmt->gkhm", y, tmpl)
    msk = P[s][:, :, None, None].float()
    sc = ((cc * msk).sum((1, 3)) / (msk.sum((1, 3)) * 1.0 + 1e-9)).cpu().numpy()   # (nG,nG)
    for g in range(nG):
        if okm[s, g]:
            scores[s, g] = sc[g]
pred = np.nanargmax(np.where(np.isnan(scores), -9, scores), 2)
acc = np.array([[pred[s, g] == g for g in range(nG) if okm[s, g]] for s in range(nS)], dtype=object)
accs = np.array([np.mean(a) for a in acc])
match = np.array([[scores[s, g, g] - np.nanmean(np.delete(scores[s, g], g)) for g in range(nG)] for s in range(nS)])
match[~okm] = np.nan
res["song_id_acc_mean"] = float(accs.mean()); res["song_id_acc_by_subject"] = [float(a) for a in accs]
res["matched_minus_mismatched_corr_mean"] = float(np.nanmean(match))
res["matched_minus_mismatched_by_subject_min_max"] = [float(np.nanmin(np.nanmean(match, 1))), float(np.nanmax(np.nanmean(match, 1)))]
# category accuracy from the predicted song (is the predicted song of the same category?)
catg = np.array([catm[okm[:, g]][0, ] if False else None for g in range(nG)], dtype=object)
songcat = [next(catm[s, g] for s in range(nS) if okm[s, g]) for g in range(nG)]
res["category_acc_from_predicted_song"] = float(np.mean([songcat[pred[s, g]] == songcat[g] for s in range(nS) for g in range(nG) if okm[s, g]]))
# relation of per-trial matched score with ratings (subject-centred), within-subject permutation
rng = np.random.default_rng(0)
rel = {}
for k, M in Rm.items():
    Mc = M - np.nanmean(M, 1, keepdims=True); Fc = match - np.nanmean(match, 1, keepdims=True)
    a = Mc[okm]; b = Fc[okm]
    r = float(np.corrcoef(a, b)[0, 1])
    null = []
    for _ in range(2000):
        M2 = Mc.copy()
        for s in range(nS):
            idx = np.where(okm[s])[0]; M2[s, idx] = Mc[s, rng.permutation(idx)]
        null.append(np.corrcoef(M2[okm], b)[0, 1])
    rel[k] = dict(r=r, p_two_sided=float((np.abs(null) >= abs(r)).mean()))
res["matched_score_vs_subject_centred_rating"] = rel
out["D_stimulus_locking"] = res
print("D", json.dumps(res, indent=1), flush=True)
del T, Cw, sumX, P
torch.cuda.empty_cache()
# ---------------------------------------------------------------- E
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
Cn = X.shape[1]
cov = np.zeros((len(sub), len(BANDS), Cn, Cn), np.float32)
for i in range(0, len(sub), 256):
    x = torch.from_numpy(X[i:i + 256]).to(dev)
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / fs).to(dev)
    Xf = torch.fft.rfft(x * torch.hann_window(x.shape[-1], device=dev), dim=-1)
    for b, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f < hi)
        A = Xf[..., m]
        cov[i:i + 256, b] = torch.einsum("wcf,wdf->wcd", A, A.conj()).real.div(int(m.sum())).cpu().numpy()
# trial-level mean covariance
tid = S * nG + G
trials = np.unique(tid)
Ct = np.stack([cov[tid == t].mean(0) for t in trials])           # (n_trials, nb, C, C)
ts, tg = trials // nG, trials % nG
tY = {k: np.array([Rm[k][s, g] for s, g in zip(ts, tg)]) for k in V}
tcat = np.array([catm[s, g] for s, g in zip(ts, tg)])


def logm_vec(C):
    C = torch.from_numpy(C).to(dev).double()
    ev, U = torch.linalg.eigh(C)
    L = U @ torch.diag_embed(torch.log(ev.clamp(min=1e-12))) @ U.transpose(-1, -2)
    iu = torch.triu_indices(Cn, Cn, device=dev)
    w = torch.where(iu[0] == iu[1], 1.0, 2 ** 0.5).double()
    return (L[:, iu[0], iu[1]] * w).cpu().numpy()


def feats(align):
    outf = []
    for b in range(len(BANDS)):
        Cb = Ct[:, b].astype(np.float64)
        if align:
            Cb = Cb.copy()
            for s in range(nS):
                m = ts == s
                M = torch.from_numpy(Cb[m].mean(0)).to(dev)
                ev, U = torch.linalg.eigh(M)
                Mi = (U @ torch.diag(ev ** -0.5) @ U.T).cpu().numpy()
                Cb[m] = Mi @ Cb[m] @ Mi
        else:
            # global log-Euclidean-style normalisation: trace-normalise each trial
            Cb = Cb / np.trace(Cb, axis1=1, axis2=2)[:, None, None] * Cn
        outf.append(logm_vec(Cb))
    return np.concatenate(outf, 1)


def kridge(A, Y, B, lam):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A = (A - mu) / sd; B = (B - mu) / sd
    ym = Y.mean(0)
    Kmat = A @ A.T
    a = np.linalg.solve(Kmat + lam * np.eye(len(A)) * A.shape[1], Y - ym)
    return B @ (A.T @ a) + ym


E = {}
for name, align in (("aligned", True), ("trace_norm_only", False)):
    Fe = feats(align)
    for lam in (0.01, 0.1, 1.0, 10.0):
        for tgt in ("valence", "arousal"):
            Y = tY[tgt][:, None]
            for variant in ("raw", "subject_centred_target"):
                Yv = Y.copy()
                if variant == "subject_centred_target":
                    for s in range(nS):
                        Yv[ts == s] -= Yv[ts == s].mean()
                p = np.zeros(len(trials)); psh = np.zeros(len(trials))
                for s in range(nS):
                    tr, te = ts != s, ts == s
                    p[te] = kridge(Fe[tr], Yv[tr], Fe[te], lam)[:, 0]
                    perm = np.random.default_rng(s).permutation(tr.sum())
                    psh[te] = kridge(Fe[tr], Yv[tr][perm], Fe[te], lam)[:, 0]
                target_eval = Yv[:, 0]
                m = trial_metrics(target_eval, p, ts); m["ci"] = boot_ci(target_eval, p, ts, n=300)
                msh = trial_metrics(target_eval, psh, ts)
                E[f"{name}/lam{lam}/{tgt}/{variant}"] = dict(r=m["r"], ccc=m["ccc"], ci_r=m["ci"]["r"], shuffled_r=msh["r"])
    # classification of category and song id (leave-one-subject-out)
    sid_arr = tg
    for nm, lab in (("category", tcat), ("song_id", sid_arr)):
        accs = []
        for s in range(nS):
            tr, te = ts != s, ts == s
            mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=200, C=1e-4).fit((Fe[tr] - mu) / sd, lab[tr])
            accs.append((clf.predict((Fe[te] - mu) / sd) == lab[te]).mean())
        E[f"{name}/{nm}_acc"] = float(np.mean(accs))
        E[f"{name}/{nm}_chance"] = float(1.0 / len(np.unique(lab)))
    print(name, {k: v for k, v in E.items() if k.startswith(name) and ("acc" in k or "chance" in k)}, flush=True)
out["E_riemannian_alignment"] = E
json.dump(out, open(sys.argv[2], "w"), indent=1)
best = sorted([(v["ccc"], k) for k, v in E.items() if isinstance(v, dict)], reverse=True)[:6]
print("top by ccc", best)
