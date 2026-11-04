"""Diagnostics A-C on MET (no model fitting on test labels).
Usage: python -I met_diag_a.py <windows.npz> <out_json>
 A. rating reliability: variance shares (subject, song, residual), split-half reliability of song means over
    random halves of the subjects (Spearman-Brown), mean agreement of each subject with the leave-one-out song mean.
 B. feature variance shares (subject, song, residual) of trial-level log band power (66 ch x 5 bands).
 C. mass-univariate test: subject-centred trial band power vs subject-centred rating (valence, arousal, like,
    familiarity); family-wise p from the max |r| over the 330 features under within-subject permutation of the
    ratings across songs (5000 permutations).
"""
import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
z = np.load(sys.argv[1])
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
chn = [str(z["chn"][i]) for i in keep]
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
fs = int(z["fs"])
sub, song = z["sub"], z["song"]
us, ug = np.unique(sub), np.unique(song)
# trial-level features: mean over windows of log band power
import torch
Fw = []
for i in range(0, len(sub), 512):
    x = torch.from_numpy(np.ascontiguousarray(z["X"][i:i + 512][:, keep])).cuda()
    n = x.shape[-1]
    p = torch.fft.rfft(x * torch.hann_window(n, device="cuda"), dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(n, 1.0 / fs).cuda()
    Fw.append(torch.cat([torch.log(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-8) for lo, hi in BANDS], -1).cpu().numpy())
Fw = np.concatenate(Fw)
S = np.searchsorted(us, sub); G = np.searchsorted(ug, song)
nS, nG = len(us), len(ug)
F = np.full((nS, nG, Fw.shape[1]), np.nan)
R = {k: np.full((nS, nG), np.nan) for k in ("valence", "arousal", "like", "fam")}
for s in range(nS):
    for g in range(nG):
        m = (S == s) & (G == g)
        if m.any():
            F[s, g] = Fw[m].mean(0)
            for k in R:
                R[k][s, g] = z[k][m][0]
out = {"n_trials": int((~np.isnan(F[:, :, 0])).sum())}


def shares(M):
    """additive two-way decomposition on a (subject x song) matrix with NaN: fraction of total SS."""
    M = M.copy(); ok = ~np.isnan(M)
    gm = np.nanmean(M)
    sm = np.nanmean(M, 1, keepdims=True) - gm
    gmm = np.nanmean(M - sm, 0, keepdims=True) - gm
    res = M - gm - sm - gmm
    tot = np.nansum((M - gm) ** 2)
    return dict(subject=float(np.nansum((sm * ok) ** 2) / tot), song=float(np.nansum((gmm * ok) ** 2) / tot),
                residual=float(np.nansum(res ** 2) / tot))


# A
rng = np.random.default_rng(0)
A = {}
for k, M in R.items():
    d = {"shares": shares(M)}
    rs = []
    for _ in range(2000):
        p = rng.permutation(nS); a, b = p[:nS // 2], p[nS // 2:]
        ma, mb = np.nanmean(M[a], 0), np.nanmean(M[b], 0)
        r = np.corrcoef(ma, mb)[0, 1]; rs.append(2 * r / (1 + r))
    d["split_half_SB_mean"] = float(np.mean(rs)); d["split_half_SB_ci"] = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
    ag = []
    for s in range(nS):
        loo = np.nanmean(np.delete(M, s, 0), 0); ok = ~np.isnan(M[s])
        ag.append(np.corrcoef(M[s, ok], loo[ok])[0, 1])
    d["subject_vs_loo_song_mean_r"] = dict(mean=float(np.mean(ag)), min=float(np.min(ag)), max=float(np.max(ag)))
    A[k] = d
out["A_rating_reliability"] = A
# B
Bsh = np.array([[shares(F[:, :, j])[x] for j in range(F.shape[2])] for x in ("subject", "song", "residual")])
out["B_feature_shares_mean"] = dict(subject=float(Bsh[0].mean()), song=float(Bsh[1].mean()), residual=float(Bsh[2].mean()))
out["B_feature_shares_max_song"] = float(Bsh[1].max())
# C
ok = ~np.isnan(F[:, :, 0])
Fc = F - np.nanmean(F, 1, keepdims=True)           # subject-centred (across songs)
Fc = np.where(ok[:, :, None], Fc, 0.0)
Fc = Fc / (np.sqrt((Fc ** 2).sum((0, 1), keepdims=True) / ok.sum()) + 1e-9)
C = {}
names = [f"{c}|{b}" for b in ("delta", "theta", "alpha", "beta", "gamma") for c in chn]
# feature ordering from torch.cat over bands: band-major (band, channel)
for k, M in R.items():
    Mc = M - np.nanmean(M, 1, keepdims=True); Mc = np.where(ok, Mc, 0.0)
    Mc = Mc / (np.sqrt((Mc ** 2).sum() / ok.sum()) + 1e-9)
    obs = np.einsum("sg,sgf->f", Mc, Fc) / ok.sum()
    mx = np.zeros(5000)
    for t in range(5000):
        Mp = np.zeros_like(Mc)
        for s in range(nS):
            idx = np.where(ok[s])[0]
            Mp[s, idx] = Mc[s, rng.permutation(idx)]
        mx[t] = np.abs(np.einsum("sg,sgf->f", Mp, Fc) / ok.sum()).max()
    j = int(np.abs(obs).argmax())
    C[k] = dict(max_abs_r=float(np.abs(obs).max()), at=names[j], r_at=float(obs[j]),
                fwe_p=float((mx >= np.abs(obs).max()).mean()), null_max_95=float(np.percentile(mx, 95)))
out["C_massuni"] = C
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
