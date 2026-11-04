"""Diagnostic F on MET: is the missing stimulus locking (diag D) caused by onset jitter?
Usage: python -I met_diag_c.py <windows.npz> <out_json>
 F1. lag-tolerant CorrCA song identification: same filters/templates as diag D, but each (test song, candidate song)
     score is the maximum normalised cross-correlation over lags of +-3 s (continuous series, zero-filled gaps).
 F2. slow band-power envelope song identification: per trial, log band power (66 ch x 5 bands) per 4 s window, z-scored over
     time; template = mean over training subjects; score = mean correlation over features (insensitive to ~1 s jitter).
Leave-one-subject-out; song-id accuracy (chance 0.05) and matched-minus-mismatched correlation.
"""
import sys, json, numpy as np, torch
z = np.load(sys.argv[1])
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]
X = np.ascontiguousarray(z["X"][:, keep]); fs = int(z["fs"])
sub, song, win = z["sub"], z["song"], z["win"]
us, ug = np.unique(sub), np.unique(song)
S = np.searchsorted(us, sub); G = np.searchsorted(ug, song)
nS, nG, K = len(us), len(ug), int(win.max()) + 1
dev = "cuda:0"; nt = X.shape[2]; C = X.shape[1]
okm = np.zeros((nS, nG), bool)
T = torch.zeros(nS, nG, K, C, nt, device=dev); P = torch.zeros(nS, nG, K, device=dev, dtype=torch.bool)
for i in range(len(sub)):
    T[S[i], G[i], win[i]] = torch.from_numpy(X[i]).to(dev); P[S[i], G[i], win[i]] = True; okm[S[i], G[i]] = True
T = T - T.mean(-1, keepdim=True)
out = {}
# ---------------- F1
Cw = torch.einsum("sgkct,sgkdt->sgkcd", T, T)
sumX = T.sum(0); ncomp = 3; maxlag = int(3 * fs)
sc1 = np.full((nS, nG, nG), np.nan)
for s in range(nS):
    tr = [i for i in range(nS) if i != s]
    Sx = sumX - T[s]
    Rw = Cw[tr].sum((0, 1, 2)); Rb = torch.einsum("gkct,gkdt->cd", Sx, Sx) - Rw; Rb = (Rb + Rb.T) / 2
    tr_ = torch.trace(Rw); Rwr = 0.9 * Rw / tr_ * C + 0.1 * torch.eye(C, device=dev); Rbr = Rb / tr_ * C
    ev, U = torch.linalg.eigh(Rwr); Wh = U @ torch.diag(ev ** -0.5) @ U.T
    e2, U2 = torch.linalg.eigh(Wh @ Rbr @ Wh); W = Wh @ U2[:, -ncomp:]
    cnt = P[tr].sum(0).clamp(min=1)
    tm = torch.einsum("cm,gkct->gkmt", W, Sx) / cnt[:, :, None, None]
    y = torch.einsum("cm,gkct->gkmt", W, T[s])
    tm = tm.permute(0, 2, 1, 3).reshape(nG, ncomp, K * nt); y = y.permute(0, 2, 1, 3).reshape(nG, ncomp, K * nt)
    tm = tm - tm.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True)
    tm = tm / tm.norm(dim=-1, keepdim=True); y = y / (y.norm(dim=-1, keepdim=True) + 1e-9)
    N = y.shape[-1]
    Fy = torch.fft.rfft(y, n=2 * N, dim=-1); Ft = torch.fft.rfft(tm, n=2 * N, dim=-1)
    for g in range(nG):
        if not okm[s, g]:
            continue
        xc = torch.fft.irfft(Fy[g][None] * Ft.conj(), n=2 * N, dim=-1)       # (cand, m, 2N)
        lags = torch.cat([xc[..., :maxlag + 1], xc[..., -maxlag:]], -1)
        sc1[s, g] = lags.max(-1).values.mean(-1).cpu().numpy()
del T, Cw, sumX; torch.cuda.empty_cache()


def summarize(sc):
    pred = np.nanargmax(np.where(np.isnan(sc), -9, sc), 2)
    acc = np.mean([pred[s, g] == g for s in range(nS) for g in range(nG) if okm[s, g]])
    mm = np.nanmean([sc[s, g, g] - np.mean(np.delete(sc[s, g], g)) for s in range(nS) for g in range(nG) if okm[s, g]])
    return dict(song_id_acc=float(acc), matched_minus_mismatched=float(mm), chance=1.0 / nG)


out["F1_lag_tolerant_corrca"] = summarize(sc1)
print("F1", out["F1_lag_tolerant_corrca"], flush=True)
# ---------------- F2
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
Fw = []
for i in range(0, len(sub), 512):
    x = torch.from_numpy(X[i:i + 512]).to(dev); n = x.shape[-1]
    p = torch.fft.rfft(x * torch.hann_window(n, device=dev), dim=-1).abs() ** 2
    f = torch.fft.rfftfreq(n, 1.0 / fs).to(dev)
    Fw.append(torch.cat([torch.log(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-8) for lo, hi in BANDS], -1).cpu().numpy())
Fw = np.concatenate(Fw)
E = np.full((nS, nG, K, Fw.shape[1]), np.nan)
for i in range(len(sub)):
    E[S[i], G[i], win[i]] = Fw[i]
mu = np.nanmean(E, 2, keepdims=True); sd = np.nanstd(E, 2, keepdims=True) + 1e-9
Ez = (E - mu) / sd
sc2 = np.full((nS, nG, nG), np.nan)
for s in range(nS):
    tr = [i for i in range(nS) if i != s]
    tmpl = np.nanmean(Ez[tr], 0)                       # (nG,K,F)
    for g in range(nG):
        if not okm[s, g]:
            continue
        y = Ez[s, g]; v = ~np.isnan(y[:, 0])
        for h in range(nG):
            t = tmpl[h][v]; yy = y[v]
            t = t - t.mean(0); yy = yy - yy.mean(0)
            sc2[s, g, h] = float(np.mean((t * yy).sum(0) / (np.linalg.norm(t, axis=0) * np.linalg.norm(yy, axis=0) + 1e-9)))
out["F2_band_power_envelope"] = summarize(sc2)
print("F2", out["F2_band_power_envelope"], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
