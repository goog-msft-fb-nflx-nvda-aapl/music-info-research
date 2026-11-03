"""Handcrafted features from windows npz -> feature npz. Usage: python -I make_band_feats.py <windows.npz> <out.npz>
keys: bp (log band power, 5 bands x C), rel (relative band power), hjorth (activity/mobility/complexity x C),
asym (log alpha power differences for left-right homologous pairs), all (concatenation of bp, rel, hjorth)."""
import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from met_eval import band_features
z = np.load(sys.argv[1]); dev = "cuda:0"
keep = [i for i, c in enumerate(z["chn"]) if str(c).upper() != "EOG"]; names = [str(z["chn"][i]) for i in keep]
X = np.ascontiguousarray(z["X"][:, keep]); fs = int(z["fs"])
bp = band_features(X, fs, dev); C = X.shape[1]
b = bp.reshape(len(X), C, 5) if False else bp.reshape(len(X), 5, C) if False else bp
# band_features concatenates per band along the last axis: (N, C*5) ordered band-major (band0 all channels, band1 ...)
b = bp.reshape(len(X), 5, C)
p = np.exp(b); rel = (p / p.sum(1, keepdims=True)).reshape(len(X), -1)
xt = torch.from_numpy(X).to(dev); d1 = xt.diff(dim=-1); d2 = d1.diff(dim=-1)
act = xt.var(-1) + 1e-8; mob = (d1.var(-1) / act).sqrt(); comp = (d2.var(-1) / (d1.var(-1) + 1e-8)).sqrt() / (mob + 1e-8)
hj = torch.cat([act.log(), mob, comp], 1).cpu().numpy()
idx = {n.upper(): i for i, n in enumerate(names)}
pairs = [(l, r) for l in idx for r in idx if l.endswith(tuple("13579")) and r == l[:-1] + str(int(l[-1]) + 1)]
asym = np.stack([b[:, 2, idx[l]] - b[:, 2, idx[r]] for l, r in pairs], 1) if pairs else np.zeros((len(X), 1))
print("asym pairs", len(pairs))
bpf = bp.reshape(len(X), -1)
np.savez_compressed(sys.argv[2], sub=z["sub"], song=z["song"], valence=z["valence"], arousal=z["arousal"],
                    bp=bpf, rel=rel, hjorth=hj, asym=asym, all=np.concatenate([bpf, rel, hj, asym], 1))
print("saved", sys.argv[2])
