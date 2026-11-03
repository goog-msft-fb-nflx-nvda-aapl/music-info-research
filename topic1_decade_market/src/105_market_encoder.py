"""M-3: nuisance-controlled market encoder on frozen features of the codec-resynthesized accompaniment (dataset B, six release markets in the 1980s), the market analogue of 97_era_encoder.py.
Input: MERT-v2 layer-L (or MuQ) mean-pooled accompaniment embedding (layer from probe_codec_<enc>_B.json), standardized on TRAIN. Nuisance N = 10 PCs of the Discogs-EffNet style tags of the original mix + 9 accompaniment signal statistics (+ the language-ID posterior of the accompaniment when NUIS=lang), standardized on TRAIN; LEACE (fitted on TRAIN) removes N from the input, then encoder Linear(d,256)-GELU-Dropout(.2)-Linear(256,8) -> z with a six-way softmax head (markets are unordered).
Fixed hyperparameters as 97_era_encoder.py (300 epochs, batch 128, AdamW lr 1e-3 wd 1e-2, 3 seeds). Reports validation (n=102) accuracy of the head, accuracy of a logistic probe (C=0.01) fit on TRAIN z, accuracy of the same probe on the LEACE-erased input without encoder (reference), and held-out nuisance R2 from z (ridge, MLP).
Saves emb/B_marketz_<enc>_<tag>_s<seed>.npy (z for all clips) and results/market_encoder_<enc>_<tag>.{json,txt}; tag = leace (style+stats) or leacel (+language). usage: python 105_market_encoder.py [mert|muq] ; NUIS=lang for the language-erased variant"""
import sys, os, json, numpy as np, torch, torch.nn as nn
from concept_erasure import LeaceEraser
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from common import ROOT, LABELS, manifest
enc = sys.argv[1] if len(sys.argv) > 1 else "mert"; pre = {"mert": "", "muq": "muq_"}[enc]; D = 8; lab = LABELS["B"]; K = len(lab); LANG = os.environ.get("NUIS") == "lang"; tag = "leacel" if LANG else "leace"
rows = manifest("B"); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
L = json.load(open(f"{ROOT}/results/probe_codec_{enc}_B.json"))["conds"]["acccodec_norm"]["layer"]; X = np.load(f"{ROOT}/emb/B_{pre}acccodec_norm.npy").astype(np.float32)[:, L]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
tags = np.load(f"{ROOT}/emb/B_dgtags.npy").astype(np.float64); st = np.nan_to_num(np.load(f"{ROOT}/emb/B_stats_acc.npy").astype(np.float64)); pca = PCA(10, random_state=0).fit(tags[tr]); parts = [pca.transform(tags), st]
if LANG: parts.append(np.load(f"{ROOT}/emb/B_lid_acc.npy").astype(np.float64).reshape(len(rows), -1))
N = np.concatenate(parts, 1); N = ((N - N[tr].mean(0)) / (N[tr].std(0) + 1e-9)).astype(np.float32)
er = LeaceEraser.fit(torch.tensor(Xs[tr], dtype=torch.float64), torch.tensor(N[tr], dtype=torch.float64)); Xs = er(torch.tensor(Xs, dtype=torch.float64)).numpy().astype(np.float32)
def logit_acc(F): m = LogisticRegression(C=0.01, max_iter=2000).fit(F[tr], y[tr]); return float((m.predict(F[va]) == y[va]).mean())
ref = logit_acc(Xs)
class Net(nn.Module):
    def __init__(s): super().__init__(); s.h = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, D)); s.cls = nn.Linear(D, K)
    def forward(s, x): z = s.h(x); return z, s.cls(z)
def train(seed):
    torch.manual_seed(seed); net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); xt = torch.tensor(Xs[tr]); yt = torch.tensor(y[tr]); g = torch.Generator().manual_seed(seed)
    for ep in range(300):
        net.train(); perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), 128):
            b = perm[i:i + 128]; z, c = net(xt[b]); loss = nn.functional.cross_entropy(c, yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()
out = {"enc": enc, "layer": L, "tag": tag, "n_nuisance": int(N.shape[1]), "erased_input_logistic_val_acc": ref, "runs": {}}; txt = []
for seed in range(3):
    net = train(seed)
    with torch.no_grad(): Z, C = net(torch.tensor(Xs)); Z = Z.numpy(); C = C.numpy()
    np.save(f"{ROOT}/emb/B_marketz_{enc}_{tag}_s{seed}.npy", Z); zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)
    rd = Ridge(alpha=1.0).fit(zs[tr], N[tr]); mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, random_state=0).fit(zs[tr], N[tr])
    def r2(m): pr = m.predict(zs[va]); return float(np.mean(1 - ((N[va] - pr) ** 2).sum(0) / ((N[va] - N[va].mean(0)) ** 2).sum(0)))
    out["runs"][f"s{seed}"] = {"val_head_acc": float((C[va].argmax(1) == y[va]).mean()), "val_logistic_on_z_acc": logit_acc(zs), "nuisance_R2_ridge": r2(rd), "nuisance_R2_mlp": r2(mlp)}
rs = list(out["runs"].values()); m = lambda k: float(np.mean([r[k] for r in rs])); sdv = lambda k: float(np.std([r[k] for r in rs]))
txt.append(f"{enc} {tag}: head acc {m('val_head_acc'):.3f}±{sdv('val_head_acc'):.3f}, logistic on z {m('val_logistic_on_z_acc'):.3f}, logistic on erased input {ref:.3f} (chance {1/K:.3f}); nuisance R2 from z (val): ridge {m('nuisance_R2_ridge'):.3f}, mlp {m('nuisance_R2_mlp'):.3f}")
json.dump(out, open(f"{ROOT}/results/market_encoder_{enc}_{tag}.json", "w"), indent=1); open(f"{ROOT}/results/market_encoder_{enc}_{tag}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
