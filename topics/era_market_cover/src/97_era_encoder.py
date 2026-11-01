"""M-1: nuisance-controlled ordinal era encoder on frozen features of the codec-resynthesized accompaniment (dataset A).
Input: MERT-v2 layer-22 (or MuQ layer-9) mean-pooled accompaniment embedding, standardized on TRAIN. Encoder h: Linear(d,256)-GELU-Dropout(.2)-Linear(256,D) -> z (D=8). Heads: CORAL ordinal head (K-1 cumulative logits from a scalar w.z + K-1 biases) for the decade, and gradient-reversal adversary heads (MLP 64) that regress the nuisance vector N (10 style PCs of the Discogs-EffNet tags of the original mix + 9 signal statistics of the original accompaniment stem, standardized on TRAIN) from z. loss = CORAL + lam * MSE(adversary) with the encoder receiving the reversed gradient (adversary trained to predict, encoder to hide).
Fixed hyperparameters (not tuned on validation): 300 epochs, batch 128, AdamW lr 1e-3 wd 1e-2, 3 training seeds, lam in {0, 0.3, 1, 3}. Validation (n=132, real codec-resynthesized stems): ordinal accuracy, MAE (decades), Spearman(expected decade, true); nuisance leakage = held-out R2 of a fresh ridge and of a fresh MLP (fit on TRAIN z, scored on validation z) from z to N (lower = better hidden).
Saves emb/A_eraz_<enc>_lam<lam>_s<seed>.npy (z for all 1290 clips), results/era_encoder.{json,txt}; generated sets are scored by the CORAL head in 98_era_axis_lora.py / 97b. usage: python 97_era_encoder.py [mert|muq] [adv|leace]"""
import sys, os, json, numpy as np, torch, torch.nn as nn
from concept_erasure import LeaceEraser
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from common import ROOT, LABELS, manifest
enc = sys.argv[1] if len(sys.argv) > 1 else "mert"; pre = {"mert": "", "muq": "muq_"}[enc]; D = 8; K = 6
rows = manifest("A"); lab = LABELS["A"]; y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
L = json.load(open(f"{ROOT}/results/probe_codec_{enc}_A.json"))["conds"]["acccodec_norm"]["layer"]; X = np.load(f"{ROOT}/emb/A_{pre}acccodec_norm.npy").astype(np.float32)[:, L]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
tags = np.load(f"{ROOT}/emb/A_dgtags.npy").astype(np.float64); st = np.nan_to_num(np.load(f"{ROOT}/emb/A_stats_acc.npy").astype(np.float64)); pca = PCA(10, random_state=0).fit(tags[tr])
N = np.concatenate([pca.transform(tags), st], 1); N = (N - N[tr].mean(0)) / (N[tr].std(0) + 1e-9); N = N.astype(np.float32)
CLS = os.environ.get("CLS") == "1"   # CLS=1: add a six-way softmax head on z so that z also encodes decade identity (not only the 1-D ordinal coordinate); files get the suffix c
SC = "c" if CLS else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "adv"   # adv: gradient-reversal heads; leace: LEACE-erase N from the input features, then lam=0 encoder
if MODE == "leace":
    er = LeaceEraser.fit(torch.tensor(Xs[tr], dtype=torch.float64), torch.tensor(N[tr], dtype=torch.float64)); Xs = er(torch.tensor(Xs, dtype=torch.float64)).numpy().astype(np.float32)
class GR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.h = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, D)); s.w = nn.Linear(D, 1, bias=False); s.b = nn.Parameter(torch.linspace(-2, 2, K - 1)); s.cls = nn.Linear(D, K); s.adv = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, N.shape[1]))
    def forward(s, x): z = s.h(x); return z, s.w(z) + s.b          # (B,D), (B,K-1) cumulative logits for y > k
def probs(c): p = torch.sigmoid(c); p = torch.cat([torch.ones_like(p[:, :1]), p, torch.zeros_like(p[:, :1])], 1); return (p[:, :-1] - p[:, 1:]).clamp_min(1e-6)
def train(lam, seed):
    torch.manual_seed(seed); net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); xt = torch.tensor(Xs[tr]); yt = torch.tensor(y[tr]); nt = torch.tensor(N[tr]); ords = (yt[:, None] > torch.arange(K - 1)[None]).float(); g = torch.Generator().manual_seed(seed)
    for ep in range(300):
        net.train(); perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), 128):
            b = perm[i:i + 128]; z, c = net(xt[b]); loss = nn.functional.binary_cross_entropy_with_logits(c, ords[b]) + (nn.functional.cross_entropy(net.cls(z), yt[b]) if CLS else 0); loss = loss + (lam * nn.functional.mse_loss(net.adv(GR.apply(z, 1.0)), nt[b]) if lam > 0 else 0)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()
out = {"enc": enc, "layer": L, "runs": {}}; txt = []
for lam in ((0.0, 0.1, 0.3) if MODE == "adv" else (0.0,)):
    for seed in range(3):
        net = train(lam, seed)
        with torch.no_grad(): Z, C = net(torch.tensor(Xs)); P = probs(C).numpy(); Z = Z.numpy()
        np.save(f"{ROOT}/emb/A_eraz_{enc}_{MODE}{SC}{lam}_s{seed}.npy", Z); E_ = P @ np.arange(K)
        r = {"val_cls_acc": float((net.cls(torch.tensor(Z[va])).argmax(1).numpy() == y[va]).mean()) if CLS else None, "val_acc": float((P[va].argmax(1) == y[va]).mean()), "val_mae_decades": float(np.abs(E_[va] - y[va]).mean()), "val_spearman": float(spearmanr(E_[va], y[va])[0])}
        zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6); rd = Ridge(alpha=1.0).fit(zs[tr], N[tr]); mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, random_state=0).fit(zs[tr], N[tr])
        def r2(m): pr = m.predict(zs[va]); return float(np.mean(1 - ((N[va] - pr) ** 2).sum(0) / ((N[va] - N[va].mean(0)) ** 2).sum(0)))
        r["nuisance_R2_ridge"] = r2(rd); r["nuisance_R2_mlp"] = r2(mlp); out["runs"][f"{MODE}{lam}_s{seed}"] = r
    rs = [out["runs"][f"{MODE}{lam}_s{s}"] for s in range(3)]; m = lambda k: float(np.mean([r[k] for r in rs])); sdv = lambda k: float(np.std([r[k] for r in rs]))
    txt.append(f"{enc} {MODE}{SC} lam={lam}: val acc {m('val_acc'):.3f}±{sdv('val_acc'):.3f}, MAE {m('val_mae_decades'):.2f} decades, spearman {m('val_spearman'):.3f}; nuisance R2 from z (val, higher = more leakage): ridge {m('nuisance_R2_ridge'):.3f}, mlp {m('nuisance_R2_mlp'):.3f}")
json.dump(out, open(f"{ROOT}/results/era_encoder_{enc}_{MODE}{SC}.json", "w"), indent=1); open(f"{ROOT}/results/era_encoder_{enc}_{MODE}{SC}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
