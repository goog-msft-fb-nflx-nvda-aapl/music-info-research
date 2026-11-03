"""Guide topic 1 on the PA1 data with proxy pairs: pairs of dataset-A clips whose Discogs-VINet embeddings (emb/A_vinet.npy, L2-normalized) have cosine >= T are taken as candidate recordings of the same composition (precision unverified: the manifest hides ids, 96_clique_proxy.py), restricted to pairs with different decade labels. For each stem condition (mix, vocals, accompaniment; MERT, layer and C of probe_A.json) the expected decade E of every labeled clip is obtained out-of-fold (5-fold stratified, a probe fit on the other folds; the two clips of a pair can be in different folds). Per pair: predicted shift = E(later clip) - E(earlier clip) with 'later' defined by the label decade; reported: number of pairs, fraction with the predicted shift positive (the predicted order agrees with the label order), mean predicted shift, mean label gap (decades), Spearman correlation of predicted shift and label gap, and the same quantities for 1000 random cross-decade pairs of dataset-A clips (control: same label gap distribution not enforced, all pairs of different decade).
usage (from ~/era_market, main venv): python src/134_proxy_pairs.py -> results/proxy_pairs.{json,txt}"""
import os, sys, json, numpy as np
sys.path.insert(0, "src")
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import ROOT, LABELS, manifest
DEC = LABELS["A"]; rows = manifest("A"); y = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rows]); lab = np.where(y >= 0)[0]; rng = np.random.default_rng(0)
V = np.load(f"{ROOT}/emb/A_vinet.npy").astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True); S = V @ V.T; np.fill_diagonal(S, -1); PJ = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]
E = {}
for c in ("mix_norm", "voc_norm", "acc_norm"):
    X = np.load(f"{ROOT}/emb/A_{c}.npy").astype(np.float32)[:, PJ[c]["layer"]]; P = np.zeros((len(rows), 6))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(lab, y[lab]):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=PJ[c]["C"], max_iter=2000)).fit(X[lab[a]], y[lab[a]]); P[lab[b]] = m.predict_proba(X[lab[b]])
    E[c] = P @ np.arange(6)
def stats(pairs):
    out = {"n": len(pairs)}
    for c in E:
        d = []; g = []
        for i, j in pairs:
            lo, hi = (i, j) if y[i] < y[j] else (j, i); d.append(E[c][hi] - E[c][lo]); g.append(y[hi] - y[lo])
        d, g = np.array(d), np.array(g); out[c] = {"frac_order_agrees": float((d > 0).mean()), "mean_shift": float(d.mean()), "mean_label_gap": float(g.mean()), "spearman": float(spearmanr(d, g)[0]) if len(d) > 3 else None}
    return out
res = {}; txt = []
for T in (0.8, 0.75, 0.7):
    iu = np.argwhere(np.triu(S >= T)); pairs = [(i, j) for i, j in iu if y[i] >= 0 and y[j] >= 0 and y[i] != y[j]]; res[f"T{T}"] = stats(pairs)
    txt.append(f"T={T}: {res[f'T{T}']['n']} cross-decade proxy pairs; " + "; ".join(f"{c.split('_')[0]}: order agrees {v['frac_order_agrees']:.2f}, mean shift {v['mean_shift']:+.2f} (label gap {v['mean_label_gap']:.2f}), spearman {v['spearman'] if v['spearman'] is None else round(v['spearman'], 2)}" for c, v in res[f"T{T}"].items() if c != "n"))
rp = []
while len(rp) < 1000:
    i, j = rng.choice(lab, 2, replace=False)
    if y[i] != y[j]: rp.append((i, j))
res["random"] = stats(rp); txt.append("random cross-decade pairs: " + "; ".join(f"{c.split('_')[0]}: order agrees {v['frac_order_agrees']:.2f}, mean shift {v['mean_shift']:+.2f} (label gap {v['mean_label_gap']:.2f}), spearman {round(v['spearman'], 2)}" for c, v in res["random"].items() if c != "n"))
json.dump(res, open(f"{ROOT}/results/proxy_pairs.json", "w"), indent=1); open(f"{ROOT}/results/proxy_pairs.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
