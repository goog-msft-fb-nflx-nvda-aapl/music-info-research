"""Proxy composition groups without Discogs ids (the public manifest hides artist, title, Discogs id and YouTube id): Discogs-VINet (Araz et al. 2024; third-party AGPL code kept outside the repo) embeddings of every clip's mix (A and B, all splits); cosine similarity; groups = connected components of the graph with edges at cosine >= T.
Outputs: results/clique_proxy.{json,txt}: similarity distribution of each clip's nearest neighbour, number of groups and of cross-split edges per threshold, and the decade / market probe accuracy under group-aware 5-fold out-of-fold evaluation (pooled labeled clips, MERT mix per-stem layer as probe_<ds>.json) versus ordinary stratified out-of-fold.
usage: python 96_clique_proxy.py (GPU for the embeddings, cached in emb/<ds>_vinet.npy)"""
import sys, os, json, numpy as np, torch, librosa, soundfile as sf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, manifest, LABELS
sys.path.insert(0, os.path.expanduser("~/era_market/ext/Discogs-VINet")); from model.nets.cqtnet import CQTNet
net = CQTNet(); net.load_state_dict(torch.load(os.path.expanduser("~/era_market/ext/Discogs-VINet/logs/checkpoints/Discogs-VINet/model_checkpoint.pth"), map_location="cpu", weights_only=True)["model_state_dict"]); net = net.cuda().eval()
def emb(y24):
    y = librosa.resample(np.asarray(y24, dtype=np.float32), orig_sr=SR, target_sr=22050); c = np.abs(librosa.cqt(y=y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)).T.astype(np.float16).astype(np.float32)
    if c.shape[0] < 8000: c = np.pad(c, ((0, 8000 - c.shape[0]), (0, 0)))
    T = c.shape[0] // 20; c = c[:T * 20].reshape(T, 20, 84).mean(1); c = np.where(c < 0, 0, c); c /= c.max() + 1e-6
    with torch.inference_mode(): return net(torch.as_tensor(c.T)[None, None].cuda()).float().cpu().numpy()[0]
out = {}; txt = []; rng = np.random.default_rng(0)
for ds in ("A", "B"):
    rows = manifest(ds); f = f"{ROOT}/emb/{ds}_vinet.npy"
    if not os.path.exists(f): np.save(f, np.stack([emb(sf.read(r["path"], dtype="float32")[0]) for r in rows]))
    V = np.load(f); V /= np.linalg.norm(V, axis=1, keepdims=True); S = V @ V.T; np.fill_diagonal(S, -1); split = np.array([r["split"] for r in rows]); nn = S.max(1)
    res = {"n": len(rows), "nn_sim_quantiles": {str(q): float(np.quantile(nn, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}, "thr": {}}
    lab = LABELS[ds]; y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); P = json.load(open(f"{ROOT}/results/probe_{ds}.json"))["conds"]["mix_norm"]; X = np.load(f"{ROOT}/emb/{ds}_mix_norm.npy")[:, P["layer"]].astype(np.float32); lb = np.where(y >= 0)[0]
    def oof(groups):
        pr = np.zeros(len(lb), int); sp = GroupKFold(5).split(X[lb], y[lb], groups[lb]) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0).split(X[lb], y[lb])
        for a, b in sp: pr[b] = make_pipeline(StandardScaler(), LogisticRegression(C=P["C"], max_iter=3000)).fit(X[lb][a], y[lb][a]).predict(X[lb][b])
        return float((pr == y[lb]).mean())
    res["oof_acc_stratified"] = oof(None)
    for T in (0.6, 0.7, 0.8, 0.9):
        i, j = np.where(np.triu(S >= T, 1)); nc, g = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(rows),) * 2), directed=False)
        cross = int(sum(split[a] != split[b] for a, b in zip(i, j))); sizes = np.bincount(g); res["thr"][str(T)] = {"edges": int(len(i)), "cross_split_edges": cross, "groups": int(nc), "clips_in_groups_ge2": int((sizes[g] >= 2).sum()), "max_group": int(sizes.max()), "oof_acc_group_kfold": oof(g) if sizes.max() < len(rows) // 5 else None}
    out[ds] = res; txt.append(f"{ds} n={res['n']}: nearest-neighbour cosine quantiles " + str(res["nn_sim_quantiles"]) + f"; stratified OOF acc {res['oof_acc_stratified']:.3f}\n" + "\n".join(f"  T={T}: " + ", ".join(f"{k} {v}" for k, v in d.items()) for T, d in res["thr"].items()))
json.dump(out, open(f"{ROOT}/results/clique_proxy.json", "w"), indent=1); open(f"{ROOT}/results/clique_proxy.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
