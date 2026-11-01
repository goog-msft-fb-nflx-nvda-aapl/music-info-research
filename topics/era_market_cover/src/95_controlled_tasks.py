"""Two controlled versions of the probe tasks (CPU, stored MERT-v2 embeddings, per-stem layer/C as in probe_<ds>.json; pooled train+val clips, NOT artist-grouped: the manifest has no artist ids, so absolute values may be optimistic and only the comparisons are meaningful).
(1) Market with sung language held fixed: dataset B clips whose Whisper-LID argmax on the vocal stem is English (n=489). Tasks: US vs UK (binary) and the five-way task without Brazil (12 clips). 5-fold stratified out-of-fold probes (3 fold seeds), reported with accuracy, balanced accuracy, macro-F1, majority-class share, 95% bootstrap CI over clips.
(2) Decade with the genre/style cluster held out: dataset A pooled clips are clustered (k-means, 6 clusters, Discogs-EffNet embedding of the original mix); the probe is trained on 5 clusters and tested on the sixth (leave-one-cluster-out), compared with ordinary stratified 6-fold out-of-fold accuracy; also the mean absolute decade error and the majority-class share per held-out cluster.
usage: python 95_controlled_tasks.py -> results/controlled_tasks.{json,txt}"""
import json, numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
rng = np.random.default_rng(0); txt = []; out = {}
def load(ds, c):
    P = json.load(open(f"{ROOT}/results/probe_{ds}.json"))["conds"][c]; return np.load(f"{ROOT}/emb/{ds}_{c}.npy")[:, P["layer"]].astype(np.float32), P["C"]
def oof(X, y, C, seed):
    pr = np.zeros(len(y), int)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y): pr[b] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[a], y[a]).predict(X[b])
    return pr
def ci(v, B=2000): b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
# (1) market, English-sung
rows = manifest("B"); lab = LABELS["B"]; y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); lang = json.load(open(f"{ROOT}/emb/lid_langs.json")); top = np.array([lang[i] for i in np.load(f"{ROOT}/emb/B_lid_voc.npy").argmax(1)])
lab_idx = np.where(y >= 0)[0]; en = np.array([i for i in lab_idx if top[i] == "en"]); txt.append(f"(1) dataset B, English-sung labeled clips n={len(en)}: " + ", ".join(f"{lab[k]} {(y[en]==k).sum()}" for k in range(6)))
for name, keep in (("US_vs_UK", [lab.index("US"), lab.index("UK")]), ("five_way_no_Brazil", [k for k in range(6) if lab[k] != "Brazil"])):
    idx = np.array([i for i in en if y[i] in keep]); yy = np.array([keep.index(v) for v in y[idx]]); maj = float(np.bincount(yy).max() / len(yy)); out[name] = {"n": int(len(idx)), "majority_share": maj}
    for c in ("mix_norm", "voc_norm", "acc_norm"):
        X, C = load("B", c); X = X[idx]; accs, bal, f1 = [], [], []
        for s in range(3): p = oof(X, yy, C, s); accs.append((p == yy).astype(float)); bal.append(balanced_accuracy_score(yy, p)); f1.append(f1_score(yy, p, average="macro"))
        a = np.mean(accs, 0); out[name][c] = {"acc": float(a.mean()), "ci": ci(a), "balanced_acc": float(np.mean(bal)), "macro_f1": float(np.mean(f1))}
    txt.append(f"  {name} (n={len(idx)}, majority {maj:.3f}, chance {1/len(keep):.3f}): " + " | ".join(f"{c} acc {v['acc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] bal {v['balanced_acc']:.3f} F1 {v['macro_f1']:.3f}" for c, v in out[name].items() if isinstance(v, dict)))
# all-language reference with the same metrics
idx = lab_idx; yy = y[idx]
for c in ("mix_norm",):
    X, C = load("B", c); p = oof(X[idx], yy, C, 0); txt.append(f"  reference, all languages, six-way (n={len(idx)}, majority {np.bincount(yy).max()/len(yy):.3f}): {c} acc {(p==yy).mean():.3f} bal {balanced_accuracy_score(yy, p):.3f} F1 {f1_score(yy, p, average='macro'):.3f}")
# (2) decade, genre cluster held out
rows = manifest("A"); lab = LABELS["A"]; y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); idx = np.where(y >= 0)[0]; yy = y[idx]
dge = np.load(f"{ROOT}/emb/A_dgemb.npy")[idx]; cl = KMeans(6, n_init=10, random_state=0).fit(dge).labels_; out["decade_loco"] = {}
txt.append(f"(2) dataset A pooled n={len(idx)}, 6 EffNet-style clusters (n per cluster {np.bincount(cl).tolist()})")
for c in ("mix_norm", "acc_norm"):
    X, C = load("A", c); X = X[idx]; pr = np.zeros(len(idx), int)
    for k in range(6):
        te = cl == k; pr[te] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[~te], yy[~te]).predict(X[te])
    ordn = oof(X, yy, C, 0); r = {"loco_acc": float((pr == yy).mean()), "loco_ci": ci((pr == yy).astype(float)), "loco_mae_decades": float(np.abs(pr - yy).mean()), "random_oof_acc": float((ordn == yy).mean()), "random_oof_mae": float(np.abs(ordn - yy).mean()), "per_cluster": []}
    for k in range(6): m = cl == k; r["per_cluster"].append({"n": int(m.sum()), "majority": float(np.bincount(yy[m], minlength=6).max() / m.sum()), "acc": float((pr[m] == yy[m]).mean())})
    out["decade_loco"][c] = r
    txt.append(f"  {c}: leave-one-cluster-out acc {r['loco_acc']:.3f} [{r['loco_ci'][0]:.3f},{r['loco_ci'][1]:.3f}] MAE {r['loco_mae_decades']:.2f} decades vs stratified OOF acc {r['random_oof_acc']:.3f} MAE {r['random_oof_mae']:.2f}; per cluster (n, majority, acc): " + "; ".join(f"{p['n']}, {p['majority']:.2f}, {p['acc']:.2f}" for p in r["per_cluster"]))
json.dump(out, open(f"{ROOT}/results/controlled_tasks.json", "w"), indent=1); open(f"{ROOT}/results/controlled_tasks.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
