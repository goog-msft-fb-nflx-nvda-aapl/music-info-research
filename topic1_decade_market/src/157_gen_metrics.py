"""Text-to-motion style generation metrics on the era axis, computed in an embedding space (existing metrics; only the embedding, the classes and the reference are chosen here).
Embedding: MERT layer 22 (--enc mert) or MuQ layer 9 (--enc muq) of generated accompaniments from results/<tag>_eval_emb.npz (names clip|cond|seed), PCA 32 fitted on the codec-resynthesised real accompaniments of dataset A (emb/A_acccodec_norm.npy layer 22, emb/A_muq_acccodec_norm.npy layer 9).
- R-Precision top-1/top-2: rank of the target decade among the 6 real-decade centroids (Euclidean distance in PCA space) for every generated clip of that target; chance 1/6 and 2/6; reference row = real clips of dataset A (leave-nothing-out, centroids from all real clips).
- MM-Dist: mean Euclidean distance from a generated clip to the real centroid of its target decade (reference: real clips to own centroid; neutral generation to the same centroid).
- Diversity: mean pairwise distance among generated clips of one target (random 300 pairs per cell fixed seed), reference real clips of the decade.
- MultiModality: mean pairwise distance between seeds of the same song and target, averaged over songs.
usage (from ~/era_market, venv_mcl): python src/157_gen_metrics.py --tags ax_onehot_pfx16_ge10_hpcp ax_ctrlc_ge10_hpcp --enc mert --out results/gen_metrics_mert"""
import argparse, json, os, sys, itertools, numpy as np
sys.path.insert(0, "src")
from common import ROOT, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--enc", choices=["mert", "muq"], required=True); ap.add_argument("--out", required=True); ap.add_argument("--pca", type=int, default=32); a = ap.parse_args()
DEC = list(LABELS["A"]); rows = manifest("A"); fn, L = ("A_acccodec_norm.npy", 22) if a.enc == "mert" else ("A_muq_acccodec_norm.npy", 9)
R = np.load(os.path.join(ROOT, "emb", fn))[:, L].astype(np.float64); lab = np.array([r["label"] for r in rows]); keep = np.isin(lab, DEC); R, lab = R[keep], lab[keep]
mu = R.mean(0); U, S, Vt = np.linalg.svd(R - mu, full_matrices=False); P = Vt[:a.pca].T
proj = lambda x: (x - mu) @ P
Rz = proj(R); C = np.stack([Rz[lab == d].mean(0) for d in DEC])
def rprec(Z, tgt):
    d = np.linalg.norm(Z[:, None] - C[None], axis=2); rk = (d < d[:, [DEC.index(tgt)]]).sum(1); return float((rk == 0).mean()), float((rk <= 1).mean())
def mmdist(Z, tgt): return float(np.linalg.norm(Z - C[DEC.index(tgt)], axis=1).mean())
def divers(Z, rng, n=300):
    if len(Z) < 2: return float("nan")
    i = rng.integers(0, len(Z), n); j = rng.integers(0, len(Z), n); m = i != j; return float(np.linalg.norm(Z[i[m]] - Z[j[m]], axis=1).mean())
rng = np.random.default_rng(0); out = {"enc": a.enc, "pca": a.pca, "real": {}, "tags": {}}; txt = [f"gen metrics ({a.enc}, PCA {a.pca}), per target decade: R-Prec top1/top2 | MM-Dist | Diversity | MultiModality"]
for d in DEC:
    z = Rz[lab == d]; t1, t2 = rprec(z, d); out["real"][d] = {"n": len(z), "rprec1": t1, "rprec2": t2, "mmdist": mmdist(z, d), "diversity": divers(z, rng)}
    txt.append(f"real {d} n={len(z)}: {t1:.3f}/{t2:.3f} | {out['real'][d]['mmdist']:.2f} | {out['real'][d]['diversity']:.2f} | -")
for g in a.tags:
    z = np.load(os.path.join(ROOT, "results", g + "_eval_emb.npz"), allow_pickle=True); names = [str(n) for n in z["names"]]; Z = proj(z[a.enc].astype(np.float64))
    clip = np.array([n.split("|")[0] for n in names]); cond = np.array([n.split("|")[1] for n in names]); res = {}
    for c in sorted(set(cond)):
        m = cond == c; Zc = Z[m]; r = {"n": int(m.sum()), "diversity": divers(Zc, rng)}
        mm = []
        for s in sorted(set(clip[m])):
            zs = Zc[clip[m] == s]
            if len(zs) > 1: mm.append(np.mean([np.linalg.norm(x - y) for x, y in itertools.combinations(zs, 2)]))
        r["multimodality"] = float(np.mean(mm)) if mm else float("nan")
        if c in DEC: r["rprec1"], r["rprec2"] = rprec(Zc, c); r["mmdist"] = mmdist(Zc, c)
        else: r["mmdist_to_each_decade"] = {d: mmdist(Zc, d) for d in DEC}
        res[c] = r
        txt.append(f"{g} {c} n={r['n']}: " + (f"{r['rprec1']:.3f}/{r['rprec2']:.3f} | {r['mmdist']:.2f}" if c in DEC else "- | -") + f" | {r['diversity']:.2f} | {r['multimodality']:.2f}")
    out["tags"][g] = res
json.dump(out, open(a.out + ".json", "w"), indent=1); open(a.out + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
