"""FAD with sample-size extrapolation (Gui et al., arXiv 2311.01616: FAD is biased by the number of generated samples; extrapolate to infinite sample size). For each cell: FAD(subsample of N generated clips, real decade) for N in --sizes (20 random subsets each), linear fit FAD = a + b/N, report a = FAD_inf. Embeddings and references as 154_cell_fad_mert.py (--enc mert) or 156_cell_fad_muq.py (--enc muq); PCA 32 on real references.
usage (from ~/era_market, venv_mcl): python src/158_fad_inf.py --tags ax_onehot_pfx16_ge10_hpcp ax_onehot_pfx16_ge10_hpcp_dose ax_onehot_pfx16_ge10_hpcp_dose2 --enc mert --out results/cell_fad_inf_mert"""
import argparse, json, os, sys, numpy as np
from scipy import linalg
sys.path.insert(0, "src")
from common import ROOT, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--enc", choices=["mert", "muq"], required=True); ap.add_argument("--out", required=True); ap.add_argument("--pca", type=int, default=32); ap.add_argument("--sizes", type=int, nargs="+", default=[36, 44, 52, 60, 72]); ap.add_argument("--reps", type=int, default=20); a = ap.parse_args()
rng = np.random.default_rng(0); DEC = list(LABELS["A"]); rows = manifest("A"); fn, L = ("A_acccodec_norm.npy", 22) if a.enc == "mert" else ("A_muq_acccodec_norm.npy", 9)
R = np.load(os.path.join(ROOT, "emb", fn))[:, L].astype(np.float64); lab = np.array([r["label"] for r in rows]); keep = np.isin(lab, DEC); R, lab = R[keep], lab[keep]
mu = R.mean(0); U, S, Vt = np.linalg.svd(R - mu, full_matrices=False); P = Vt[:a.pca].T; proj = lambda x: (x - mu) @ P
def fad(x, y):
    m1, m2 = x.mean(0), y.mean(0); c1, c2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False); cs = linalg.sqrtm(c1 @ c2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.trace(cs))
Rd = {d: proj(R[lab == d]) for d in DEC}
def fad_inf(Z, ref):
    xs, ys = [], []
    for n in a.sizes:
        n = min(n, len(Z))
        for _ in range(a.reps): xs.append(1.0 / n); ys.append(fad(Z[rng.choice(len(Z), n, replace=False)], ref))
    b, c = np.polyfit(xs, ys, 1); return float(c), float(fad(Z, ref))
out = {"enc": a.enc, "pca": a.pca, "sizes": a.sizes, "dirs": {}}; txt = [f"FAD_inf (Gui et al. extrapolation), {a.enc}, PCA {a.pca}: cell: FAD_inf (FAD at full n)"]
for g in a.tags:
    z = np.load(os.path.join(ROOT, "results", g + "_eval_emb.npz"), allow_pickle=True); names = [str(n) for n in z["names"]]; Z = proj(z[a.enc].astype(np.float64)); cond = np.array([n.split("|")[1] for n in names]); res = {}
    for c in sorted(set(cond), key=lambda c: (c != "neutral", c)):
        Zc = Z[cond == c]
        if len(Zc) < 60 or not (c in DEC or c == "neutral" or c.startswith("a")): continue
        refs = [c] if c in DEC else (["1960s", "2010s"] if c.startswith("a") else DEC)
        res[c] = {d: dict(zip(("fad_inf", "fad_full"), fad_inf(Zc, Rd[d]))) for d in refs}
        txt.append(f"{g} {c} (n={len(Zc)}): " + ", ".join(f"to real {d} {v['fad_inf']:.1f} ({v['fad_full']:.1f})" for d, v in res[c].items()))
    out["dirs"][g] = res
json.dump(out, open(a.out + ".json", "w"), indent=1); open(a.out + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
