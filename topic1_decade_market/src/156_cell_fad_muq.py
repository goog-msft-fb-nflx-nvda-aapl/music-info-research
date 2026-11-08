"""MuQ variant (layer 9 = codec-matched MuQ probe layer, results/probe_codec_muq_A.json; reference emb/A_muq_acccodec_norm.npy). Derived from the MERT variant of 152_cell_fad.py: embedding = MERT layer 22 (codec-matched probe layer) of the generated accompaniments, read from results/<tag>_eval_emb.npz (names clip|cond|seed, written by 10_eval_gen.py), reference = MERT layer 22 of the codec-resynthesised real accompaniments of the dataset-A clips of the target decade (emb/A_acccodec_norm.npy; layer 22 of the eval embedding equals this array, checked on one clip, correlation 1.0000). usage (from ~/era_market, venv_mcl): python src/154_cell_fad_mert.py --tags ax_onehot_pfx16_ge10_hpcp ax_onehot_pfx16_ge10_hpcp_dose ax_onehot_pfx16_ge10_hpcp_dose2 --out results/cell_fad_mert

Original docstring: Per-cell Fréchet audio distance (Kilgour et al., arXiv 1812.08466; existing metric, only the embedding and the reference set are chosen here). Embedding: LAION-CLAP htsat-unfused audio embedding (512-d, L2-normalised), generated accompaniments from emb_clap/<gendir>.npz (150_archive_round.py), reference = real accompaniment CLAP embeddings of the dataset-A clips of the target decade (emb/A_clap_acc_norm.npy, manifest order, labelled clips only). With 72 generated clips per cell the 512-d covariance is rank deficient, so both sets are projected on the first PCA_DIM principal components of the reference clips of all decades (default 32) before the Fréchet distance. Reported per generation directory and target decade: FAD(generated cell, real target decade), FAD(neutral generated cell, real target decade) and the difference (negative = closer to the target decade than the unconditioned generation); floor = FAD between two random halves of the real target decade.
usage (from ~/era_market, venv_mcl): python src/152_cell_fad.py --gendirs gen_ax_onehot_pfx16_ge10_hpcp gen_ax_ctrlc_ge10_hpcp --out results/cell_fad_clap"""
import argparse, json, os, sys, numpy as np
from scipy import linalg
sys.path.insert(0, "src")
from common import ROOT, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pca", type=int, default=32); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); DEC = list(LABELS["A"]); rows = manifest("A"); R = np.load(os.path.join(ROOT, "emb", "A_muq_acccodec_norm.npy"))[:, 9].astype(np.float64); lab = np.array([r["label"] for r in rows]); keep = np.isin(lab, DEC); R, lab = R[keep], lab[keep]
mu = R.mean(0); U, S, Vt = np.linalg.svd(R - mu, full_matrices=False); P = Vt[:a.pca].T
def proj(x): return (x - mu) @ P
def fad(x, y):
    m1, m2 = x.mean(0), y.mean(0); c1, c2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False); cs, _ = linalg.sqrtm(c1 @ c2, disp=False); cs = cs.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.trace(cs))
Rd = {d: proj(R[lab == d]) for d in DEC}
floor = {}
for d in DEC:
    x = Rd[d]; f = []
    for _ in range(20): p = rng.permutation(len(x)); h = len(x) // 2; f.append(fad(x[p[:h]], x[p[h:2 * h]]))
    floor[d] = float(np.mean(f))
out = {"embedding": "MuQ codec-matched probe layer (1024-d)", "pca_dim": a.pca, "real_halves_floor": floor, "dirs": {}}; txt = [f"per-cell FAD, MuQ, PCA {a.pca}; real-half floor " + " ".join(f"{d} {floor[d]:.2f}" for d in DEC)]
for g in a.tags:
    z = np.load(os.path.join(ROOT, "results", g + "_eval_emb.npz"), allow_pickle=True); names = [str(n) for n in z["names"]]; E = z["muq"].astype(np.float64); cond = np.array([n.split("|")[1] for n in names]); Z = proj(E)
    conds = sorted(set(cond), key=lambda c: (c != "neutral", c)); res = {}
    for c in conds:
        zc = Z[cond == c]; res[c] = {"n": int(len(zc)), **{"fad_vs_" + d: fad(zc, Rd[d]) for d in DEC}}
    out["dirs"][g] = res
    for c in conds:
        if c in DEC and "neutral" in res:
            dn = res[c]["fad_vs_" + c] - res["neutral"]["fad_vs_" + c]; res[c]["fad_target_minus_neutral"] = float(dn)
            txt.append(f"{g} target {c} (n={res[c]['n']}): FAD to real {c} {res[c]['fad_vs_' + c]:.2f}, neutral generation to real {c} {res['neutral']['fad_vs_' + c]:.2f}, difference {dn:+.2f}")
        elif c.startswith("a"):
            txt.append(f"{g} dose {c} (n={res[c]['n']}): FAD to real 1960s {res[c]['fad_vs_1960s']:.2f}, to real 2010s {res[c]['fad_vs_2010s']:.2f}")
json.dump(out, open(a.out + ".json", "w"), indent=1); open(a.out + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
