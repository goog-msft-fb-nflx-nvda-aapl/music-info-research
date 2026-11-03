"""Evaluate VA-conditioned generations with an independent audio evaluator.
Usage: python -I gen_eval.py <out_json> <gpu> <run1> [run2 ...]   (runs = names of data/feat/gen_<run>.npz; 'base' = unconditioned model with null VA)
Evaluator: MERT-v1-95M layer-6 time-mean features + ridge (lambda 1) trained on the PMEmo TRAIN songs only (same song split as sao_va.py, seed 0).
Reported per run: Pearson r / CCC / Spearman between target VA and evaluator-predicted VA over all generated clips (9 cells x n clips), mean predicted VA per cell,
slope of predicted on target, and a Frechet distance (PCA-16 of MERT layer-6 features, fitted on real test songs) to the held-out real clips.
Reference rows: the evaluator on real held-out PMEmo test songs (30 s chorus clips) vs their labels."""
import sys, os, json, glob, numpy as np, pandas as pd, torch, librosa
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import sqrtm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
H = os.path.expanduser("~/music_therapy"); out, gpu, runs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]; dev = f"cuda:{gpu}"; L = 6
sys.argv = [sys.argv[0], "x", "x", "0"]                                         # reuse split helper without running sao_va main
ann = pd.read_csv(f"{H}/data/pmemo/x/PMEmo2019/annotations/static_annotations.csv").set_index("musicId")
ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(f"{H}/data/pmemo/x/PMEmo2019/chorus/*.mp3")); ids = [i for i in ids if i in ann.index]
rng = np.random.default_rng(0); perm = rng.permutation(len(ids)); te = set(np.array(ids)[perm[:int(0.15 * len(ids))]].tolist())
P = np.load(f"{H}/data/feat/mert_pmemo.npz"); fid = [int(f.split(".")[0]) for f in P["files"]]; sel = [k for k, i in enumerate(fid) if i in ann.index]
X = P["F"][sel][:, L]; sid = np.array([fid[k] for k in sel]); Y = ann.loc[sid][["Valence(mean)", "Arousal(mean)"]].values; ist = np.isin(sid, list(te))
mu, sd = X[~ist].mean(0), X[~ist].std(0) + 1e-6
def fit(lam=1.0):
    A = (X[~ist] - mu) / sd; ym = Y[~ist].mean(0); W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y[~ist] - ym)); return lambda Z: ((Z - mu) / sd) @ W + ym
pred = fit()
def ccc(a, b): return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))
def stats(t, p):
    out_ = {}
    rg = np.random.default_rng(0)
    for j, k in enumerate(("valence", "arousal")):
        bs = [rg.integers(0, len(t), len(t)) for _ in range(1000)]
        rs = [pearsonr(t[b, j], p[b, j])[0] for b in bs if p[b, j].std() > 0 and t[b, j].std() > 0]; sl = [np.polyfit(t[b, j], p[b, j], 1)[0] for b in bs if t[b, j].std() > 0]
        out_[k] = dict(r=float(pearsonr(t[:, j], p[:, j])[0]), rho=float(spearmanr(t[:, j], p[:, j])[0]), ccc=ccc(t[:, j], p[:, j]), slope=float(np.polyfit(t[:, j], p[:, j], 1)[0]),
                       r_ci=[float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))], slope_ci=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))])
    return out_
res = {"reference_real_test": stats(Y[ist], pred(X[ist])), "n_real_test": int(ist.sum())}
proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True); m = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(dev).eval()
def feats(audio, sr):
    F = []
    for a in audio:
        x = librosa.resample(a.astype(np.float32).mean(0), orig_sr=sr, target_sr=24000); inp = proc(x, sampling_rate=24000, return_tensors="pt").to(dev)
        with torch.no_grad(): h = m(**inp, output_hidden_states=True).hidden_states[L]
        F.append(h.mean(1)[0].cpu().numpy())
    return np.stack(F)
real = X[ist]; pc = real - real.mean(0); U = np.linalg.svd(pc, full_matrices=False)[2][:16].T
def fd(a, b):
    a, b = (a - real.mean(0)) @ U, (b - real.mean(0)) @ U; m1, m2 = a.mean(0), b.mean(0); c1, c2 = np.cov(a.T), np.cov(b.T)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * np.real(sqrtm(c1 @ c2))))
for r in runs:
    d = np.load(f"{H}/data/feat/gen_{r}.npz"); F = feats(d["audio"].astype(np.float32), int(d["sr"])); p = pred(F); t = d["va"]
    cells = {f"{a:.1f}/{b:.1f}": p[(np.abs(t[:, 0] - a) < 1e-6) & (np.abs(t[:, 1] - b) < 1e-6)].mean(0).round(3).tolist() for a in (0.2, 0.5, 0.8) for b in (0.2, 0.5, 0.8)}
    res[r] = {"n": int(len(t)), **stats(t, p), "cell_means_(valence,arousal)_by_target_v/a": cells, "fd_pca16_to_real_test": fd(F, real), "pred_mean": p.mean(0).round(3).tolist(), "pred_std": p.std(0).round(3).tolist()}
    print(r, json.dumps({k: res[r][k] for k in ("n", "valence", "arousal", "fd_pca16_to_real_test")}))
json.dump(res, open(out, "w"), indent=1); print("reference", json.dumps(res["reference_real_test"]))
