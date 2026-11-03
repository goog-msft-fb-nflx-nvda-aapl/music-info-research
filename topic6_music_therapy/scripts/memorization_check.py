"""Nearest-neighbour check for memorisation: cosine similarity (MERT-v1-95M layer-6 time-mean features) of generated clips to PMEmo TRAIN songs, versus
held-out real test songs to the same TRAIN songs. Usage: python -I memorization_check.py <out_json> <gpu> <gen_run1> [...]   (same song split as sao_va.py)"""
import sys, os, json, glob, numpy as np, pandas as pd, torch, librosa
from transformers import AutoModel, Wav2Vec2FeatureExtractor
H = os.path.expanduser("~/music_therapy"); out, gpu, runs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]; dev = f"cuda:{gpu}"; L = 6
ann = pd.read_csv(f"{H}/data/pmemo/x/PMEmo2019/annotations/static_annotations.csv").set_index("musicId")
ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(f"{H}/data/pmemo/x/PMEmo2019/chorus/*.mp3")); ids = [i for i in ids if i in ann.index]
rng = np.random.default_rng(0); perm = rng.permutation(len(ids)); te = set(np.array(ids)[perm[:int(0.15 * len(ids))]].tolist())
P = np.load(f"{H}/data/feat/mert_pmemo.npz"); fid = np.array([int(f.split(".")[0]) for f in P["files"]]); ok = np.array([i in ann.index for i in fid]); X = P["F"][ok][:, L]; sid = fid[ok]; ist = np.isin(sid, list(te))
mu, sd = X[~ist].mean(0), X[~ist].std(0) + 1e-6; nz = lambda a: ((a - mu) / sd) / np.linalg.norm((a - mu) / sd, axis=1, keepdims=True)
tr, tst = nz(X[~ist]), nz(X[ist]); res = {"heldout_real_to_train": {"max_cos_mean": float((tst @ tr.T).max(1).mean()), "max_cos_p95": float(np.percentile((tst @ tr.T).max(1), 95)), "max_cos_max": float((tst @ tr.T).max())}}
proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True); m = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(dev).eval()
for r in runs:
    d = np.load(f"{H}/data/feat/gen_{r}.npz"); F = []
    for a in d["audio"].astype(np.float32):
        x = librosa.resample(a.mean(0), orig_sr=int(d["sr"]), target_sr=24000); inp = proc(x, sampling_rate=24000, return_tensors="pt").to(dev)
        with torch.no_grad(): F.append(m(**inp, output_hidden_states=True).hidden_states[L].mean(1)[0].cpu().numpy())
    s = (nz(np.stack(F)) @ tr.T).max(1); res[r] = {"max_cos_mean": float(s.mean()), "max_cos_p95": float(np.percentile(s, 95)), "max_cos_max": float(s.max())}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
