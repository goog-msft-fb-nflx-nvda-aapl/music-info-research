"""B1 / M-5: does a culture-aware music encoder read the release market? Layer-wise mean-pooled embeddings of CultureMERT-95M (ntua-slp, continually pre-trained on Greek, Turkish, Carnatic and Hindustani music), dataset B (six markets, 1980s) on the original mix and on the separated accompaniment stem; logistic probe (C chosen by 5-fold CV on TRAIN) validation accuracy per layer (chance .167).
usage (from ~/era_market, main venv): python src/140_culturemert_market.py -> results/culturemert_market.{json,txt}, emb/B_cm_{model}_{cond}.npy"""
import os, sys, json, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "src"); from common import ROOT, LABELS, manifest
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lab = LABELS["B"]; rows = manifest("B"); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]
fe = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M"); out = {"chance": 1 / len(lab), "n_train": len(tr), "n_val": len(va), "models": {}}; txt = []
def wave(r, cond):
    if cond == "mix": y0, sr = sf.read(r["path"], dtype="float32"); y0 = y0.mean(1) if y0.ndim > 1 else y0
    else: y0, sr = np.load(f"{ROOT}/stems/B/{r['sample_id']}.npz")["accomp"].astype(np.float32), 24000
    return librosa.resample(y0, orig_sr=sr, target_sr=24000) if sr != 24000 else y0
for name, mid in [("culturemert", "ntua-slp/CultureMERT-95M")]:
    model = AutoModel.from_pretrained(mid, trust_remote_code=True).cuda().eval().half(); out["models"][name] = {}
    cap = []; [l.register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).float().mean(1)[0])) for l in model.encoder.layers]  # output_hidden_states is not returned by this remote code; hooks capture the 12 transformer-layer outputs (layers 1-12)
    for cond in ["mix", "acc"]:
        fp = f"{ROOT}/emb/B_cm_{name}_{cond}.npy"
        if os.path.exists(fp): X = np.load(fp)
        else:
            X = []
            for r in rows:
                inp = fe(wave(r, cond), sampling_rate=24000, return_tensors="pt")
                cap.clear()
                with torch.inference_mode(): model(input_values=inp["input_values"].cuda().half())
                X.append(torch.stack(cap).cpu().numpy())
            X = np.stack(X); np.save(fp, X)
        accs = []
        for L in range(X.shape[1]):
            F = X[:, L]; mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; F = (F - mu) / sd
            best = max([(cross_val_score(LogisticRegression(C=C, max_iter=2000), F[tr], y[tr], cv=5).mean(), C) for C in (0.001, 0.01, 0.1)])
            m = LogisticRegression(C=best[1], max_iter=2000).fit(F[tr], y[tr]); accs.append({"layer": L, "cv": float(best[0]), "C": best[1], "val_acc": float((m.predict(F[va]) == y[va]).mean())})
        b = max(accs, key=lambda a: a["cv"]); out["models"][name][cond] = {"layers": accs, "best_layer_by_cv": b}
        txt.append(f"{name} {cond}: best layer by CV {b['layer']} (cv {b['cv']:.3f}, C {b['C']}), validation acc {b['val_acc']:.3f}; best validation over layers {max(a['val_acc'] for a in accs):.3f} (chance {1/len(lab):.3f})"); print(txt[-1], flush=True)
json.dump(out, open(f"{ROOT}/results/culturemert_market.json", "w"), indent=1); open(f"{ROOT}/results/culturemert_market.txt", "w").write("\n".join(txt) + "\n")
