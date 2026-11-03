"""T-4a evaluation: does a timbre-only voice conversion (108_svc_market.py) move the market score of a vocal stem? Converted vocals (svc_B/<id>_<market>.wav, 102 dataset-B validation clips x 6 market voices) are loudness-normalized to -23 LUFS and embedded with MERT-v2-30s (layer of probe_B.json voc_norm) and MuQ (layer of probe_muq_B.json voc_norm), the same pooling as 02_embed.py / 02b_embed_muq.py. Probes: StandardScaler + LogisticRegression(C from the probe json) fit on dataset-B TRAIN vocal embeddings (emb/B_voc_norm.npy, emb/B_muq_voc_norm.npy), as 04_probe.py; sanity: validation accuracy on the original vocals must equal the probe json.
Reports per encoder: (a) accuracy on the original vocals, (b) accuracy of predicting the ORIGINAL market label from converted vocals (mean over the six voices), (c) accuracy of predicting the TARGET voice market, (d) per target market m: Delta_m = P(m | converted to m) - P(m | converted to the other five voices), per clip, mean with song-bootstrap 95% CI, (e) P(original label) change = P(own label | converted, mean over voices) - P(own label | original).
usage (from ~/era_market, main venv, one GPU): python src/109_svc_market_eval.py -> results/svc_market_eval.{json,txt}"""
import os, sys, json, numpy as np, torch, soundfile as sf, pyloudnorm as pyln
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
from muq_util import load_muq
MK = LABELS["B"]; rows = manifest("B"); y = np.array([MK.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "validation") & (y >= 0))[0]; ids = [rows[i]["sample_id"] for i in va]
meter = pyln.Meter(SR); rng = np.random.default_rng(0)
def norm(w, t=-23.0):
    if np.abs(w).max() < 1e-4: return w
    l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, t), -1, 1).astype(np.float32)
proc, mert = load_mert(); muq = load_muq()
def emb_mert(w, L):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[L] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[L].mean(1)).squeeze(0).float().cpu().numpy()
def emb_muq(w, L):
    with torch.inference_mode(): o = muq(torch.from_numpy(w)[None].cuda(), output_hidden_states=True)
    return o["hidden_states"][L].float().mean(1).squeeze(0).cpu().numpy()
cfg = {"mert": (json.load(open(f"{ROOT}/results/probe_B.json"))["conds"]["voc_norm"], np.load(f"{ROOT}/emb/B_voc_norm.npy"), emb_mert), "muq": (json.load(open(f"{ROOT}/results/probe_muq_B.json"))["conds"]["voc_norm"], np.load(f"{ROOT}/emb/B_muq_voc_norm.npy"), emb_muq)}
out = {}; txt = []; cache = {}; SELF = os.environ.get("SELF") == "1"; DIR = "svc_B_self" if SELF else os.environ.get("SVCDIR", "svc_B"); TAG = "_self" if SELF else ("" if DIR == "svc_B" else "_" + DIR.replace("svc_B_", ""))
for enc, (pj, E, fn) in cfg.items():
    L, C = pj["layer"], pj["C"]; clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E[tr, L].astype(np.float32), y[tr])
    acc0 = float((clf.predict(E[va, L].astype(np.float32)) == y[va]).mean()); assert abs(acc0 - pj["val_acc"]) < 1e-6, (enc, acc0, pj["val_acc"])
    P0 = clf.predict_proba(E[va, L].astype(np.float32)); Pc = np.zeros((len(va), 6, 6))   # clip, voice, class
    for vi, sid in enumerate(ids):
        for mi, m in enumerate(["self"] if SELF else MK):
            w = norm(sf.read(f"{ROOT}/{DIR}/{sid}_{m}.wav", dtype="float32")[0])
            Pc[vi, mi] = clf.predict_proba(fn(w, L)[None].astype(np.float32))[0]
    if SELF:
        yv = y[va]; res = {"layer": L, "C": C, "val_acc_original": acc0, "acc_original_label_from_self_converted": float((Pc[:, 0].argmax(1) == yv).mean()), "delta_P_original_label": float(np.mean([Pc[i, 0, yv[i]] - P0[i, yv[i]] for i in range(len(va))]))}
        out[enc] = res; txt.append(f"{enc} self-voice control: original-vocal acc {acc0:.3f}; self-converted -> original label acc {res['acc_original_label_from_self_converted']:.3f}; delta P(own label) {res['delta_P_original_label']:+.3f}"); continue
    yv = y[va]; acc_orig_lab = float(np.mean([(Pc[:, mi].argmax(1) == yv).mean() for mi in range(6)])); acc_target = float(np.mean([(Pc[:, mi].argmax(1) == mi).mean() for mi in range(6)]))
    dP_own = float(np.mean([Pc[i, :, yv[i]].mean() - P0[i, yv[i]] for i in range(len(va))])); res = {"layer": L, "C": C, "val_acc_original": acc0, "acc_original_label_from_converted": acc_orig_lab, "acc_target_voice_market": acc_target, "delta_P_original_label": dP_own, "delta_target": {}}
    for mi, m in enumerate(MK):
        d = np.array([Pc[i, mi, mi] - np.delete(Pc[i, :, mi], mi).mean() for i in range(len(va))]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; res["delta_target"][m] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    out[enc] = res; txt.append(f"{enc} (layer {L}, C {C}): original-vocal val acc {acc0:.3f}; converted -> original label acc {acc_orig_lab:.3f}; converted -> target voice market acc {acc_target:.3f} (chance .167); delta P(own label) {dP_own:+.3f}; delta P(target | own-voice vs other voices) " + " ".join(f"{m} {v['delta']:+.3f}[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for m, v in res["delta_target"].items()))
    np.save(f"{ROOT}/results/svc_market_Pconv_{enc}.npy", Pc)
json.dump(out, open(f"{ROOT}/results/svc_market_eval{TAG}.json", "w"), indent=1); open(f"{ROOT}/results/svc_market_eval{TAG}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
