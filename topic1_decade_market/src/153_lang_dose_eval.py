"""Language dose evaluation (derived from 123_two_axis.py setup).
"""
import os, sys, json, numpy as np, soundfile as sf, torch, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
DEC = LABELS["A"]; rng = np.random.default_rng(0); T = json.load(open(f"{ROOT}/results/lyrics_scale_text_A.json")); ids = [s for s in T if os.path.exists(f"{ROOT}/ace_a_swap_voc/{s}_orig_s0.wav")]; LG = ["orig", "es", "pt", "de", "it"]; CODE = {"es": "es", "pt": "pt", "de": "de", "it": "it"}
pj = json.load(open(f"{ROOT}/results/probe_A.json"))["conds"]["mix_norm"]; L, C = pj["layer"], pj["C"]; rowsA = manifest("A"); yA = np.array([DEC.index(r["label"]) if r["label"] else -1 for r in rowsA]); spA = np.array([r["split"] for r in rowsA]); trA = np.where((spA == "train") & (yA >= 0))[0]; vaA = np.where((spA == "validation") & (yA >= 0))[0]
EA = np.load(f"{ROOT}/emb/A_mix_norm.npy"); clfA = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(EA[trA, L].astype(np.float32), yA[trA]); assert abs(float((clfA.predict(EA[vaA, L].astype(np.float32)) == yA[vaA]).mean()) - pj["val_acc"]) < 1e-6
pb = json.load(open(f"{ROOT}/results/probe_B.json"))["conds"]["voc_norm"]; LB, CB = pb["layer"], pb["C"]; MKB = LABELS["B"]; rowsB = manifest("B"); yB = np.array([MKB.index(r["label"]) if r["label"] else -1 for r in rowsB]); trB = np.where((np.array([r["split"] for r in rowsB]) == "train") & (yB >= 0))[0]
EB = np.load(f"{ROOT}/emb/B_voc_norm.npy"); clfB = make_pipeline(StandardScaler(), LogisticRegression(C=CB, max_iter=2000)).fit(EB[trB, LB].astype(np.float32), yB[trB])
proc, mert = load_mert(); wproc = WhisperProcessor.from_pretrained("openai/whisper-large-v3"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval(); tok = wproc.tokenizer; lang_ids = tok.convert_tokens_to_ids([f"<|{c}|>" for c in LANGUAGES]); lang_list = list(LANGUAGES); sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
import pyloudnorm as pyln; meter = pyln.Meter(SR)
def norm(w): l = meter.integrated_loudness(w.astype(np.float64)); return w if not np.isfinite(l) or np.abs(w).max() < 1e-4 else np.clip(pyln.normalize.loudness(w.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def emb(w, Lx):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return ((o.hidden_states[Lx] * m).sum(1) / m.sum(1).clamp_min(1) if m is not None else o.hidden_states[Lx].mean(1)).squeeze(0).float().cpu().numpy()
def lid(y24):
    f = wproc(librosa.resample(y24, orig_sr=SR, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.inference_mode(): lg = wm(input_features=f, decoder_input_ids=torch.tensor([[sot]]).cuda()).logits[0, -1]
doc = """Language dose: P(intended language) of the mix (Whisper large-v3 language token) and P(target market) of the isolated generated vocal (dataset-B vocal probe) as the fraction f of the lyrics in the target language rises 0, .25, .5, .75, 1 (target es -> Spain, de -> Germany), accompaniment = original. results/lang_dose.{json,txt}. usage (from ~/era_market, main venv): python src/153_lang_dose_eval.py"""
from scipy.stats import spearmanr
SEEDS = [0, 1, 2]; FR = [0, 25, 50, 75, 100]; MK = {"es": "Spain", "de": "Germany"}
def paths(s, L, f, sd):
    if f in (0, 100):
        c = "orig" if f == 0 else L; mix = "ace_a_swap" + ("" if sd == 0 else f"_sd{sd}"); return f"{ROOT}/{mix}/{s}_{c}_s{sd}.wav", f"{ROOT}/{mix}_voc/{s}_{c}_s{sd}.wav"
    return f"{ROOT}/ace_ld_swap/{s}_{L}{f}_s{sd}.wav", f"{ROOT}/ace_ld_swap_voc/{s}_{L}{f}_s{sd}.wav"
R = {}
for s in ids:
    z = np.load(f"{ROOT}/stems/A/{s}.npz"); sl = T[s].get("src_lang", "en")
    for L in ("es", "de"):
        for f in FR:
            for sd in SEEDS:
                pm, pv = paths(s, L, f, sd)
                if not (os.path.exists(pm) and os.path.exists(pv)): continue
                mix = sf.read(pm, dtype="float32")[0]; v = sf.read(pv, dtype="float32")[0]; n = len(z["vocals"]); v = np.pad(v, (0, max(0, n - len(v))))[:n]; mix = np.pad(mix, (0, max(0, n - len(mix))))[:n]
                p = lid(mix); pk = clfB.predict_proba(emb(norm(v), LB)[None].astype(np.float32))[0]
                R.setdefault((s, L, f), []).append((float(p[lang_list.index(L)]), float(p[lang_list.index(sl)]), float(pk[MKB.index(MK[L])])))
    print(s, flush=True)
rng = np.random.default_rng(0); out = {"cells": {}}; txt = [f"language dose, {len(ids)} songs, seeds {SEEDS}; fraction f of the lyrics in the target language"]
def boot(x): b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for L in ("es", "de"):
    mean_by_f = []
    for f in FR:
        S = [s for s in ids if (s, L, f) in R]; A = np.array([np.mean(R[(s, L, f)], 0) for s in S])
        if not len(S): continue
        out["cells"][f"{L}|{f}"] = {"n_songs": len(S), "P_target_language": float(A[:, 0].mean()), "P_target_language_ci": boot(A[:, 0]), "P_source_language": float(A[:, 1].mean()), "P_target_market_vocal": float(A[:, 2].mean()), "P_target_market_ci": boot(A[:, 2])}
        mean_by_f.append((f, A[:, 0].mean(), A[:, 2].mean())); c = out["cells"][f"{L}|{f}"]
        txt.append(f"{L} f={f}% (n={len(S)}): P({L}) {c['P_target_language']:.3f} [{c['P_target_language_ci'][0]:.3f},{c['P_target_language_ci'][1]:.3f}], P(source language) {c['P_source_language']:.3f}, P({MK[L]}) of the vocal {c['P_target_market_vocal']:.3f} [{c['P_target_market_ci'][0]:.3f},{c['P_target_market_ci'][1]:.3f}]")
    out[f"spearman_{L}"] = {"language": float(spearmanr([m[0] for m in mean_by_f], [m[1] for m in mean_by_f])[0]), "market": float(spearmanr([m[0] for m in mean_by_f], [m[2] for m in mean_by_f])[0])}; txt.append(f"{L}: Spearman(f, mean P(language)) {out[f'spearman_{L}']['language']:+.2f}, Spearman(f, mean P(market)) {out[f'spearman_{L}']['market']:+.2f}")
json.dump(out, open(f"{ROOT}/results/lang_dose.json", "w"), indent=1); open(f"{ROOT}/results/lang_dose.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
