"""T-10b (derived from 123_two_axis.py): the same composition on the full grid: accompaniment from the one-hot+16 adapter with HPCP chroma (ACCDIR, default gen_ax_onehot_pfx16_ge10_hpcp), all six decades plus neutral, language read-out for every decade, results/two_axis_grid.{json,txt}. usage (from ~/era_market, main venv, one GPU): python src/151_two_axis_grid.py

Original docstring: T-10 two-axis composition on the 24 evaluation songs of dataset A: language axis from the lyric route (translated lyrics, ACE-Step cover mode strength 0.95 on the original mix, vocal separated with HT Demucs, RMS-matched to the original vocal: `ace_a_swap_voc/<id>_<lang>_s0.wav`, languages orig, es, pt, de, it) x era axis from the era-axis adapter (accompaniment of `gen_ax_ctrlc_ge10`, seed 0, decades 1960s, 1980s, 2010s, plus the neutral accompaniment). Remix = lyric-route vocal + era-axis accompaniment (both at the level of the original stems). Read-outs on the remix (cells = language x decade): (a) mix-level era probe (MERT layer 12, probe_A.json 'mix_norm'): P(decade); (b) Whisper large-v3 language-token P(intended language) of the remix; (c) vocal market probe (dataset-B vocal probe, MERT layer 16) on the generated vocal alone (does not depend on the decade). Main effects and the interaction of the 2x2 design (decade: neutral vs era; language: orig vs target) are computed per decade and language as differences of differences with song-bootstrap 95% CI (2000 resamples).
usage (from ~/era_market, main venv, one GPU): python src/123_two_axis.py -> results/two_axis.{json,txt}"""
import os, sys, json, numpy as np, soundfile as sf, torch, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
DEC = LABELS["A"]; rng = np.random.default_rng(0); T = json.load(open(f"{ROOT}/results/lyrics_scale_text_A.json")); ids = [s for s in T if all(os.path.exists(f"{ROOT}/ace_a_swap_voc/{s}_{c}_s0.wav") for c in ("orig",))]; LG = ["orig", "es", "pt", "de", "it"]; CODE = {"es": "es", "pt": "pt", "de": "de", "it": "it"}
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
    return torch.softmax(lg[lang_ids].float(), -1).cpu().numpy()
R = {}; RS = {}; SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]; OUTN = os.environ.get("OUTN", "two_axis_grid"); DECS = ["neutral"] + list(DEC); ACCDIR = os.environ.get("ACCDIR", "gen_ax_onehot_pfx16_ge10_hpcp")
for s in ids:
    z = np.load(f"{ROOT}/stems/A/{s}.npz"); voc0 = z["vocals"].astype(np.float32); sl = T[s].get("src_lang", "en")
    for sd in SEEDS:
        vd = "ace_a_swap_voc" if sd == 0 else f"ace_a_swap_sd{sd}_voc"
        for lg in LG:
            fp = f"{ROOT}/{vd}/{s}_{lg}_s{sd}.wav"
            if not os.path.exists(fp): continue
            v = sf.read(fp, dtype="float32")[0][:len(voc0)]; v = np.pad(v, (0, len(voc0) - len(v))); RS.setdefault((s, lg, "vocmarket"), []).append(clfB.predict_proba(emb(norm(v), LB)[None].astype(np.float32))[0])
            for d in DECS:
                acc = sf.read(f"{ROOT}/{ACCDIR}/{s}_{d}_s{sd}_acc.wav", dtype="float32")[0][:len(voc0)]; mix = np.clip(v + acc, -1, 1); nm = norm(mix)
                RS.setdefault((s, lg, d, "era"), []).append(clfA.predict_proba(emb(nm, L)[None].astype(np.float32))[0]); code = sl if lg == "orig" else lg
                RS.setdefault((s, lg, d, "lang"), []).append(float(lid(mix)[lang_list.index(code)]))
    print(s, flush=True)
R = {k: np.mean(v, 0) if k[-1] != "lang" else float(np.mean(v)) for k, v in RS.items()}   # per-song mean over the generation seeds present
def boot(d): bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {"cells": {}, "effects": {}}; txt = [f"two-axis composition, {len(ids)} songs of dataset A; mix-level era probe MERT layer {L} (val acc {pj['val_acc']:.3f})"]
for lg in LG:
    for d in DECS:
        S = [s for s in ids if (s, lg, d, "era") in R]
        if S: out["cells"][f"{lg}|{d}"] = {"n": len(S), **{f"P_{e}": float(np.mean([R[(s, lg, d, "era")][DEC.index(e)] for s in S])) for e in DEC}}
for e in DEC:
    k = DEC.index(e)
    for lg in LG[1:]:
        S = [s for s in ids if (s, lg, e, "era") in R and (s, "orig", e, "era") in R and (s, lg, "neutral", "era") in R and (s, "orig", "neutral", "era") in R]
        if len(S) < 5: continue
        era_main_orig = np.array([R[(s, "orig", e, "era")][k] - R[(s, "orig", "neutral", "era")][k] for s in S]); era_main_lg = np.array([R[(s, lg, e, "era")][k] - R[(s, lg, "neutral", "era")][k] for s in S]); inter = era_main_lg - era_main_orig
        out["effects"][f"era_{e}|lang_{lg}"] = {"n": len(S), "era_effect_source_language": float(era_main_orig.mean()), "era_effect_target_language": float(era_main_lg.mean()), "interaction": float(inter.mean()), "interaction_ci": boot(inter)}
        txt.append(f"era {e}, language {lg} (n={len(S)}): P({e}) gain from the era-axis accompaniment with the source-language vocal {era_main_orig.mean():+.3f}, with the {lg} vocal {era_main_lg.mean():+.3f}; interaction (target minus source) {inter.mean():+.3f} [{boot(inter)[0]:+.3f},{boot(inter)[1]:+.3f}]")
for lg in LG[1:]:
    for d in DECS:
        S = [s for s in ids if (s, lg, d, "lang") in R and (s, "orig", d, "lang") in R]
        if len(S) < 5: continue
        a = np.array([R[(s, lg, d, "lang")] for s in S]); b = np.array([R[(s, "orig", d, "lang")] for s in S]); out["effects"][f"lang_{lg}|accomp_{d}"] = {"n": len(S), "P_intended_language": float(a.mean()), "P_source_language_control": float(b.mean())}
        txt.append(f"language {lg} with {d} accompaniment (n={len(S)}): P(intended language) {a.mean():.3f} (source-language control {b.mean():.3f})")
json.dump(out, open(f"{ROOT}/results/{OUTN}.json", "w"), indent=1); open(f"{ROOT}/results/{OUTN}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
