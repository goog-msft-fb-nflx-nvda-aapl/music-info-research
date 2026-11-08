"""Combined era x language dose grid (derived from 123_two_axis.py setup; see the doc string below)."""
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
    return torch.softmax(lg[lang_ids].float(), -1).cpu().numpy()
L_ = L   # MERT layer of the era probe (L is reused below for the target language)
doc = """Combined dose grid (2D control plane): era dose a in (0, .3, .5, .7, 1) of the one-hot+16 adapter (accompaniment, HPCP chroma) x language dose f in (0, 25, 50, 75, 100 %) of the target language (es, de) in the lyric-route vocal; accompaniment seed s with vocal seed s (0-2), 16 songs. Read-outs per cell: era = expected decade index of the mix-level MERT era probe (0 = 1960s, 5 = 2010s), language = Whisper large-v3 P(target language) of the mix, market = dataset-B vocal probe P(target market) of the isolated vocal. Effects with song-bootstrap 95 % CI: era effect (a = 1 minus a = 0) at each f, language effect (f = 100 minus f = 0) and market effect at each a, and Spearman of each read-out along both axes. results/two_dose_grid.{json,txt}. usage (from ~/era_market, main venv): python src/155_two_dose_grid.py"""
from scipy.stats import spearmanr
SEEDS = [0, 1, 2]; AS = ["a0", "a0.3", "a0.5", "a0.7", "a1"]; FR = [0, 25, 50, 75, 100]; MK = {"es": "Spain", "de": "Germany"}; ACCD = ["gen_ax_onehot_pfx16_ge10_hpcp_gridA", "gen_ax_onehot_pfx16_ge10_hpcp_gridB"]
def accpath(s, a, sd):
    for d in ACCD:
        p = f"{ROOT}/{d}/{s}_{a}_s{sd}_acc.wav"
        if os.path.exists(p): return p
def vpaths(s, L, f, sd):
    if f in (0, 100):
        c = "orig" if f == 0 else L; m = "ace_a_swap_voc" if sd == 0 else f"ace_a_swap_sd{sd}_voc"; return f"{ROOT}/{m}/{s}_{c}_s{sd}.wav"
    return f"{ROOT}/ace_ld_swap_voc/{s}_{L}{f}_s{sd}.wav"
R = {}
for s in ids:
    z = np.load(f"{ROOT}/stems/A/{s}.npz"); n = len(z["vocals"]); sl = T[s].get("src_lang", "en")
    for L in ("es", "de"):
        for f in FR:
            for sd in SEEDS:
                pv = vpaths(s, L, f, sd)
                if not os.path.exists(pv): continue
                v = sf.read(pv, dtype="float32")[0]; v = np.pad(v, (0, max(0, n - len(v))))[:n]; pk = clfB.predict_proba(emb(norm(v), LB)[None].astype(np.float32))[0][MKB.index(MK[L])]
                for a_ in AS:
                    pa = accpath(s, a_, sd)
                    if pa is None: continue
                    acc = sf.read(pa, dtype="float32")[0][:n]; acc = np.pad(acc, (0, max(0, n - len(acc)))); mix = np.clip(v + acc, -1, 1)
                    pe = clfA.predict_proba(emb(norm(mix), L_)[None].astype(np.float32))[0]; pl = lid(mix)[lang_list.index(L)]
                    R.setdefault((s, L, f, a_), []).append((float(pe @ np.arange(6)), float(pl), float(pk)))
    print(s, flush=True)
rng = np.random.default_rng(0); out = {"cells": {}, "effects": {}}; txt = [f"two-dose grid, {len(ids)} songs, seeds {SEEDS}; rows: era dose a, columns: language share f; entries: era index / P(language) / P(market)"]
def boot(x): x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def cell(L, f, a_):
    S = [s for s in ids if (s, L, f, a_) in R]; return S, np.array([np.mean(R[(s, L, f, a_)], 0) for s in S])
for L in ("es", "de"):
    txt.append(f"target {L}")
    for a_ in AS:
        row = []
        for f in FR:
            S, A = cell(L, f, a_)
            if len(S): out["cells"][f"{L}|{a_}|{f}"] = {"n_songs": len(S), "era_index": float(A[:, 0].mean()), "P_language": float(A[:, 1].mean()), "P_market": float(A[:, 2].mean())}; row.append(f"{A[:, 0].mean():.2f}/{A[:, 1].mean():.2f}/{A[:, 2].mean():.2f}")
        txt.append(f"  {a_}: " + "  ".join(row))
    for f in FR:
        S = [s for s in ids if all((s, L, f, a_) in R for a_ in ("a0", "a1"))]
        if len(S) < 5: continue
        d = np.array([np.mean(R[(s, L, f, "a1")], 0)[0] - np.mean(R[(s, L, f, "a0")], 0)[0] for s in S]); out["effects"][f"{L}|era_effect|f{f}"] = {"n": len(S), "mean": float(d.mean()), "ci": boot(d)}; txt.append(f"  era index effect a1-a0 at f={f}%: {d.mean():+.2f} [{boot(d)[0]:+.2f},{boot(d)[1]:+.2f}] (n={len(S)})")
    for a_ in AS:
        S = [s for s in ids if all((s, L, f, a_) in R for f in (0, 100))]
        if len(S) < 5: continue
        for k, nm in ((1, "P(language)"), (2, "P(market)")):
            d = np.array([np.mean(R[(s, L, 100, a_)], 0)[k] - np.mean(R[(s, L, 0, a_)], 0)[k] for s in S]); out["effects"][f"{L}|{nm}|lang_effect|{a_}"] = {"n": len(S), "mean": float(d.mean()), "ci": boot(d)}; txt.append(f"  {nm} effect f=100%-0% at era dose {a_}: {d.mean():+.3f} [{boot(d)[0]:+.3f},{boot(d)[1]:+.3f}] (n={len(S)})")
    mat = {k: np.array([[cell(L, f, a_)[1][:, j].mean() for f in FR] for a_ in AS]) for j, k in enumerate(("era", "lang", "market"))}
    av = np.arange(5); fv = np.array(FR)
    txt.append(f"  Spearman along era dose: era {spearmanr(av, mat['era'].mean(1))[0]:+.2f}, language {spearmanr(av, mat['lang'].mean(1))[0]:+.2f}, market {spearmanr(av, mat['market'].mean(1))[0]:+.2f}; along language share: era {spearmanr(fv, mat['era'].mean(0))[0]:+.2f}, language {spearmanr(fv, mat['lang'].mean(0))[0]:+.2f}, market {spearmanr(fv, mat['market'].mean(0))[0]:+.2f}")
json.dump(out, open(f"{ROOT}/results/two_dose_grid.json", "w"), indent=1); open(f"{ROOT}/results/two_dose_grid.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
