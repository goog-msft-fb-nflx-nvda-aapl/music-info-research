"""T-4p stage 1: candidate style words per market, generated on dataset-B TRAIN clips. For each market m the candidate words are the style names of top_lift and top_kl of results/market_styles.json for (m, 1980s) (deduplicated, at most 14). Prompt "a <adjective> recording in the style <word>"; 6 TRAIN clips (the first TRAIN clip by sample id of each market), MusicGen-Melody CFG 3 with the chroma of the clip's mix, seed 0, plus the neutral prompt. Output GENDIR(=gen_dsel)/<id>_<market>_<word>_s0_acc.wav and candidates.json. Selection in 129_dsel_pick.py.
usage (from ~/era_market, main venv): GENDIR=gen_dsel python src/128_dsel_gen.py [--rot k] (parallel workers)"""
import argparse, os, sys, json, re, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, "src")
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
ADJ = {"US": "American", "UK": "British", "Brazil": "Brazilian", "Spain": "Spanish", "Germany": "German", "Italy": "Italian"}; MK = list(ADJ)
MS = json.load(open(f"{ROOT}/results/market_styles.json")); CAND = {}
for m in MK:
    c = MS[f"{m}|1980s"]; names = []
    for x in c["top_lift"] + c.get("top_kl", []):
        w = x["style"].split("---")[1]
        if w not in names: names.append(w)
    CAND[m] = names[:14]
ap = argparse.ArgumentParser(); ap.add_argument("--rot", type=int, default=0); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen_dsel")); os.makedirs(G, exist_ok=True); json.dump(CAND, open(os.path.join(G, "candidates.json"), "w"), indent=1)
rows = manifest("B"); pick = [sorted([r for r in rows if r["split"] == "train" and r["label"] == m], key=lambda r: r["sample_id"])[0] for m in MK]
proc = AutoProcessor.from_pretrained("facebook/musicgen-melody"); model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody", torch_dtype=torch.float16).to("cuda").eval()
jobs = [(r, "neutral", "neutral", "a song recording") for r in pick] + [(r, m, w, f"a {ADJ[m]} recording in the style {w}") for r in pick for m in MK for w in CAND[m]]; jobs = jobs[a.rot:] + jobs[:a.rot]; n = 0
for r, m, w, prompt in jobs:
    fp = os.path.join(G, f"{r['sample_id']}_{m}_{re.sub(r'[^A-Za-z0-9]', '', w)}_s0_acc.wav")
    if os.path.exists(fp): continue
    mix = sf.read(r["path"], dtype="float32")[0]; m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy(); acc0 = np.load(os.path.join(ROOT, "stems", "B", r["sample_id"] + ".npz"))["accomp"].astype(np.float32)
    inp = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}; torch.manual_seed(0)
    with torch.no_grad(): out = model.generate(**inp, do_sample=True, guidance_scale=3.0, max_new_tokens=1500)
    g = AF.resample(out[0, 0].float().cpu(), 32000, SR).numpy()[:len(acc0)]; g = np.pad(g, (0, len(acc0) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9)); sf.write(fp, g, SR); n += 1; print(r["sample_id"], m, w, n, flush=True)
print("done", n)
