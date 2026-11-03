"""T-4e: market conditioning with data-derived style descriptors. MusicGen-Melody (facebook/musicgen-melody, fp16, CFG 3, chroma of the ORIGINAL MIX as in 07_generate.py) regenerates the accompaniment of the dataset-B validation clips (4 per market, 24 clips, first by manifest order) with (a) neutral prompt, (b) 'hw_<market>' = the hand-written market prompts of 07_generate.py (MARKETS), (c) 'dk_<market>' = the same template with the three styles of largest P x log(lift) (top_kl of 113), 'ds_<market>' = prompts built from the Discogs-VI metadata: "a <market adjective> recording in the styles <S1>, <S2> and <S3>" with the three styles of highest lift in (market, 1980s) from results/market_styles.json (frequency >= 200 versions). Output GENDIR/<id>_<cond>_s<seed>_{acc,remix}.wav (remix = original vocal + generated accompaniment RMS-matched), picked.txt.
usage (from ~/era_market, main venv): GENDIR=gen_mstyle python src/114_market_style_gen.py --seeds 0 1 [--reverse]   (two workers: with and without --reverse)"""
import argparse, os, sys, json, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, "src")
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest, LABELS
ADJ = {"US": "American", "UK": "British", "Brazil": "Brazilian", "Spain": "Spanish", "Germany": "German", "Italy": "Italian"}
HW = {"US": "an American 1980s pop and rock recording, gated reverb drums, electric guitars, synthesizers, radio-friendly production", "UK": "a British 1980s new wave and synth pop recording, drum machine, analog synthesizers, jangly guitar, dry English production",
      "Brazil": "a Brazilian 1980s MPB and samba-pop recording, rich percussion, cavaquinho, acoustic guitar, warm bass, tropical rhythm", "Spain": "a Spanish 1980s pop and rock recording, flamenco-influenced guitar, Latin pop rhythm, Mediterranean sound",
      "Germany": "a German 1980s Neue Deutsche Welle and electronic recording, cold synthesizers, sequencers, drum machine, Berlin sound", "Italy": "an Italian 1980s pop and italo disco recording, melodic synthesizers, disco bass, string pads, Italian pop production"}
MS = json.load(open(f"{ROOT}/results/market_styles.json")); P = {"neutral": "a song recording"}
for m in HW:
    P["hw_" + m] = HW[m]; names = [x["style"].split("---")[1] for x in MS[f"{m}|1980s"]["top_lift"][:3]]; P["ds_" + m] = f"a {ADJ[m]} recording in the styles {names[0]}, {names[1]} and {names[2]}"
    if "top_kl" in MS[f"{m}|1980s"]: kn = [x["style"].split("---")[1] for x in MS[f"{m}|1980s"]["top_kl"][:3]]; P["dk_" + m] = f"a {ADJ[m]} recording in the styles {kn[0]}, {kn[1]} and {kn[2]}"
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--rot", type=int, default=0, help="rotate the clip order by this many clips (parallel workers on other GPUs)"); ap.add_argument("--guidance", type=float, default=3.0); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen_mstyle")); os.makedirs(G, exist_ok=True)
rows = [r for r in manifest("B") if r["split"] == "validation"]; pick = []
for lab in LABELS["B"]: pick += [r for r in rows if r["label"] == lab][:4]
open(os.path.join(G, "picked.txt"), "w").write("\n".join(r["sample_id"] + "," + r["label"] for r in pick)); json.dump(P, open(os.path.join(G, "prompts.json"), "w"), indent=1)
proc = AutoProcessor.from_pretrained("facebook/musicgen-melody"); model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody", torch_dtype=torch.float16).to("cuda").eval(); n = 0
order = pick[::-1] if a.reverse else pick; order = order[a.rot:] + order[:a.rot]
for r in order:
    mix = sf.read(r["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "B", r["sample_id"] + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for cond, prompt in P.items():
        for seed in a.seeds:
            fp = os.path.join(G, f"{r['sample_id']}_{cond}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            inp = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}; torch.manual_seed(seed)
            with torch.no_grad(): out = model.generate(**inp, do_sample=True, guidance_scale=a.guidance, max_new_tokens=1500)
            g = AF.resample(out[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(r["sample_id"], r["label"], cond, seed, n, flush=True)
print("done", n)
