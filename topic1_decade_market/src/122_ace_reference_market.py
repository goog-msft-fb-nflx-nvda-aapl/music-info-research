"""T-4m: reference-audio style transfer for the market axis. ACE-Step 1.5 turbo (cover mode, instrumental) regenerates the ACCOMPANIMENT of the 24 dataset-B evaluation clips (4 per source market, first by manifest order, as 114_market_style_gen.py) from the original accompaniment stem (src_audio, strength --strength) with reference_audio = the accompaniment stem of a TRAIN clip of the target market (market m in US, UK, Brazil, Spain, Germany, Italy; reference k = (clip index + offset) mod 3 of the first three train clips of the market by sample id, so that no reference is tied to one evaluation clip); condition 'neutral' = same call without reference audio (control of the pipeline). caption 'a song recording', lyrics [Instrumental]. Output GENDIR/<id>_<cond>_s0_{acc,remix}.wav (acc = generated, RMS-matched to the original accompaniment; remix = original vocal + acc), named by market so that 37_eval_market.py can read it.
Run with ~/era_market/venv_ace from ~/era_market/ext/ACE-Step-1.5. usage: GENDIR=ace_mref python ~/era_market/src/122_ace_reference_market.py --strength 0.5 ; read-out: GENDIR=ace_mref python src/37_eval_market.py --out results/ace_mref_eval.json --seeds 0"""
import argparse, os, sys, json, csv, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd())
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
H = os.path.expanduser("~/era_market"); SR = 24000; MK = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
ap = argparse.ArgumentParser(); ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
rows = list(csv.DictReader(open(f"{H}/data/ex/dataset_B/manifest.csv"))); val = [r for r in rows if r["split"] == "validation" and r["label"]]; pick = []
for lab in MK: pick += [r for r in val if r["label"] == lab][:4]
refs = {m: sorted(r["sample_id"] for r in rows if r["split"] == "train" and r["label"] == m)[:3] for m in MK}
G = os.path.join(H, os.environ.get("GENDIR", "ace_mref")); tmp = os.path.join(H, "ace_work"); os.makedirs(G, exist_ok=True); os.makedirs(tmp, exist_ok=True); open(os.path.join(G, "picked.txt"), "w").write("\n".join(r["sample_id"] + "," + r["label"] for r in pick)); json.dump(refs, open(os.path.join(G, "references.json"), "w"))
def ref_wav(sid):
    p = f"{tmp}/{sid}_refacc.wav"
    if not os.path.exists(p): sf.write(p, np.load(f"{H}/stems/B/{sid}.npz")["accomp"].astype(np.float32), SR)
    return p
dit = AceStepHandler(); llm = LLMHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda"), flush=True); n = 0; SI, SN = map(int, a.shard.split("/"))
for ci, r in list(enumerate(pick))[SI::SN]:
    sid = r["sample_id"]; z = np.load(f"{H}/stems/B/{sid}.npz"); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); src = f"{tmp}/{sid}_acc_src.wav"; sf.write(src, acc0, SR)
    for cond in ["neutral"] + MK:
        fp = f"{G}/{sid}_{cond}_s0"
        if os.path.exists(fp + "_remix.wav"): continue
        ref = None if cond == "neutral" else ref_wav(refs[cond][(ci + MK.index(cond)) % 3])
        p = GenerationParams(task_type="cover", src_audio=src, reference_audio=ref, caption="a song recording", lyrics="[Instrumental]", instrumental=True, audio_cover_strength=a.strength, duration=30.0, inference_steps=8, shift=3.0, seed=0, thinking=False)
        res = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=tmp)
        if not res.success: print("FAIL", sid, cond, res.error, flush=True); continue
        w = sf.read(res.audios[0]["path"], dtype="float32")[0]; w = w.mean(1) if w.ndim == 2 else w; g = AF.resample(torch.from_numpy(np.ascontiguousarray(w)), 48000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
        sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(sid, cond, n, flush=True); os.remove(res.audios[0]["path"])
        if a.limit and n >= a.limit: raise SystemExit
print("done", n)
