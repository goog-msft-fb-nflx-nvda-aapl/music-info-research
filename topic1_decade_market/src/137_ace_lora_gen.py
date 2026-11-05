"""N-7 pilot generation: ACE-Step 1.5 turbo + the era LoRA of 136_ace_lora.sh in cover mode on the ORIGINAL ACCOMPANIMENT STEM of the 24 evaluation songs (dataset A validation, gen/picked.txt), protocol of 55_ace_cover.py (instrumental, original vocal remixed, output RMS-matched to the original accompaniment): conditions neutral and the six decade prompts of prompts_era.py (the training captions), cover strength --strength, seeds --seeds; --lora <dir> loads the adapter (omit for the no-LoRA control with the same prompts), --scale sets the adapter scale. Output <outdir>/<id>_<era>_s<seed>_{acc,remix}.wav + picked.txt. Read-out: ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=<outdir> python src/10_eval_gen.py --out results/<outdir>_eval.json --train_cond acccodec_norm --seeds ... (codec-matched MERT/MuQ era probes), then 78_six_decades.py.
Run with ~/era_market/venv_ace from ~/era_market/ext/ACE-Step-1.5 (CUDA_VISIBLE_DEVICES set).
usage: python ~/era_market/src/137_ace_lora_gen.py --lora ace_lora_out_e3/<checkpoint dir> --strength 0.5 --outdir ace_lora_gen_e3 --seeds 0 1"""
import argparse, os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.expanduser("~/era_market/src"))
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
import prompts_era
ERAS = dict(prompts_era.ERAS); ERAS.pop("1960s_b", None); ERAS.pop("1960s_c", None)
H = os.path.expanduser("~/era_market"); SR = 24000
ap = argparse.ArgumentParser(); ap.add_argument("--lora", default=None); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
ids = [l.split(",")[0] for l in open(f"{H}/gen/picked.txt").read().split("\n")]; out = os.path.join(H, a.outdir); os.makedirs(out, exist_ok=True); open(os.path.join(out, "picked.txt"), "w").write(open(f"{H}/gen/picked.txt").read()); tmp = os.path.join(H, "ace_work"); os.makedirs(tmp, exist_ok=True)
dit = AceStepHandler(); llm = LLMHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda"), flush=True)
if a.lora:
    print(dit.load_lora(os.path.abspath(a.lora)), flush=True)
    try: print(dit.set_lora_scale(a.scale) if hasattr(dit, "set_lora_scale") else "", flush=True)
    except Exception as e: print("scale not set:", e, flush=True)
n = 0
for s in ids:
    z = np.load(f"{H}/stems/A/{s}.npz"); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); src = os.path.join(tmp, s + "_acc_lora_src.wav"); sf.write(src, acc0, SR)
    for era, prompt in ERAS.items():
        if os.environ.get("YEAR_CAPTION") and era != "neutral": prompt = prompt + ", recorded in " + era[:3] + "4"   # year-augmented caption variant (144_ace_lora_year.sh)
        for sd in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{sd}")
            if os.path.exists(fp + "_remix.wav"): continue
            p = GenerationParams(task_type="cover", src_audio=src, caption=prompt, lyrics="[Instrumental]", instrumental=True, audio_cover_strength=a.strength, duration=30.0, inference_steps=8, shift=3.0, seed=sd, thinking=False)
            r = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=tmp)
            if not r.success: print("FAIL", s, era, sd, r.error, flush=True); continue
            w = sf.read(r.audios[0]["path"], dtype="float32")[0]; w = w.mean(1) if w.ndim == 2 else w
            g = AF.resample(torch.from_numpy(np.ascontiguousarray(w)), 48000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, sd, n, flush=True); os.remove(r.audios[0]["path"])
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
