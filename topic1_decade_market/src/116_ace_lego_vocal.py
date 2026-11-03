"""T-4g: vocal-only regeneration. ACE-Step 1.5 BASE model (acestep-v15-base, MIT; ACE-Step/acestep-v15-base) in 'lego' mode: given the ORIGINAL ACCOMPANIMENT STEM of a pilot clip as audio context, generate the 'vocals' track with new lyrics (translated transcript, 110_lyrics_translate.py) in the target language. Conditions per clip: orig (English transcript; pipeline control), es, pt, de, it. instruction 'Generate the vocals track based on the audio context:', caption 'a pop song sung in <language>', vocal_language = code, duration 30 s, base model --steps 50, guidance --cfg 7. Output (48 kHz) resampled to 24 kHz mono: <outdir>/<id>_<cond>_s<seed>.wav (the raw model output; whether it contains the context is checked by the read-out 112_lyric_route_eval.py via VINet similarity to the original mix, and by --remix which adds the generated track to the original accompaniment RMS-matched to the original vocal).
Run with ~/era_market/venv_ace from ~/era_market/ext/ACE-Step-1.5. usage: python ~/era_market/src/116_ace_lego_vocal.py --outdir ace_lego --seeds 0"""
import argparse, os, sys, json, csv, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd())
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
H = os.path.expanduser("~/era_market"); SR = 24000
LANG = {"orig": ("en", "English"), "es": ("es", "Spanish"), "pt": ("pt", "Portuguese"), "de": ("de", "German"), "it": ("it", "Italian")}
ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="ace_lego"); ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=7.0); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
T = json.load(open(f"{H}/results/lyrics_route_text.json")); out = os.path.join(H, a.outdir); os.makedirs(out, exist_ok=True); tmp = os.path.join(H, "ace_work"); os.makedirs(tmp, exist_ok=True)
dit = AceStepHandler(); llm = LLMHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-base", device="cuda"), flush=True); n = 0
for sid, d in T.items():
    z = np.load(f"{H}/stems/B/{sid}.npz"); acc = z["accomp"].astype(np.float32); src = os.path.join(tmp, sid + "_accomp.wav"); sf.write(src, acc, SR)
    for cond, (code, name) in LANG.items():
        text = d["text_en"] if cond == "orig" else d[cond]
        for sd in a.seeds:
            fp = os.path.join(out, f"{sid}_{cond}_s{sd}.wav")
            if os.path.exists(fp): continue
            p = GenerationParams(task_type="lego", src_audio=src, instruction="Generate the vocals track based on the audio context:", caption=f"a pop song sung in {name}", lyrics=text[:3500], instrumental=False, vocal_language=code, duration=30.0, inference_steps=a.steps, guidance_scale=a.cfg, seed=sd, thinking=False)
            try: r = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=tmp)
            except Exception as e: print("EXC", sid, cond, str(e)[:160], flush=True); continue
            if not r.success: print("FAIL", sid, cond, sd, r.error, flush=True); continue
            w = sf.read(r.audios[0]["path"], dtype="float32")[0]; w = w.mean(1) if w.ndim == 2 else w
            g = AF.resample(torch.from_numpy(np.ascontiguousarray(w)), 48000, SR).numpy()[:len(acc)]; sf.write(fp, g, SR); n += 1; print(sid, cond, sd, n, flush=True); os.remove(r.audios[0]["path"])
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
