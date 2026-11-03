"""T-4b stage 2: ACE-Step 1.5 cover mode on the ORIGINAL MIX with the transcript translated to another market's language as lyrics (pilot: English-sung US/UK validation clips of dataset B, translations from 110_lyrics_translate.py). Conditions per clip: orig (original English transcript, vocal_language en; control for the pipeline) and es / pt / de / it (translated lyrics, vocal_language = language code, caption 'a pop song sung in <language>'). task_type=cover, src_audio = 30 s mix, audio_cover_strength --strength, turbo model 8 steps, instrumental=False. Output (48 kHz) resampled to 24 kHz mono: <outdir>/<id>_<cond>_s<seed>.wav
Run with ~/era_market/venv_ace from ~/era_market/ext/ACE-Step-1.5. TEXT=lyrics_scale_text.json selects the scale-up text file (conditions = orig + the translations present). usage: python ~/era_market/src/111_ace_lyric_route.py --strength 0.5 --outdir ace_lyric --seeds 0"""
import argparse, os, sys, json, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd())
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
H = os.path.expanduser("~/era_market"); SR = 24000
LANG = {"orig": ("en", "English"), "en": ("en", "English"), "es": ("es", "Spanish"), "pt": ("pt", "Portuguese"), "de": ("de", "German"), "it": ("it", "Italian")}
ap = argparse.ArgumentParser(); ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--outdir", default="ace_lyric"); ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1", help="i/n: process clips i, i+n, ... (parallel workers)"); a = ap.parse_args()
TEXT = os.environ.get("TEXT", "lyrics_route_text.json"); T = json.load(open(f"{H}/results/{TEXT}")); import csv; DS = os.environ.get("DS", "B"); man = {r["sample_id"]: r for r in csv.DictReader(open(f"{H}/data/ex/dataset_{DS}/manifest.csv"))}
out = os.path.join(H, a.outdir); os.makedirs(out, exist_ok=True); tmp = os.path.join(H, "ace_work"); os.makedirs(tmp, exist_ok=True)
dit = AceStepHandler(); llm = LLMHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda"), flush=True); n = 0
SI, SN = map(int, a.shard.split("/"))
for sid, d in list(T.items())[SI::SN]:
    mix = sf.read(os.path.join(H, f"data/ex/dataset_{DS}", man[sid]["audio_path"]), dtype="float32")[0]; src = os.path.join(tmp, sid + "_mix.wav"); sf.write(src, mix, SR)
    SRCL = d.get("src_lang", "en"); NAMES = {"en": "English", "es": "Spanish", "pt": "Portuguese", "de": "German", "it": "Italian", "fr": "French"}
    for cond, (code, name) in LANG.items():
        if cond not in ("orig",) and cond not in d: continue
        if cond == "orig": code, name = (SRCL, NAMES.get(SRCL, SRCL))
        text = d.get("text_src", d.get("text_en")) if cond == "orig" else d[cond]
        for sd in a.seeds:
            fp = os.path.join(out, f"{sid}_{cond}_s{sd}.wav")
            if os.path.exists(fp): continue
            p = GenerationParams(task_type="cover", src_audio=src, caption=f"a pop song sung in {name}", lyrics=text[:3500], instrumental=False, vocal_language=code, audio_cover_strength=a.strength, duration=30.0, inference_steps=8, shift=3.0, seed=sd, thinking=False)
            r = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=tmp)
            if not r.success: print("FAIL", sid, cond, sd, r.error, flush=True); continue
            w = sf.read(r.audios[0]["path"], dtype="float32")[0]; w = w.mean(1) if w.ndim == 2 else w
            g = AF.resample(torch.from_numpy(np.ascontiguousarray(w)), 48000, SR).numpy()[:len(mix)]; sf.write(fp, g, SR); n += 1; print(sid, cond, sd, n, flush=True); os.remove(r.audios[0]["path"])
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
