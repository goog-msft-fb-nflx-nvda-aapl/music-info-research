"""B-1: SongEcho_base / SongEcho_large / MuseControlLite adapters of the Text-to-Music control family (Tsai et al., HF fundwotsai2001/Text-to-Music_control_family, Stable Audio Open 1.0 base) as external melody-controlled baselines, same protocol as 27_musecontrol.py: melody extracted from the ORIGINAL MIX of each of the 24 validation songs, era prompts of prompts_era.py, seeds 0-2, first 30 s resampled to 24 kHz mono, RMS-matched to the original accompaniment stem, remix with the original vocal stem.
run from ~/era_market/ext/Text-to-Music_control_family with venv_mcl: python ../../src/100_songecho_sao.py --method SongEcho_base --outdir sao_songecho_base --eras neutral 1960s 1970s 1980s 1990s 2000s 2010s
Status: written, NOT run (see WORKLOG 2026-11-01: loading the pipeline code from the Hugging Face hub was not permitted in the session)."""
import argparse, os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from diffusers import DiffusionPipeline
from common import ROOT, SR, manifest
import prompts_era
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True, choices=["SongEcho_base", "SongEcho_large", "MuseControlLite"]); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--eras", nargs="+", default=["neutral", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gcon", type=float, default=1.0); ap.add_argument("--gtext", type=float, default=7.0); a = ap.parse_args()
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", custom_pipeline="fundwotsai2001/Text-to-Music_control_family", attn_processor_type=a.method, torch_dtype=torch.float16, adapter_repo_id="fundwotsai2001/Text-to-Music_control_family",
    adapter_checkpoint=f"{a.method}/adapters.safetensors", melody_encoder_checkpoint=f"{a.method}/melody_encoder.safetensors").to("cuda")
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); open(os.path.join(out, "picked.txt"), "w").write(open(os.path.join(G, "picked.txt")).read()); work = os.path.join(ROOT, "mcl_work"); os.makedirs(work, exist_ok=True); n = 0
for s in ids:
    mx = AF.resample(torch.from_numpy(sf.read(rows[s]["path"], dtype="float32")[0]), SR, 44100).numpy(); cond_wav = os.path.join(work, s + "_mix44k.wav"); sf.write(cond_wav, np.stack([mx, mx], 1), 44100)
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); cond = pipe.encode_melody_from_path(cond_wav)
    for era in a.eras:
        prompt = prompts_era.ERAS[era] if era != "neutral" else "a song recording"
        for sd in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{sd}")
            if os.path.exists(fp + "_remix.wav"): continue
            with torch.no_grad(): w = pipe(extracted_condition=cond, prompt=prompt, negative_prompt="", num_inference_steps=50, guidance_scale_text=a.gtext, guidance_scale_con=a.gcon, num_waveforms_per_prompt=1, audio_end_in_s=2097152 / 44100, generator=torch.Generator().manual_seed(sd)).audios
            g = w[0].float().mean(0).cpu(); g = AF.resample(g[: 44100 * 30], 44100, SR).numpy()[: len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, sd, n, flush=True)
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
