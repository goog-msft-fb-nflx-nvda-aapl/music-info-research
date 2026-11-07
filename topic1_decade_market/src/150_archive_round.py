"""Archive one generation directory before its wavs are deleted: (1) CLAP audio embeddings (laion/clap-htsat-unfused) of every accompaniment wav (`<clip>_<cond>_s<seed>_acc.wav`) -> emb_clap/<gendir>.npz (names, emb); MERT/MuQ embeddings already exist as results/<tag>_eval_emb.npz written by 10_eval_gen.py, Discogs-EffNet via 36_discogs_tagger.py (venv_ess); (2) a demo set: the first N clips (sorted ids) x all conditions, seed 0, accompaniment only, written as OGG (quality 0.4) to demo_keep/<config>/ with manifest.json (git-ignored, publish accompaniment-only files only).
usage (from ~/era_market, venv_mcl (torch 2.6, needed for the CLAP checkpoint), one GPU): python src/150_archive_round.py --gendir gen_ax_onehot_pfx16_ge10_hpcp --config adapter_onehot16_hpcp [--clips 3] [--seed 0]"""
import argparse, glob, json, os, numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from transformers import ClapModel, ClapProcessor
ap = argparse.ArgumentParser(); ap.add_argument("--gendir", required=True); ap.add_argument("--config", required=True); ap.add_argument("--clips", type=int, default=3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_clap", action="store_true"); a = ap.parse_args()
H = os.path.expanduser("~/era_market"); G = os.path.join(H, a.gendir); files = sorted(glob.glob(os.path.join(G, "*_acc.wav"))); print(len(files), "accompaniment files", flush=True)
if not a.no_clap:
    mid = "laion/clap-htsat-unfused"; proc = ClapProcessor.from_pretrained(mid); model = ClapModel.from_pretrained(mid).cuda().eval(); names, embs = [], []
    for i in range(0, len(files), 16):
        wavs = []
        for f in files[i:i + 16]:
            x, sr = sf.read(f, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x; wavs.append(AF.resample(torch.from_numpy(x), sr, 48000).numpy()); names.append(os.path.basename(f)[:-8])
        inp = proc(audios=wavs, sampling_rate=48000, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(): embs.append(model.get_audio_features(**inp).float().cpu().numpy())
    os.makedirs(os.path.join(H, "emb_clap"), exist_ok=True); np.savez(os.path.join(H, "emb_clap", a.gendir + ".npz"), names=np.array(names), emb=np.concatenate(embs)); print("clap", np.concatenate(embs).shape, flush=True)
clips = sorted({os.path.basename(f).split("_")[0] + "_" + os.path.basename(f).split("_")[1] for f in files})[:a.clips]; D = os.path.join(H, "demo_keep", a.config); os.makedirs(D, exist_ok=True); man = []
for f in files:
    b = os.path.basename(f)[:-8]; p = b.split("_"); cid = p[0] + "_" + p[1]
    if cid in clips and b.endswith(f"_s{a.seed}"):
        x, sr = sf.read(f, dtype="float32"); out = os.path.join(D, b + ".ogg"); sf.write(out, x, sr, format="OGG", subtype="VORBIS"); man.append({"file": b + ".ogg", "clip": cid, "condition": p[2], "seed": a.seed, "source_dir": a.gendir})
json.dump({"config": a.config, "gendir": a.gendir, "files": man}, open(os.path.join(D, "manifest.json"), "w"), indent=1); print("demo", len(man), "files ->", D)
