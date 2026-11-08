"""Internal demo table of the 2D control (era x language): mix = vocal stem (ACE-Step cover-mode lyric route + HT Demucs, `ace_a_swap_voc`, languages orig/es/de) + era accompaniment (one-hot+16 adapter with HPCP chroma, `gen_ax_onehot_pfx16_ge10_hpcp`, decades neutral/1960s/1980s/2010s), seed 0, first N songs that have all files. Vocal material derives from licensed commercial recordings: files go to `demo_internal/` (git-ignored, never published); the public page may use accompaniment-only clips. Writes OGG files and `demo_internal/manifest.json`.
usage (from ~/era_market, any venv with soundfile): python src/161_compose_demo.py --n 4"""
import argparse, glob, json, os, numpy as np, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
H = os.path.expanduser("~/era_market"); DEC = ["neutral", "1960s", "1980s", "2010s"]; LG = ["orig", "es", "de"]; O = os.path.join(H, "demo_internal"); os.makedirs(O, exist_ok=True); man = []; done = 0
ids = sorted({os.path.basename(f).split("_orig_")[0] for f in glob.glob(f"{H}/ace_a_swap_voc/*_orig_s{a.seed}.wav")})
for s in ids:
    if done >= a.n: break
    vf = {lg: f"{H}/ace_a_swap_voc/{s}_{lg}_s{a.seed}.wav" for lg in LG}; af = {d: f"{H}/gen_ax_onehot_pfx16_ge10_hpcp/{s}_{d}_s{a.seed}_acc.wav" for d in DEC}
    if not all(os.path.exists(p) for p in list(vf.values()) + list(af.values())): continue
    V = {lg: sf.read(p, dtype="float32") for lg, p in vf.items()}; A = {d: sf.read(p, dtype="float32") for d, p in af.items()}
    for d in DEC:
        for lg in LG:
            v, sr = V[lg]; acc, sra = A[d]; v = v.mean(1) if v.ndim > 1 else v; acc = acc.mean(1) if acc.ndim > 1 else acc; assert sr == sra, (sr, sra); n = min(len(v), len(acc)); mix = np.clip(v[:n] + acc[:n], -1, 1)
            fn = f"{s}_{d}_{lg}.ogg"; sf.write(os.path.join(O, fn), mix, sr, format="OGG", subtype="VORBIS"); man.append({"file": fn, "song": s, "era": d, "language": lg})
    done += 1
json.dump({"seed": a.seed, "files": man}, open(os.path.join(O, "manifest.json"), "w"), indent=1); print(done, "songs,", len(man), "files ->", O)
