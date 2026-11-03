"""T-4g: remix of the vocal-only regeneration: generated vocals track (ace_lego/<id>_<cond>_s0.wav, 24 kHz) RMS-matched to the ORIGINAL vocal stem of the clip + the original accompaniment stem -> ace_lego_mix/<id>_<cond>_s0.wav, read out with 112_lyric_route_eval.py (GEN=ace_lego_mix).
usage (from ~/era_market, main venv): python src/117_lego_remix.py"""
import os, json, numpy as np, soundfile as sf
H = os.path.expanduser("~/era_market"); T = json.load(open(f"{H}/results/lyrics_route_text.json")); os.makedirs(f"{H}/ace_lego_mix", exist_ok=True); n = 0
for sid in T:
    z = np.load(f"{H}/stems/B/{sid}.npz"); voc0, acc = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    for c in ("orig", "es", "pt", "de", "it"):
        fp = f"{H}/ace_lego/{sid}_{c}_s0.wav"
        if not os.path.exists(fp): continue
        v = sf.read(fp, dtype="float32")[0]; v = np.pad(v, (0, max(0, len(acc) - len(v))))[:len(acc)]; v = v * (np.sqrt((voc0 ** 2).mean()) / (np.sqrt((v ** 2).mean()) + 1e-9))
        sf.write(f"{H}/ace_lego_mix/{sid}_{c}_s0.wav", np.clip(acc + v, -1, 1), 24000); n += 1
print("remixed", n)
