"""T-4h-b step 1: export the separated vocal stem of the first N clips (24 kHz wav) for SoulX-Singer annotation. usage (from ~/era_market, main venv): N=24 python src/138_sx_prep.py"""
import os, json, numpy as np, soundfile as sf
H = os.path.expanduser("~/era_market"); N = int(os.environ.get("N", "24")); T = json.load(open(f"{H}/results/lyrics_route_text.json")); os.makedirs(f"{H}/sx/in", exist_ok=True)
for sid in list(T)[:N]:
    z = np.load(f"{H}/stems/B/{sid}.npz"); sf.write(f"{H}/sx/in/{sid}.wav", z["vocals"].astype(np.float32), 24000)
print("exported", min(N, len(T)))
