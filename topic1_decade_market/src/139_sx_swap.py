"""T-4h-b step 4: SoulX-Singer vocal (melody-driven resynthesis, 138_sx_synth.sh) RMS-matched to the original vocal stem and added to the ORIGINAL accompaniment stem -> sx_swap[_score]/<id>_orig_s0.wav; read out with GEN=sx_swap python src/112_lyric_route_eval.py. usage (from ~/era_market): [CONTROL=score] python src/139_sx_swap.py"""
import os, json, numpy as np, soundfile as sf
H = os.path.expanduser("~/era_market"); C = os.environ.get("CONTROL", "melody"); DST = "sx_swap" if C == "melody" else "sx_swap_score"; os.makedirs(f"{H}/{DST}", exist_ok=True); n = 0
for sid in json.load(open(f"{H}/results/lyrics_route_text.json")):
    fp = f"{H}/sx/gen_{C}/{sid}/generated.wav"
    if not os.path.exists(fp): continue
    z = np.load(f"{H}/stems/B/{sid}.npz"); voc0, acc = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    v = sf.read(fp, dtype="float32")[0]; v = v.mean(1) if v.ndim > 1 else v; v = np.pad(v, (0, max(0, len(acc) - len(v))))[:len(acc)]
    v = v * (np.sqrt((voc0 ** 2).mean()) / (np.sqrt((v ** 2).mean()) + 1e-9)); sf.write(f"{H}/{DST}/{sid}_orig_s0.wav", np.clip(acc + v, -1, 1), 24000); n += 1
print("swapped", n)
