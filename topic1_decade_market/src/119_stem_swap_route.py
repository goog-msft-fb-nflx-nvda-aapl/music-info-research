"""T-4i: combine the routes. The cover-route output of T-4b (ACE-Step cover mode, `ace_lyric_s095/<id>_<cond>_s0.wav`, strength 0.95: the model carries the target language) is separated with HT Demucs; its VOCAL stem (RMS-matched to the original vocal stem) is added to the ORIGINAL accompaniment stem of the clip (the song carrier) -> `ace_swap/<id>_<cond>_s0.wav` (the separated vocal alone, RMS-matched, also in `ace_swap_voc/`), read out with 112_lyric_route_eval.py (GEN=ace_swap).
usage (from ~/era_market, main venv): [SRC=ace_lyric_s095] python src/119_stem_swap_route.py"""
import os, json, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from demucs.pretrained import get_model
from demucs.apply import apply_model
H = os.path.expanduser("~/era_market"); SRC = os.environ.get("SRC", "ace_lyric_s095"); DST = os.environ.get("DST") or ("ace_swap" if SRC == "ace_lyric_s095" else "ace_swap_" + SRC.replace("ace_lyric_", "")); TEXT = os.environ.get("TEXT", "lyrics_route_text.json"); T = json.load(open(f"{H}/results/{TEXT}")); os.makedirs(f"{H}/{DST}", exist_ok=True); os.makedirs(f"{H}/{DST}_voc", exist_ok=True)
model = get_model("htdemucs").cuda().eval(); n = 0
for sid in T:
    z = np.load(f"{H}/stems/B/{sid}.npz"); voc0, acc = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    for c in ("orig", "en", "es", "pt", "de", "it"):
        fp = f"{H}/{SRC}/{sid}_{c}_s0.wav"
        if not os.path.exists(fp): continue
        y = sf.read(fp, dtype="float32")[0]; x = AF.resample(torch.from_numpy(y)[None], 24000, 44100).repeat(2, 1)[None].cuda()
        with torch.no_grad(): s = apply_model(model, x, shifts=1, overlap=0.25, device="cuda")[0]
        v = s[model.sources.index("vocals")].mean(0, keepdim=True); v = AF.resample(v.cpu(), 44100, 24000)[0].numpy(); v = np.pad(v, (0, max(0, len(acc) - len(v))))[:len(acc)]; v = v * (np.sqrt((voc0 ** 2).mean()) / (np.sqrt((v ** 2).mean()) + 1e-9))
        sf.write(f"{H}/{DST}/{sid}_{c}_s0.wav", np.clip(acc + v, -1, 1), 24000); sf.write(f"{H}/{DST}_voc/{sid}_{c}_s0.wav", np.clip(v, -1, 1), 24000); n += 1
print("swapped", n)
