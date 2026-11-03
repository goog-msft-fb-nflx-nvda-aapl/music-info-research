"""N-7 pilot (ACE-Step 1.5 LoRA with era captions): build the training set from the 1,026 labeled dataset-A TRAIN clips: the ORIGINAL ACCOMPANIMENT STEM (HT Demucs, 24 kHz mono wav, instrumental) with caption = the era prompt of the clip's decade (src/prompts_era.py ERAS: 1960s-2010s; the six main prompts), lyrics '[Instrumental]', language unknown. Output ace_lora/data/<id>.wav, <id>.json (caption, instrumental), <id>.lyrics.txt. (PA1 audio stays on the server; it is not released.)
usage (from ~/era_market, main venv): python src/136_ace_lora_data.py"""
import os, sys, json, numpy as np, soundfile as sf
sys.path.insert(0, "src")
from common import ROOT, SR, LABELS, manifest
import prompts_era
out = os.path.join(ROOT, "ace_lora", "data"); os.makedirs(out, exist_ok=True); n = 0
for r in manifest("A"):
    if r["split"] != "train" or not r["label"]: continue
    sid = r["sample_id"]; acc = np.load(os.path.join(ROOT, "stems", "A", sid + ".npz"))["accomp"].astype(np.float32); sf.write(os.path.join(out, sid + ".wav"), acc, SR)
    json.dump({"caption": prompts_era.ERAS[r["label"]], "lyrics": "[Instrumental]", "language": "unknown", "instrumental": True}, open(os.path.join(out, sid + ".json"), "w")); open(os.path.join(out, sid + ".lyrics.txt"), "w").write("[Instrumental]\n"); n += 1
print("clips", n)
