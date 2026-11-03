"""T-4b stage 1: transcribe the vocal stems of English-sung dataset-B validation clips (Whisper large-v3, local weights) and translate the transcripts to the languages of the other markets (M2M-100 1.2B, facebook/m2m100_1.2B, MIT license; en -> es, pt, de, it). Pilot set: validation clips of market US or UK whose Whisper language group is English, first --n by sample id. Output: results/lyrics_route_text.json {sample_id: {market, text_en, es, pt, de, it}}.
usage (from ~/era_market, main venv, one GPU): python src/110_lyrics_translate.py --n 12"""
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, "src")
from common import ROOT, LABELS, manifest
from transformers import WhisperProcessor, WhisperForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
rows = manifest("B"); lid = np.load(f"{ROOT}/emb/B_lid3_voc.npy"); langs = json.load(open(f"{ROOT}/emb/lid_langs.json"))
pick = [r["sample_id"] for i, r in enumerate(rows) if r["split"] == "validation" and r["label"] in ("US", "UK") and langs[int(np.argmax(lid[i]))] == "en"]; pick = sorted(pick)[:a.n]; mk = {r["sample_id"]: r["label"] for r in rows}
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval()
import torchaudio.functional as AF
out = {}
for sid in pick:
    v = np.load(f"{ROOT}/stems/B/{sid}.npz")["vocals"].astype(np.float32); v = AF.resample(torch.from_numpy(v), 24000, 16000).numpy()
    f = proc(v, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.no_grad(): ids = wm.generate(f, language="en", task="transcribe", max_new_tokens=220)
    out[sid] = {"market": mk[sid], "text_en": proc.batch_decode(ids, skip_special_tokens=True)[0].strip()}; print(sid, out[sid]["text_en"][:90], flush=True)
del wm; torch.cuda.empty_cache()
from huggingface_hub import hf_hub_download
from transformers import M2M100Config
tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B"); mt = M2M100ForConditionalGeneration(M2M100Config.from_pretrained("facebook/m2m100_1.2B"))
# the checkpoint is a pickle .bin and the installed torch (2.5.1) is below the version transformers requires for its own loader: load the state dict directly with weights_only=True
sd = torch.load(hf_hub_download("facebook/m2m100_1.2B", "pytorch_model.bin"), map_location="cpu", weights_only=True); print(mt.load_state_dict(sd, strict=False)); mt = mt.half().cuda().eval(); tok.src_lang = "en"
for sid, d in out.items():
    for L in ("es", "pt", "de", "it"):
        enc = tok(d["text_en"], return_tensors="pt").to("cuda")
        with torch.no_grad(): g = mt.generate(**enc, forced_bos_token_id=tok.get_lang_id(L), max_new_tokens=256, num_beams=4)
        d[L] = tok.batch_decode(g, skip_special_tokens=True)[0]
    print(sid, d["es"][:80], flush=True)
json.dump(out, open(f"{ROOT}/results/lyrics_route_text.json", "w"), indent=1, ensure_ascii=False)
