"""T-4k stage 1 (scale-up of 110_lyrics_translate.py): transcribe the vocal stems of ALL 102 dataset-B validation clips with Whisper large-v3 (language forced to the clip's Whisper language group when it is one of the five target languages, else automatic detection restricted to nothing: language token given by the group) and translate to every language of {en, es, pt, de, it} other than the source language with M2M-100 1.2B (MIT; state dict loaded with weights_only=True, see 110). Clips whose transcript has fewer than 20 letters are dropped. Output results/lyrics_scale_text.json {sample_id: {market, src_lang, text_src, en, es, pt, de, it (all but the source)}} (git-ignored: lyrics of commercial songs).
usage (from ~/era_market, main venv, one GPU): python src/121_lyric_scale_text.py"""
import os, sys, json, re, numpy as np, torch
sys.path.insert(0, "src")
from common import ROOT, LABELS, manifest
from transformers import WhisperProcessor, WhisperForConditionalGeneration, M2M100ForConditionalGeneration, M2M100Tokenizer, M2M100Config
from huggingface_hub import hf_hub_download
import torchaudio.functional as AF
TG = ["en", "es", "pt", "de", "it"]; rows = manifest("B"); lid = np.load(f"{ROOT}/emb/B_lid3_voc.npy"); langs = json.load(open(f"{ROOT}/emb/lid_langs.json")); mk = {r["sample_id"]: r["label"] for r in rows}
pick = [(r["sample_id"], langs[int(np.argmax(lid[i]))]) for i, r in enumerate(rows) if r["split"] == "validation"]
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval(); out = {}
for sid, lg in pick:
    v = np.load(f"{ROOT}/stems/B/{sid}.npz")["vocals"].astype(np.float32); v = AF.resample(torch.from_numpy(v), 24000, 16000).numpy(); f = proc(v, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.no_grad(): ids = wm.generate(f, language=lg, task="transcribe", max_new_tokens=220)
    txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if len(re.findall(r"[^\W\d_]", txt)) < 20: print("drop", sid, lg, repr(txt[:40]), flush=True); continue
    out[sid] = {"market": mk[sid], "src_lang": lg, "text_src": txt}; print(sid, lg, txt[:60], flush=True)
del wm; torch.cuda.empty_cache()
tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B"); mt = M2M100ForConditionalGeneration(M2M100Config.from_pretrained("facebook/m2m100_1.2B"))
sd = torch.load(hf_hub_download("facebook/m2m100_1.2B", "pytorch_model.bin"), map_location="cpu", weights_only=True); print(mt.load_state_dict(sd, strict=False)); mt = mt.half().cuda().eval()
for sid, d in out.items():
    try: tok.src_lang = d["src_lang"]; enc = tok(d["text_src"], return_tensors="pt").to("cuda")
    except Exception as e: print("src lang not supported", sid, d["src_lang"]); d["skip"] = True; continue
    for L in TG:
        if L == d["src_lang"]: continue
        with torch.no_grad(): g = mt.generate(**enc, forced_bos_token_id=tok.get_lang_id(L), max_new_tokens=256, num_beams=4)
        d[L] = tok.batch_decode(g, skip_special_tokens=True)[0]
    print("translated", sid, flush=True)
out = {k: v for k, v in out.items() if not v.get("skip")}; json.dump(out, open(f"{ROOT}/results/lyrics_scale_text.json", "w"), indent=1, ensure_ascii=False); print(len(out), "clips")
