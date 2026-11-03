"""T-4b stage 3: read-outs for the lyric route (111_ace_lyric_route.py outputs ace_lyric/<id>_<cond>_s<seed>.wav, 12 pilot clips x {orig, es, pt, de, it}). Per generated clip: (a) language of the singing: Whisper large-v3 language-token softmax at the first decoder step on the generated mix (as 05_langid.py), reported as P(intended language) and top-1 hit; the same on the original mix as reference; (b) lyric intelligibility: Whisper large-v3 transcript with the language forced to the intended language, similarity to the intended lyrics = difflib ratio of lower-cased letters (1 = identical) and the same similarity to the other four lyric versions (a clip that sings the right text is closest to its own); (c) song identity: Discogs-VINet cosine between the generated mix and the original mix (and retrieval top-1 among the 12 originals).
usage (from ~/era_market, main venv): [GEN=ace_lyric_s08] python src/112_lyric_route_eval.py -> results/lyric_route_eval.{json,txt}"""
import os, sys, json, csv, difflib, re, numpy as np, torch, librosa, soundfile as sf
SEED = int(os.environ.get("SEED", "0"))   # generation seed of the files to read (default 0)
sys.path.insert(0, "src"); sys.path.insert(0, os.path.expanduser("~/era_market/ext/Discogs-VINet"))
from common import ROOT, SR
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from model.nets.cqtnet import CQTNet
GD = os.environ.get("GEN", "ace_lyric"); TAG = "" if GD == "ace_lyric" else "_" + GD.replace("ace_lyric_", ""); TEXT = os.environ.get("TEXT", "lyrics_route_text.json"); T = json.load(open(f"{ROOT}/results/{TEXT}")); man = {r["sample_id"]: r for r in csv.DictReader(open(f"{ROOT}/data/ex/dataset_B/manifest.csv"))}; ids = list(T); CONDS = ["orig", "en", "es", "pt", "de", "it"]; CODE = {"orig": "en", "en": "en", "es": "es", "pt": "pt", "de": "de", "it": "it"}
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).cuda().eval(); tok = proc.tokenizer
lang_ids = tok.convert_tokens_to_ids([f"<|{c}|>" for c in LANGUAGES]); lang_list = list(LANGUAGES); sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
net = CQTNet(); sd = torch.load(os.path.expanduser("~/era_market/ext/Discogs-VINet/logs/checkpoints/Discogs-VINet/model_checkpoint.pth"), map_location="cpu", weights_only=True)["model_state_dict"]; net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}); net = net.cuda().eval()
def vemb(y24):
    y = librosa.resample(np.asarray(y24, dtype=np.float32), orig_sr=SR, target_sr=22050); c = np.abs(librosa.cqt(y=y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)).T.astype(np.float16).astype(np.float32)
    if c.shape[0] < 8000: c = np.pad(c, ((0, 8000 - c.shape[0]), (0, 0)))
    Tn = c.shape[0] // 20; c = c[:Tn * 20].reshape(Tn, 20, 84).mean(1); c = np.where(c < 0, 0, c); c /= c.max() + 1e-6
    with torch.inference_mode(): return net(torch.as_tensor(c.T)[None, None].cuda()).float().cpu().numpy()[0]
def lid(y24):
    f = proc(librosa.resample(y24, orig_sr=SR, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.inference_mode(): lg = wm(input_features=f, decoder_input_ids=torch.tensor([[sot]]).cuda()).logits[0, -1]
    return torch.softmax(lg[lang_ids].float(), -1).cpu().numpy()
def asr(y24, code):
    f = proc(librosa.resample(y24, orig_sr=SR, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    with torch.inference_mode(): g = wm.generate(f, language=code, task="transcribe", max_new_tokens=220)
    return proc.batch_decode(g, skip_special_tokens=True)[0].strip()
norm_t = lambda s: re.sub(r"[^a-zà-ÿ]", "", s.lower()); sim = lambda a, b: difflib.SequenceMatcher(None, norm_t(a), norm_t(b)).ratio()
orig = {s: sf.read(f"{ROOT}/data/ex/dataset_B/{man[s]['audio_path']}", dtype="float32")[0] for s in ids}; OE = np.stack([vemb(orig[s]) for s in ids]); res = {"rows": []}
for i, s in enumerate(ids):
    SL = T[s].get("src_lang", "en"); res["rows"].append({"id": s, "cond": "original_mix", "P_en": float(lid(orig[s])[lang_list.index("en")])})
    for c in CONDS:
        fp = f"{ROOT}/{GD}/{s}_{c}_s{SEED}.wav"
        if not os.path.exists(fp): continue
        cc = SL if c == "orig" else CODE[c]; y = sf.read(fp, dtype="float32")[0]; p = lid(y); intended = T[s].get("text_src", T[s].get("text_en")) if c == "orig" else T[s][c]; hyp = asr(y, cc); e = vemb(y); sims = OE @ e
        res["rows"].append({"id": s, "cond": c, "P_intended_lang": float(p[lang_list.index(cc)]), "lang_top1": lang_list[int(p.argmax())], "hit": int(lang_list[int(p.argmax())] == cc), "sim_to_intended_lyrics": sim(hyp, intended), "sim_to_other_versions": {k: sim(hyp, (T[s].get("text_src", T[s].get("text_en")) if k == "orig" else T[s][k])) for k in CONDS if k == "orig" or k in T[s]}, "vinet_cos_own": float(sims[i]), "vinet_top1": int(sims.argmax() == i), "transcript": hyp[:200]})
    print(s, flush=True)
txt = []
for c in CONDS:
    R = [r for r in res["rows"] if r["cond"] == c]
    if R: txt.append(f"{c}: n={len(R)} P(intended language) {np.mean([r['P_intended_lang'] for r in R]):.3f}, language top-1 hit {np.mean([r['hit'] for r in R]):.3f}, lyric similarity to intended text {np.mean([r['sim_to_intended_lyrics'] for r in R]):.3f}, intended text is the closest of the five versions in {np.mean([max(r['sim_to_other_versions'], key=r['sim_to_other_versions'].get) == c for r in R]):.2f}, VINet cos to the original {np.mean([r['vinet_cos_own'] for r in R]):.3f}, VINet top-1 among {len(ids)} {np.mean([r['vinet_top1'] for r in R]):.2f}")
txt.append(f"original mix: P(en) {np.mean([r['P_en'] for r in res['rows'] if r['cond'] == 'original_mix']):.3f}")
json.dump(res, open(f"{ROOT}/results/lyric_route_eval{TAG}.json", "w"), indent=1, ensure_ascii=False); open(f"{ROOT}/results/lyric_route_eval{TAG}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
