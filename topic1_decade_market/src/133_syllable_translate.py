"""T-4f singable-translation proxy: choose, among 8 beam candidates of the M2M-100 translation (M2M-100 1.2B, MIT; weights loaded with weights_only=True as 110), the candidate whose syllable count is closest to the syllable count of the source transcript (syllables = vowel groups of the lower-cased text; a crude proxy, not a prosodic model); ties -> the better-ranked beam. Input results/lyrics_scale_text.json (121_lyric_scale_text.py, not committed), output results/lyrics_scale_text_syl.json (same structure, git-ignored) and results/syllable_translate_stats.json: mean absolute syllable-count difference to the source for the top beam and for the selected candidate, per target language.
usage (from ~/era_market, main venv, one GPU): python src/133_syllable_translate.py"""
import os, sys, json, re, numpy as np, torch
sys.path.insert(0, "src")
from common import ROOT
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, M2M100Config
from huggingface_hub import hf_hub_download
V = re.compile(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüæœ]+")
def syl(t): return len(V.findall(t.lower()))
T = json.load(open(f"{ROOT}/results/lyrics_scale_text.json")); tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B"); mt = M2M100ForConditionalGeneration(M2M100Config.from_pretrained("facebook/m2m100_1.2B"))
sd = torch.load(hf_hub_download("facebook/m2m100_1.2B", "pytorch_model.bin"), map_location="cpu", weights_only=True); mt.load_state_dict(sd, strict=False); mt = mt.half().cuda().eval(); stats = {}; out = {}
for sid, d in T.items():
    tok.src_lang = d["src_lang"]; enc = tok(d["text_src"], return_tensors="pt").to("cuda"); ns = syl(d["text_src"]); o = dict(d)
    for L in ("en", "es", "pt", "de", "it"):
        if L not in d or L == d["src_lang"]: continue
        with torch.no_grad(): g = mt.generate(**enc, forced_bos_token_id=tok.get_lang_id(L), max_new_tokens=256, num_beams=8, num_return_sequences=8)
        c = tok.batch_decode(g, skip_special_tokens=True); diffs = [abs(syl(x) - ns) for x in c]; j = int(np.argmin(diffs)); o[L] = c[j]; stats.setdefault(L, []).append((abs(syl(c[0]) - ns), diffs[j], (syl(c[0]) - ns) / max(1, ns), (syl(c[j]) - ns) / max(1, ns)))
    out[sid] = o; print(sid, flush=True)
json.dump(out, open(f"{ROOT}/results/lyrics_scale_text_syl.json", "w"), indent=1, ensure_ascii=False)
json.dump({L: {"n": len(v), "mean_abs_diff_top_beam": float(np.mean([x[0] for x in v])), "mean_abs_diff_selected": float(np.mean([x[1] for x in v])), "mean_rel_diff_top_beam": float(np.mean([x[2] for x in v])), "mean_rel_diff_selected": float(np.mean([x[3] for x in v]))} for L, v in stats.items()}, open(f"{ROOT}/results/syllable_translate_stats.json", "w"), indent=1)
