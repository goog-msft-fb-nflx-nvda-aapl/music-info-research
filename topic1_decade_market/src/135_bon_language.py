"""N-5a idea applied to the market axis: best-of-N over generation seeds for the lyric route, with the selector = Whisper large-v3 language reading of the swapped mix (P(intended language), 112_lyric_route_eval.py output per seed) and the evaluation = the vocal market probe on the separated generated vocal (plain probe and language-erased probe; per-clip probabilities of 120_swap_market_probe.py `_clips.json`). Candidates: seeds 0-3 of ace_scale_swap{,_seed1,_seed2,_seed3} (same 77 clips and translations as T-4k). For every (clip, target language): value of the selected candidate, the mean over the four seeds, and the oracle (best candidate by the probe itself, an upper bound); Delta vs the source-language control of the same seed set (mean over seeds of the 'orig' condition). Also the language top-1 hit of the selected and of the average candidate.
usage (CPU, from ~/era_market): python src/135_bon_language.py -> results/bon_language.{json,txt}"""
import json, os, sys, numpy as np
sys.path.insert(0, "src")
from common import ROOT, LABELS
MK = LABELS["B"]; TG = {"es": "Spain", "pt": "Brazil", "de": "Germany", "it": "Italy", "en": "US+UK"}; rng = np.random.default_rng(0)
SEEDS = [("", "ace_scale_swap", "scale_market_probe"), ("_seed1", "ace_scale_swap_seed1", "scale_market_probe_seed1"), ("_seed2", "ace_scale_swap_seed2", "scale_market_probe_seed2"), ("_seed3", "ace_scale_swap_seed3", "scale_market_probe_seed3")]
def tp(p, m): return p[MK.index("US")] + p[MK.index("UK")] if m == "US+UK" else p[MK.index(m)]
LID = []; PR = []; PE = []
for sfx, gd, on in SEEDS:
    ev = json.load(open(f"{ROOT}/results/lyric_route_eval_{gd}.json"))["rows"]; LID.append({(r["id"], r["cond"]): r["P_intended_lang"] for r in ev if "P_intended_lang" in r}); PR.append(json.load(open(f"{ROOT}/results/{on}_clips.json"))); PE.append(json.load(open(f"{ROOT}/results/{on}_leace_clips.json")))
out = {}; txt = []
for c, m in TG.items():
    S = [s for s in PR[0] if c in PR[0][s] and "orig" in PR[0][s] and all((s, c) in L for L in LID)]; res = {"n": len(S)}
    for name, PX in (("plain", PR), ("erased", PE)):
        sel, avg, orc, ctl = [], [], [], []
        for s in S:
            v = [tp(np.array(P[s][c]), m) for P in PX]; k = int(np.argmax([L[(s, c)] for L in LID])); sel.append(v[k]); avg.append(np.mean(v)); orc.append(max(v)); ctl.append(np.mean([tp(np.array(P[s]["orig"]), m) for P in PX]))
        sel, avg, orc, ctl = map(np.array, (sel, avg, orc, ctl)); bs = [(sel - ctl)[rng.integers(0, len(S), len(S))].mean() for _ in range(2000)]
        res[name] = {"selected_minus_control": float((sel - ctl).mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "average_minus_control": float((avg - ctl).mean()), "oracle_minus_control": float((orc - ctl).mean())}
    hit_sel = np.mean([int(LID[int(np.argmax([L[(s, c)] for L in LID]))][(s, c)] > 0.5) for s in S]); hit_avg = np.mean([np.mean([int(L[(s, c)] > 0.5) for L in LID]) for s in S]); res["P_language_gt_half_selected"] = float(hit_sel); res["P_language_gt_half_average"] = float(hit_avg); out[c] = res
    txt.append(f"{c} -> {m} (n={len(S)}): plain probe Delta selected {res['plain']['selected_minus_control']:+.3f} [{res['plain']['ci'][0]:+.3f},{res['plain']['ci'][1]:+.3f}], average over seeds {res['plain']['average_minus_control']:+.3f}, oracle {res['plain']['oracle_minus_control']:+.3f}; erased probe selected {res['erased']['selected_minus_control']:+.3f}, average {res['erased']['average_minus_control']:+.3f}; share of clips with P(intended language) > .5: selected {hit_sel:.2f}, average {hit_avg:.2f}")
json.dump(out, open(f"{ROOT}/results/bon_language.json", "w"), indent=1); open(f"{ROOT}/results/bon_language.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
