"""T-4e metadata-derived market descriptors: style (and genre) distributions per release country and decade from the Discogs-VI metadata (first track of each version; release_styles are [genre, style] pairs; a style counts once per version). For the six dataset-B markets (US, UK, Brazil, Spain, Germany, Italy) and decades 1960s-2010s: lift of a style in (market, decade) = P(style | market, decade) / P(style | decade, all countries), reported for styles with >= 200 versions in the cell; also P(style | market, decade) so that frequent styles are visible. 'top_kl' = the eight styles with the largest contribution P(style | market, decade) x log(lift) among styles with lift > 1 (distinctive and frequent). Output: results/market_styles.{json,txt}. The style names follow the Discogs taxonomy used by Discogs-EffNet ('Genre---Style').
usage: unzip -p data/dvi/main.zip main/Discogs-VI-20240701.jsonl | python 113_market_styles.py"""
import sys, json, re
import numpy as np
from collections import Counter, defaultdict
MK = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]; cell = defaultdict(Counter); cn = Counter(); dec = defaultdict(Counter); dn = Counter()
for ci, line in enumerate(sys.stdin):
    c = json.loads(line)
    for v in c["versions"]:
        if not v["tracks"]: continue
        t = v["tracks"][0]; yr = (t.get("released") or "")[:4]; ct = (t.get("country") or "").strip()
        if not re.fullmatch(r"\d{4}", yr) or not (1950 <= int(yr) <= 2024): continue
        d = f"{int(yr) // 10 * 10}s"; st = {f"{p[0]}---{p[1]}" for p in (t.get("release_styles") or []) if len(p) == 2}
        dn[d] += 1; dec[d].update(st)
        if ct in MK: cn[(ct, d)] += 1; cell[(ct, d)].update(st)
    if ci % 100000 == 0: print("clique", ci, file=sys.stderr, flush=True)
out = {}; txt = []
for m in MK:
    for d in ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]:
        n = cn[(m, d)]; rows = []
        for s, k in cell[(m, d)].items():
            if k >= 200: rows.append((s, k / n, (k / n) / (dec[d][s] / dn[d]), k))
        top_lift = sorted(rows, key=lambda r: -r[2])[:6]; top_freq = sorted(rows, key=lambda r: -r[1])[:6]; top_kl = sorted([r for r in rows if r[2] > 1], key=lambda r: -(r[1] * np.log(r[2])))[:8]
        out[f"{m}|{d}"] = {"versions": n, "top_lift": [{"style": s, "p": p, "lift": l, "n": k} for s, p, l, k in top_lift], "top_freq": [{"style": s, "p": p, "lift": l, "n": k} for s, p, l, k in top_freq], "top_kl": [{"style": s, "p": p, "lift": l, "n": k} for s, p, l, k in top_kl]}
        txt.append(f"{m} {d} (n={n}): kl " + "; ".join(f"{s} {l:.1f}x ({p:.3f})" for s, p, l, k in top_kl[:4]) + " | lift " + "; ".join(f"{s} {l:.1f}x ({p:.3f})" for s, p, l, k in top_lift) + " | freq " + "; ".join(f"{s} {p:.3f}" for s, p, l, k in top_freq))
json.dump(out, open("results/market_styles.json", "w"), indent=1, ensure_ascii=False); open("results/market_styles.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt[:6]))
