"""Mean Discogs-EffNet expected decade index (0 = 1960s .. 5 = 2010s) per dose condition from results/discogs_decade_<name>.json (rows: cond, probs). usage: python src/160_effnet_dose.py --names dose ctrldose over --out results/effnet_dose_summary"""
import argparse, json, numpy as np
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); out = {}; txt = []
for n in a.names:
    rows = json.load(open(f"results/discogs_decade_{n}.json"))["rows"]; conds = sorted({r["cond"] for r in rows if r["cond"].startswith("a")}, key=lambda c: float(c[1:])); res = {c: float(np.mean([np.dot(r["probs"], np.arange(6)) for r in rows if r["cond"] == c])) for c in conds}
    sp = float(spearmanr([float(c[1:]) for c in conds], [res[c] for c in conds])[0]); out[n] = {"effnet_exp_decade": res, "spearman": sp}; txt.append(f"{n}: Spearman {sp:+.2f}; " + ", ".join(f"{c} {v:.2f}" for c, v in res.items()))
json.dump(out, open(a.out + ".json", "w"), indent=1); open(a.out + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
