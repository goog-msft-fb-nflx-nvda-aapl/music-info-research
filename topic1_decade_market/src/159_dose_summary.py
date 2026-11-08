"""Per-dose summary of eval json files (results/<tag>_eval.json rows): mean MERT / MuQ expected decade index, chroma cosine to the source, and Spearman of the dose against each read-out, per tag. usage: python src/159_dose_summary.py --tags ax_onehot_pfx16_ge10_hpcp_dose ax_ctrlc_ge10_hpcp_dose ax_onehot_pfx16_ge10_hpcp_over --out results/dose_summary"""
import argparse, json, re, numpy as np
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); out = {}; txt = []
for t in a.tags:
    rows = json.load(open(f"results/{t}_eval.json"))["rows"]; conds = sorted({r["cond"] for r in rows if re.match(r"^[abc]-?[0-9.]+$", r["cond"])}, key=lambda c: float(c[1:])); res = {}
    for c in conds:
        rr = [r for r in rows if r["cond"] == c]; res[c] = {"n": len(rr), "mert": float(np.mean([r["mert_exp_decade"] for r in rr])), "muq": float(np.mean([r["muq_exp_decade"] for r in rr])), "chroma_cos": float(np.mean([r["chroma_cos"] for r in rr if "chroma_cos" in r]))}
    x = [float(c[1:]) for c in conds]; sp = {k: float(spearmanr(x, [res[c][k] for c in conds])[0]) for k in ("mert", "muq")}; out[t] = {"doses": res, "spearman": sp}
    txt.append(f"{t}: Spearman dose vs MERT {sp['mert']:+.2f}, MuQ {sp['muq']:+.2f}"); txt += [f"  {c}: MERT {v['mert']:.2f} MuQ {v['muq']:.2f} chroma_cos {v['chroma_cos']:.3f} (n={v['n']})" for c, v in res.items()]
json.dump(out, open(a.out + ".json", "w"), indent=1); open(a.out + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
