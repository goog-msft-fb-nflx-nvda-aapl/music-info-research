"""Grid measurement for a controllable-generation grid (era axis; the market axis plugs in with the same row format): from the probe rows of 10_eval_gen.py (clip, cond, seed, <probe>_probs) compute per condition: target-hit, mean P(target) and its gain over the neutral condition (clip-level bootstrap 95% CI), off-target mass on the adjacent decades, expected-decade response and its Spearman monotonicity over the requested decade (clip-mean), and the 6x6 requested-by-read confusion matrix.
usage: python src/145_grid_metrics.py results/<name>_eval.json [...] -> results/grid_metrics.json (+ .txt)"""
import sys, json, numpy as np
from scipy.stats import spearmanr
E6 = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; rng = np.random.default_rng(0); out = {}; txt = []
def boot(v, n=2000): v = np.asarray(v, float); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
for path in sys.argv[1:]:
    rows = json.load(open(path))["rows"]; name = path.split("/")[-1].replace("_eval.json", ""); out[name] = {}
    for pr in ("mert", "muq"):
        R = {}; neu = {}
        for x in rows:
            if pr + "_probs" not in x: continue
            P = np.array(x[pr + "_probs"])
            if x["cond"] == "neutral": neu.setdefault(x["clip"], []).append(P)
            elif x["cond"] in E6: R.setdefault(x["cond"], {}).setdefault(x["clip"], []).append(P)
        if not R: continue
        neu = {c: np.mean(v, 0) for c, v in neu.items()}; res = {}; conf = np.zeros((6, 6)); E_by = []
        for k, e in enumerate(E6):
            if e not in R: E_by.append(np.nan); continue
            clips = sorted(R[e]); Pm = {c: np.mean(R[e][c], 0) for c in clips}; allp = [p for c in clips for p in R[e][c]]
            hit = float(np.mean([np.argmax(p) == k for p in allp])); tp = [Pm[c][k] for c in clips]
            gain = [Pm[c][k] - neu[c][k] for c in clips if c in neu]; adj = [Pm[c][j] for c in clips for j in (k - 1, k + 1) if 0 <= j < 6]; off = [1 - Pm[c][k] - sum(Pm[c][j] for j in (k - 1, k + 1) if 0 <= j < 6) for c in clips]
            ex = float(np.mean([(Pm[c] * np.arange(6)).sum() for c in clips])); E_by.append(ex)
            for p in allp: conf[k, np.argmax(p)] += 1
            res[e] = {"n": len(allp), "hit": hit, "P_target": float(np.mean(tp)), "gain_vs_neutral": float(np.mean(gain)), "gain_ci": boot(gain), "mass_adjacent": float(np.mean(adj)), "mass_far": float(np.mean(off)), "expected_decade_idx": ex}
        ok = [i for i, v in enumerate(E_by) if not np.isnan(v)]; rho = float(spearmanr(ok, [E_by[i] for i in ok])[0]) if len(ok) > 2 else None
        out[name][pr] = {"per_era": res, "spearman_requested_vs_expected": rho, "confusion_rows_requested": (conf / conf.sum(1, keepdims=True).clip(1)).round(3).tolist()}
        txt.append(f"{name} {pr}: Spearman(requested, expected decade) {rho:.3f}; " + " | ".join(f"{e} hit {r['hit']:.2f} gain {r['gain_vs_neutral']:+.3f} [{r['gain_ci'][0]:+.3f},{r['gain_ci'][1]:+.3f}] far {r['mass_far']:.2f}" for e, r in res.items()))
json.dump(out, open("results/grid_metrics.json", "w")); open("results/grid_metrics.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
