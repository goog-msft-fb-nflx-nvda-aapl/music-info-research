"""Ordinal control score of a generated set: for the six decade prompts (neutral excluded), per-clip expected decade E (codec-matched probe, MERT and MuQ); report (a) Spearman correlation between the prompted decade index and E over all clips (n = 24 songs x 6 prompts x 3 seeds), (b) the least-squares slope of E on the prompted index, (c) the range of the per-prompt mean E (last minus first decade), (d) the mean E of the neutral clips, and (e) song-level bootstrap 95% CIs of (a) (2000 resamples over the 24 songs).
usage: python 102_ordinal_control.py label=eval.json[+eval2.json] ... -> results/ordinal_control.{json,txt}   (OUT_PREFIX env sets the file name prefix)"""
import sys, os, json, numpy as np
from scipy.stats import spearmanr
D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; rng = np.random.default_rng(0); out = {}; txt = []
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); R = []
    for f in rest.split("+"): R += json.load(open(f))["rows"]
    songs = sorted({r["clip"] for r in R}); out[label] = {}; line = [label]
    for enc in ("mert", "muq"):
        E = lambda r: float(np.dot(r[f"{enc}_probs"], np.arange(6)))
        rows = [(D.index(r["cond"]), r["clip"], E(r)) for r in R if r["cond"] in D]; x = np.array([a for a, _, _ in rows]); e = np.array([c for _, _, c in rows]); cl = np.array([b for _, b, _ in rows])
        rho = spearmanr(x, e)[0]; slope = np.polyfit(x, e, 1)[0]; means = [float(np.mean([c for a, _, c in rows if a == k])) for k in range(6)]
        bs = []
        for _ in range(2000):
            pick = rng.choice(songs, len(songs)); m = np.concatenate([np.where(cl == s)[0] for s in pick]); bs.append(spearmanr(x[m], e[m])[0])
        neu = float(np.mean([E(r) for r in R if r["cond"] == "neutral"])) if any(r["cond"] == "neutral" for r in R) else None
        out[label][enc] = {"spearman": float(rho), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "slope": float(slope), "range": means[-1] - means[0], "mean_E": means, "neutral_E": neu}
        line.append(f"{enc}: rho {rho:.3f} [{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}] slope {slope:.3f} range {means[-1] - means[0]:.2f} E " + " ".join(f"{m:.2f}" for m in means) + (f" neutral {neu:.2f}" if neu is not None else ""))
    txt.append("\n  ".join(line))
PRE = os.environ.get("OUT_PREFIX", "results/ordinal_control"); json.dump(out, open(PRE + ".json", "w"), indent=1); open(PRE + ".txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
