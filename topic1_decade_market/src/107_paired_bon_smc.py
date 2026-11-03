"""N-5 paired comparison: best-of-N selection (pool of ax_ctrlc_bon32, first N seeds, selector = MERT probe after LEACE erasure of style and bandwidth) versus probe-guided SMC (N = 8 particles; sets smc_ctrlc_N8_b{2,4}[_f500]) and the unguided system (N = 1), per song and decade, same 24 songs (CPU, stored embeddings; usage: python 107_paired_bon_smc.py -> results/paired_bon_smc.{json,txt}).
Per (song, decade): value = P(target decade | clip) under the held-out probe minus the mean P(target decade | neutral clips of that set). Best-of-N value = the clip with the highest selector score among seeds 0..N-1; SMC value = mean over its seeds 0-2 clips (each already one SMC run of 8 particles); N = 1 value = mean over seeds 0-2 of the ctrlc clips. Paired difference per song (best-of-8 minus SMC) with song-bootstrap 95% CI (2000 resamples) and exact Wilcoxon signed-rank p over the 24 songs, per decade and probe (muq_unerased, muq_erased, mert_unerased). The selector and the probes are fitted on dataset-A TRAIN only, so no song-level tuning occurs and no selection/evaluation song split is needed."""
import sys, os, json, numpy as np
from scipy.stats import wilcoxon
sys.argv_ = sys.argv; sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "94_nuisance_steering.py")).read().split("for s in sys.argv[1:]:")[0]
exec(compile(src, "94_nuisance_steering_head", "exec"))
DEC = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; PN = ("muq_unerased", "muq_erased", "mert_unerased")
def load(S):
    Rr = json.load(open(f"{ROOT}/results/{S}_eval.json"))["rows"]; Zl = np.load(f"{ROOT}/results/{S}_eval_emb.npz")
    def probs(k, name):
        X = Zl[k]; ok = np.isfinite(X).all(1); f, m = M[(k, name)]; pr = np.full((len(X), 6), np.nan); pr[ok] = m.predict_proba(f(X[ok])); return pr
    PR = {"sel": probs("mert", "both"), "muq_unerased": probs("muq", "none"), "muq_erased": probs("muq", "both"), "mert_unerased": probs("mert", "none")}
    return PR, {(r["clip"], r["cond"], r["seed"]): i for i, r in enumerate(Rr)}, sorted({r["clip"] for r in Rr})
PRB, IB, songs = load("ax_ctrlc_bon32")
def neutral(PR, I, p, t, x, seeds): return np.mean([PR[p][I[(x, "neutral", sd)], t] for sd in seeds])
def vals_bon(N, p, t, e):
    out = []
    for x in songs:
        pool = list(range(N)); best = max(pool, key=lambda sd: PRB["sel"][IB[(x, e, sd)], t]); out.append(PRB[p][IB[(x, e, best)], t] - neutral(PRB, IB, p, t, x, range(32)))
    return np.array(out)
def vals_n1(p, t, e): return np.array([np.mean([PRB[p][IB[(x, e, sd)], t] for sd in range(3)]) - neutral(PRB, IB, p, t, x, range(3)) for x in songs])
SM = {k: load(f"smc_ctrlc_N8_{k}") for k in ("b4", "b2", "b4_f500", "b2_f500")}
def vals_smc(k, p, t, e):
    PR, I, ss = SM[k]; assert ss == songs
    return np.array([np.mean([PR[p][I[(x, e, sd)], t] for sd in range(3)]) - neutral(PR, I, p, t, x, range(3)) for x in songs])
def ci(d): bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {}; txt = []
for p in PN:
    for t, e in enumerate(DEC):
        b8, b16, b32, n1 = vals_bon(8, p, t, e), vals_bon(16, p, t, e), vals_bon(32, p, t, e), vals_n1(p, t, e); cell = {"N1": float(n1.mean()), "bon8": float(b8.mean()), "bon16": float(b16.mean()), "bon32": float(b32.mean())}
        for k in SM:
            s = vals_smc(k, p, t, e); d = b8 - s; cell[f"smc_{k}"] = float(s.mean()); cell[f"bon8_minus_smc_{k}"] = {"mean": float(d.mean()), "ci": ci(d), "wilcoxon_p": float(wilcoxon(d).pvalue), "songs_bon8_higher": int((d > 0).sum())}
        out[f"{p}|{e}"] = cell
        txt.append(f"{p} {e}: N1 {cell['N1']:+.3f} | smc b4 {cell['smc_b4']:+.3f} b4f500 {cell['smc_b4_f500']:+.3f} b2 {cell['smc_b2']:+.3f} | bon8 {cell['bon8']:+.3f} bon16 {cell['bon16']:+.3f} bon32 {cell['bon32']:+.3f} | bon8-smc_b4 {cell['bon8_minus_smc_b4']['mean']:+.3f} [{cell['bon8_minus_smc_b4']['ci'][0]:+.3f},{cell['bon8_minus_smc_b4']['ci'][1]:+.3f}] p={cell['bon8_minus_smc_b4']['wilcoxon_p']:.3f} ({cell['bon8_minus_smc_b4']['songs_bon8_higher']}/24)")
json.dump(out, open(f"{ROOT}/results/paired_bon_smc.json", "w"), indent=1); open(f"{ROOT}/results/paired_bon_smc.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
