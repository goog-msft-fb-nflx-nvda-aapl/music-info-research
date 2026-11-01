"""N-5a: best-of-N selection with a nuisance-controlled scorer (inference-time, no training; CPU, stored embeddings).
Pool: N_MAX generation seeds per (song, decade prompt) of one era-axis system (stored eval json + eval_emb.npz + content json of the same set). Scorer (selector): MERT codec-probe retrained after LEACE erasure of style (10 PCs of Discogs-EffNet tags) and bandwidth statistics (9 accompaniment statistics) from the feature space (as 94_nuisance_steering.py, 'both'), score = P(target decade | clip). Per song and decade the clip with the highest score among the first N seeds is kept (N = 1, 2, 4, 8; N = 1 is the average over the single seeds = random choice). Reported on the kept clips (24 songs, 95% song-bootstrap CI): Delta of P(target) - P(. | neutral) with the neutral reference = mean over all pool seeds, under (a) the selector probe (MERT erased; optimistic, selection-biased), (b) held-out MuQ unerased probe, (c) held-out MuQ erased probe, (d) MERT unerased probe, plus mean song-identity top-1 and chord agreement of the kept clips (content json).
usage: python 103_best_of_n.py <set> [N_MAX=8]   -> results/best_of_n_<set>.{json,txt}   (run in the era_market working directory with the venv of 94_nuisance_steering.py)"""
import sys, os, json, numpy as np, torch
sys.argv_ = sys.argv; S = sys.argv[1]; NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 8; sys.argv = [sys.argv[0]]
import importlib.util
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "94_nuisance_steering.py")).read().split("for s in sys.argv[1:]:")[0]
exec(compile(src, "94_nuisance_steering_head", "exec"))
Rr = json.load(open(f"{ROOT}/results/{S}_eval.json"))["rows"]; Zl = np.load(f"{ROOT}/results/{S}_eval_emb.npz"); Cr = json.load(open(f"{ROOT}/results/{S}_content.json")); Cr = Cr if isinstance(Cr, list) else Cr["rows"]
C = {(r["clip"], r["cond"], r["seed"]): r for r in Cr}; DEC = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; songs = sorted({r["clip"] for r in Rr})
def probs(k, name):
    X = Zl[k]; ok = np.isfinite(X).all(1); f, m = M[(k, name)]; pr = np.full((len(X), 6), np.nan); pr[ok] = m.predict_proba(f(X[ok])); return pr
PR = {"sel_mert_erased": probs("mert", "both"), "muq_unerased": probs("muq", "none"), "muq_erased": probs("muq", "both"), "mert_unerased": probs("mert", "none")}
idx = {(r["clip"], r["cond"], r["seed"]): i for i, r in enumerate(Rr)}; seeds = sorted({r["seed"] for r in Rr if r["seed"] < NMAX}); out = {}; txt = [f"{S}: pool seeds {seeds}, {len(songs)} songs"]
for N in (1, 2, 4, 8):
    if N > len(seeds): continue
    res = {}
    for pname, pr in PR.items():
        cells = {}
        for t, e in enumerate(DEC):
            neu = np.array([np.mean([pr[idx[(x, "neutral", sd)], t] for sd in seeds]) for x in songs]); vals = []
            for x in songs:
                sc = lambda sd: PR["sel_mert_erased"][idx[(x, e, sd)], t]
                if N == 1: vals.append(np.mean([pr[idx[(x, e, sd)], t] for sd in seeds]))
                else: pool = seeds[:N]; best = max(pool, key=sc); vals.append(pr[idx[(x, e, best)], t])
            d = np.array(vals) - neu; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; cells[e] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        res[pname] = cells
    ct = []; ch = []
    for x in songs:
        for t, e in enumerate(DEC):
            pool = seeds if N == 1 else seeds[:N]; sel = pool if N == 1 else [max(pool, key=lambda sd: PR["sel_mert_erased"][idx[(x, e, sd)], t])]
            ct.append(np.mean([C[(x, e, sd)]["top1"] for sd in sel])); ch.append(np.mean([C[(x, e, sd)]["chord_agree"] for sd in sel]))
    res["content"] = {"top1": float(np.mean(ct)), "chord_agree": float(np.mean(ch))}; out[f"N{N}"] = res
    txt.append(f"N={N}: " + " ; ".join(f"{p}: " + " ".join(f"{e[:4]} {c['delta']:+.3f}" for e, c in cells.items()) for p, cells in res.items() if p != "content") + f" ; top1 {res['content']['top1']:.3f} chord {res['content']['chord_agree']:.3f}")
json.dump(out, open(f"{ROOT}/results/best_of_n_{S}.json", "w"), indent=1); open(f"{ROOT}/results/best_of_n_{S}.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
