"""Figures from results/six_decades.json (written by 78_six_decades.py): (1) expected probe decade E per prompted decade, (2) probe-probability change vs neutral with 95% song-bootstrap CI.
usage: python 91_fig_six_decades.py -> results/figs/six_decades_E.png, results/figs/six_decades_delta.png. Configs: base_ge7, lora_ge{7,10,15,20}; seed adapters are drawn as thin lines of the same ge."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("results/six_decades.json")); D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
main = {"base_ge7": "#7a7a7a", "lora_ge7": "#9ecae1", "lora_ge10": "#4292c6", "lora_ge15": "#2171b5", "lora_ge20": "#08306b"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, enc in zip(ax, ("mert", "muq")):
    a.plot(range(6), range(6), ls=":", c="k", lw=1, label="expected decade = prompted decade")
    for k in d:
        if "seed" in k: a.plot(range(6), [d[k][f"{enc}_{x}"]["expected_decade"] for x in D], c=main["lora_" + k.split("_")[1]], lw=0.8, alpha=0.6, ls="--")
    for k, c in main.items(): a.plot(range(6), [d[k][f"{enc}_{x}"]["expected_decade"] for x in D], c=c, lw=2, marker="o", ms=4, label=k)
    a.set_xticks(range(6)); a.set_xticklabels(D); a.set_xlabel("prompted decade"); a.set_title(enc.upper() + " codec probe"); a.grid(alpha=0.3)
ax[0].set_ylabel("expected decade index of the probe (0 = 1960s)"); ax[0].legend(fontsize=7, loc="upper left"); fig.text(0.5, -0.02, "dashed thin lines: seed adapters (same guidance as the matching solid line)", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig("results/figs/six_decades_E.png", dpi=160, bbox_inches="tight")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); w = 0.16
for a, enc in zip(ax, ("mert", "muq")):
    for i, (k, c) in enumerate(main.items()):
        m = np.array([d[k][f"{enc}_{x}"]["delta"] for x in D]); lo = np.array([d[k][f"{enc}_{x}"]["ci"][0] for x in D]); hi = np.array([d[k][f"{enc}_{x}"]["ci"][1] for x in D])
        a.bar(np.arange(6) + (i - 2) * w, m, w, color=c, label=k, yerr=[m - lo, hi - m], error_kw=dict(lw=0.7, capsize=1.5))
    a.axhline(0, c="k", lw=0.8); a.set_xticks(range(6)); a.set_xticklabels(D); a.set_xlabel("prompted decade"); a.set_title(enc.upper() + " codec probe"); a.grid(alpha=0.3, axis="y")
ax[0].set_ylabel("change in P(prompted decade) vs neutral"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, loc="lower center", ncol=5)
fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig("results/figs/six_decades_delta.png", dpi=160, bbox_inches="tight")
