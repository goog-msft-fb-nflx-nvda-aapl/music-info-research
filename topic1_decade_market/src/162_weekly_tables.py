"""Collects the dose-response numbers of the weekly report from existing result files (no new computation): MERT, MuQ and EffNet expected decade index per dose for the adapter and the control (seed 0), and MERT mean/std over the three training seeds. usage: python src/162_weekly_tables.py -> results/weekly_tables.json"""
import json, numpy as np
R = "results/"
def L(f): return json.load(open(R + f + ".json"))
def doses(f, tag): return L(f)[tag]["doses"]
A0 = {**doses("dose_summary", "ax_onehot_pfx16_ge10_hpcp_dose"), **doses("dfine_summary", "ax_onehot_pfx16_ge10_hpcp_dfine")}
C0 = {**doses("dose_summary", "ax_ctrlc_ge10_hpcp_dose"), **doses("dfine_summary", "ax_ctrlc_ge10_hpcp_dfine")}
EA = {**L("effnet_dose_summary")["dose"]["effnet_exp_decade"], **L("effnet_dfine_summary")["dfine"]["effnet_exp_decade"]}
EC = {**L("effnet_dose_summary")["ctrldose"]["effnet_exp_decade"], **L("effnet_dfine_summary")["ctrldfine"]["effnet_exp_decade"]}
key = lambda c: float(c[1:]); D = sorted(A0, key=key)
seed = {"adapter": {0: A0}, "control": {0: C0}}
for m, tg in (("adapter", "ax_onehot_pfx16_s{}_ge10_hpcp_{}"), ("control", "ax_ctrlc_s{}_ge10_hpcp_{}")):
    for s in (1, 2):
        d = {}
        for f in ("dseed_summary", "dseed3_summary", "dseed4_summary", "dfine_summary"):
            t = tg.format(s, "dseed" + {"dseed_summary": "", "dseed3_summary": "3", "dseed4_summary": "4", "dfine_summary": "2"}[f])
            try: d.update(doses(f, t))
            except KeyError: pass
        seed[m][s] = d
out = {"doses": D, "seed0": {"adapter": {c: {"mert": A0[c]["mert"], "muq": A0[c]["muq"], "effnet": EA.get(c)} for c in D}, "control": {c: {"mert": C0[c]["mert"], "muq": C0[c]["muq"], "effnet": EC.get(c)} for c in D}}, "mert_seeds": {}}
for m in seed:
    out["mert_seeds"][m] = {}
    for c in D:
        v = [seed[m][s][c]["mert"] for s in (0, 1, 2) if c in seed[m][s]]
        out["mert_seeds"][m][c] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n_seeds": len(v), "values": [float(x) for x in v]}
json.dump(out, open(R + "weekly_tables.json", "w"), indent=1)
for c in D: print(c, "MERT adapter %.2f (%d seeds, sd %.2f) control %.2f (%d, sd %.2f) | EffNet seed0 %s %s" % (out["mert_seeds"]["adapter"][c]["mean"], out["mert_seeds"]["adapter"][c]["n_seeds"], out["mert_seeds"]["adapter"][c]["sd"], out["mert_seeds"]["control"][c]["mean"], out["mert_seeds"]["control"][c]["n_seeds"], out["mert_seeds"]["control"][c]["sd"], *(("%.2f" % x if x is not None else "-") for x in (EA.get(c), EC.get(c)))))
