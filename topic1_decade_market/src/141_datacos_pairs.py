"""P-3: pair statistics of Da-TACOS metadata (Yesiler et al. ISMIR 2019; Zenodo 4717628, CC BY-NC-SA 4.0; metadata only, no audio): cliques (SecondHandSongs works) with at least two performances, release-year gaps between performances, share with MusicBrainz ids. usage: python 141_datacos_pairs.py <da-tacos_metadata dir> > results/datacos_pairs.json"""
import sys, json, itertools, collections, numpy as np
out = {}
for sub in ("benchmark", "coveranalysis"):
    d = json.load(open(f"{sys.argv[1]}/da-tacos_{sub}_subset_metadata.json")); gaps = []; npairs = 0; ncl = 0; mb = 0; nperf = 0; dec = collections.Counter(); wide = 0
    for w, perfs in d.items():
        ys = {p: int(v["release_year"]) for p, v in perfs.items() if str(v.get("release_year", "")).isdigit()}; nperf += len(perfs); mb += sum(1 for v in perfs.values() if v.get("mb_performances"))
        for v in perfs.values():
            if str(v.get("release_year", "")).isdigit(): dec[int(v["release_year"]) // 10 * 10] += 1
        if len(ys) >= 2:
            ncl += 1; g = [abs(a - b) for a, b in itertools.combinations(ys.values(), 2)]; gaps += g; npairs += len(g); wide += int(max(ys.values()) - min(ys.values()) >= 20)
    gaps = np.array(gaps); out[sub] = {"works": len(d), "performances": nperf, "performances_with_mbid": mb, "works_with_year_gap_info": ncl, "pairs": npairs, "gap_years_median": float(np.median(gaps)), "gap_ge_10": int((gaps >= 10).sum()), "gap_ge_20": int((gaps >= 20).sum()), "works_with_span_ge_20": wide, "performances_by_decade": dict(sorted(dec.items()))}
print(json.dumps(out, indent=1))
