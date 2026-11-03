"""Metadata-only candidate pairs for the two-axis (era x market) cover study, from the Discogs-VI metadata (Araz et al. 2024; CC BY-NC-SA 4.0), no audio. usage: unzip -p main.zip main/Discogs-VI-20240701.jsonl | python 106_discogs_vi_pairs.py -> results/discogs_vi_pairs.{json,txt}
Per version the FIRST track's metadata is used (release_id, master_id, country, released year, track artist ids, YouTube urls of the release). Eligible version: valid year (1900-2024), known country. Pair = two eligible versions of one clique (same composition by Discogs-VI) with different release_id. The format list of the metadata has no edition descriptors (Reissue, Remastered, Repress are NOT present), so reissues cannot be removed by label; a pair is called 'strict' when the track-artist id sets differ (a performance by another artist, a cover) and 'same-artist' otherwise (a re-release, a re-recording, or another edition: not separable without audio).
Criteria: era move = year gap >= 20 (and different decade); market move = different release country; 'six' = both countries in {US, UK, Brazil, Spain, Germany, Italy}; US->Spain = one version US and the other Spain. Counts: cliques with >= 1 qualifying pair, and pairs (cliques with more than 60 eligible versions are subsampled to 60 with a fixed seed for the pair counts; their clique-level flags use all versions). 'with video' = both versions have at least one YouTube url in the release metadata."""
import sys, json, re, itertools, random
from collections import Counter, defaultdict
SIX = {"US", "UK", "Brazil", "Spain", "Germany", "Italy"}; rnd = random.Random(0)
cells = defaultdict(lambda: {"cliques": set(), "pairs": 0}); n_cl = n_v = n_el = 0; ncl_el2 = 0
def add(key, ci, k=1):
    c = cells[key]; c["cliques"].add(ci); c["pairs"] += k
for ci, line in enumerate(sys.stdin):
    c = json.loads(line); vs = []
    for v in c["versions"]:
        n_v += 1
        if not v["tracks"]: continue
        t = v["tracks"][0]; yr = (t.get("released") or "")[:4]; ct = (t.get("country") or "").strip()
        if not re.fullmatch(r"\d{4}", yr) or not (1900 <= int(yr) <= 2024) or not ct: continue
        vs.append((int(yr), ct, t.get("release_id", ""), frozenset(t.get("track_artist_ids") or []), bool(t.get("release_videos"))))
    n_cl += 1; n_el += len(vs)
    if len(vs) < 2: continue
    ncl_el2 += 1
    sub = vs if len(vs) <= 60 else rnd.sample(vs, 60); sub_ids = {id(x) for x in sub}
    for a, b in itertools.combinations(vs, 2):
        if a[2] == b[2]: continue
        inpair = id(a) in sub_ids and id(b) in sub_ids
        strict = a[3] != b[3]; dy = abs(a[0] - b[0]); era = dy >= 20 and a[0] // 10 != b[0] // 10; mk = a[1] != b[1]; six = a[1] in SIX and b[1] in SIX; us_es = {a[1], b[1]} == {"US", "Spain"}; vid = a[4] and b[4]
        for sname, sflag in (("strict", strict), ("any", True)):
            if not sflag: continue
            def put(name, cond):
                if cond: add((sname, name), ci, 1 if inpair else 0); add((sname + "+video", name), ci, 1 if (inpair and vid) else 0) if vid else None
            put("all pairs", True); put("era move", era); put("market move", mk); put("market move six", mk and six); put("era+market", era and mk); put("era+market six", era and mk and six); put("era+US-Spain", era and us_es)
    if ci % 100000 == 0: print("clique", ci, file=sys.stderr, flush=True)
out = {"versions": n_v, "cliques": n_cl, "eligible_versions": n_el, "cliques_with_2_eligible": ncl_el2, "cells": {f"{k[0]} | {k[1]}": {"cliques": len(v["cliques"]), "pairs": v["pairs"]} for k, v in sorted(cells.items())}}
json.dump(out, open("results/discogs_vi_pairs.json", "w"), indent=1)
txt = [f"versions {n_v}; cliques {n_cl}; eligible versions (valid year, known country) {n_el}; cliques with >= 2 eligible {ncl_el2}"] + [f"{k}: cliques {v['cliques']}, pairs {v['pairs']}" for k, v in out["cells"].items()]
open("results/discogs_vi_pairs.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
