"""P-3b: MusicBrainz country/date for Da-TACOS cover-analysis performances that have `mb_performances` ids (recording MBIDs): GET /ws/2/recording/<mbid>?inc=releases&fmt=json (1 request/s, public API, User-Agent names the project repository, no personal contact). Writes results/datacos_mb_country.json: per performance id the list of (release country, date) of the matched recordings. usage: python 142_mb_country.py <da-tacos_metadata dir> [N]"""
import sys, json, time, os, urllib.request
d = json.load(open(f"{sys.argv[1]}/da-tacos_coveranalysis_subset_metadata.json")); N = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9; fp = "results/datacos_mb_country.json"; res = json.load(open(fp)) if os.path.exists(fp) else {}
todo = [(p, list(v["mb_performances"])[0]) for w, ps in d.items() for p, v in ps.items() if v.get("mb_performances") and p not in res][:N]
for i, (p, m) in enumerate(todo):
    req = urllib.request.Request(f"https://musicbrainz.org/ws/2/recording/{m}?inc=releases&fmt=json", headers={"User-Agent": "TimeTravelerResearch/0.1 (https://github.com/goog-msft-fb-nflx-nvda-aapl/music-info-research)"})
    try: r = json.load(urllib.request.urlopen(req, timeout=30)); res[p] = {"mbid": m, "releases": [{"country": x.get("country"), "date": x.get("date")} for x in r.get("releases", [])]}
    except Exception as e: res[p] = {"mbid": m, "error": str(e)[:80]}
    if i % 50 == 0: json.dump(res, open(fp, "w")); print(i, len(todo), flush=True)
    time.sleep(1.1)
json.dump(res, open(fp, "w"))
