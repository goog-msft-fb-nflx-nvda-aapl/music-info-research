"""Summarise MET (OpenNeuro ds008701) ratings.tsv. Usage: python3 -I scripts/met_ratings_summary.py
Fetches ratings.tsv from the OpenNeuroDatasets GitHub mirror (no EEG data downloaded)."""
import csv, io, urllib.request, statistics as st, collections
URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds008701/HEAD/ratings.tsv"
rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(URL).read().decode()), delimiter="\t"))
print(len(rows), "rows; participants", len({r["participant_id"] for r in rows}))
print("rows with Remark:", collections.Counter(r["Remark"] for r in rows if r["Remark"].strip()))
ok = [r for r in rows if not r["Remark"].strip()]
songs = collections.defaultdict(list)
for r in ok: songs[r["song_id"]].append(r)
print("distinct songs", len(songs), "listeners/song min,median,max",
      min(map(len, songs.values())), st.median(map(len, songs.values())), max(map(len, songs.values())))
print("song categories", collections.Counter(v[0]["Song Catergory"] for v in songs.values()))
for key in ("Valence", "Arousal"):
    vals = [int(r[key]) for r in ok]; gm = st.mean(vals); tot = sum((x-gm)**2 for x in vals)
    by_song = sum(len(v)*(st.mean(int(r[key]) for r in v)-gm)**2 for v in songs.values())
    subj = collections.defaultdict(list)
    for r in ok: subj[r["participant_id"]].append(int(r[key]))
    by_subj = sum(len(v)*(st.mean(v)-gm)**2 for v in subj.values())
    print(key, "histogram", sorted(collections.Counter(vals).items()),
          "| share of variance explained by song %.3f, by participant mean %.3f" % (by_song/tot, by_subj/tot))
