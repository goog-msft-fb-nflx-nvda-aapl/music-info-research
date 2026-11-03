# Survey prompt v4 (Time Traveler: era- and market-controlled cover generation), datasets first

Paste everything below the line into a deep-research tool. Save the reply as `survey4/<tool>_v4.md`.

---

You are helping with a graduate research project (music generation). Be rigorous: every paper, dataset and system must be real, with title, authors, venue/year and a URL or DOI that you actually opened; mark anything you could not verify as "UNVERIFIED"; never invent numbers; separate reported facts from your interpretation.

## Project ("Time Traveler")
Same-song cover generation controlled along two axes: (1) time (render a song as if released 20 years earlier or 30 years later) and (2) market (a song released in the United States rendered as if released in Spain). The professor said the TA can provide more data, for example Discogs-VI. Work so far: probes show decade information lives mostly in the accompaniment and market information mostly in vocals and language; era control of regenerated accompaniment (MusicGen-Melody with a learned era-axis adapter, keeping the original vocal and melody) works and is measured with codec-matched MERT/MuQ probes; market control from accompaniment alone is weak (6-way accuracy .22-.33, chance .167). Current data: the course's 30 s subset of Discogs-VI (1,290 clips in six decades, 1,002 clips in six 1980s markets), no artist/title/Discogs ids, audio not redistributable.

## Questions (for each: best available option, access route, license, size, caveats)
1. **Original Discogs-VI and Discogs-VI-YT (Araz et al., ISMIR 2024).** Exactly what is released (metadata, YouTube ids, audio features: which model and dimension), whether audio is downloadable, the request procedure, license of each part, and whether the YouTube terms of use permit research downloads. Is there an official or community copy of the audio?
2. **Other datasets with BOTH release year (or decade) AND release country/market, with audio or audio features**, usable for same-composition pairs: cover and version datasets (SHS100K, Da-TACOS, Second Hand Songs based sets, DALI, others), MusicBrainz/AcousticBrainz, Discogs data dumps, Million Song Dataset links, Free Music Archive, Jamendo, Music4All, any set with country labels. For each: how to obtain the audio, license, whether versions of the same work are grouped, whether country and year labels exist per recording, and how reissues are handled.
3. **Pair construction.** Methods to build trustworthy pairs of different recordings of the same song across decades or markets and to exclude reissues/remasters (Discogs master/release ids, MusicBrainz release groups, audio fingerprint or version-identification filters such as Discogs-VINet).
4. **Era and market prediction from audio**: current best systems and numbers (decade accuracy, MAE in years; country/region prediction), encoders used, and known confounds (language, genre, production, mastering, dataset bias).
5. **Controllable cover / style transfer generation**: systems that keep melody and lyrics while changing production style, era, genre or region (cover generation, style transfer, timbre transfer, accompaniment generation, singing voice conversion, accent or language conversion), with their conditioning mechanisms, evaluation metrics and failure modes. Include time-conditioned or year-conditioned music generation and any work on "what would this song sound like in another era or country".
6. **Evaluating a two-axis move**: how to measure that a generated clip moved along the intended axis (probes, classifiers, human tests), leakage between axes, melody retention, and the pitfalls of circular evaluation with probes trained on the same data.
7. **Legal and ethical points for using YouTube-linked or commercial audio in research and for publishing generated material** (what can be released: features, ids, generated clips, code).

## Output format
For every item: a short table of options (name, URL, size, labels, access, license), then the recommended route for a three-person academic team with a GPU server and no budget, then open risks. End with a ranked list of the three most promising data routes for building cross-decade and cross-country pairs and the first concrete step for each. Every claim carries a source link.
