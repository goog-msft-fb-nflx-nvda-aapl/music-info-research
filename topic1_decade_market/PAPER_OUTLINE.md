# Time Traveler: paper outline (working draft; every claim points to a WORKLOG entry; to be converted to the target template)

## Working title and claims
Title candidates: "Time Traveler: era- and market-controlled cover generation"; "Moving a song along time and place".
Claims the current results support (probe-based, 24 evaluation songs for the era axis, 77 validation clips for the market axis, same-family probes unless stated):
1. Era axis: a learned era coordinate z (nuisance-controlled encoder, bias injection in a frozen MusicGen-Melody) moves the accompaniment along six decades (ordinal correlation .957 MERT); inference-time selection (best-of-N, probe-guided SMC) raises the held-out effect (WORKLOG 2026-11-02/03).
2. Market axis: the market label is carried mostly by the language of the vocal. Accompaniment-only and timbre-only routes do not move it (M-3, T-4a, T-4n); translating the lyrics and singing them in the target language moves the vocal market probe toward the target market for five targets (T-4k, two seeds) and the move disappears when language is erased from the probe (T-4l).
3. Composition: the era-axis accompaniment and the lyric route compose without a measurable interaction on 16 songs (T-10); era-specific vocal and mix-bus processing adds to the era gain (T-9d/e/f), super-additively for the 1980s.
4. Negative and methodological findings worth reporting: metadata-derived style descriptors move only Brazil on two independent instruments, and selecting words against one instrument inflates that instrument only (T-4e/o/p/q); the PA1 subset carries almost no same-composition pairs (proxy-pair analysis), so the guide's cover-pair questions need the original Discogs-VI audio; the lyric route trades language realization against song identity unless the original accompaniment is kept (T-4b/g/i).
Not claimed: human-perceived era or market (no listening test by decision), a market move beyond language, real cover-pair results.

## Sections
1. Introduction: cover generation as content-preserving style transfer (course slides, lecture 11b), two control axes, what is new (axis analysis with nuisance control and independent read-outs; lyric route for the market).
2. Related work (to be completed with opened papers, `SURVEY4.md`): era recognition (He et al. 2024; Tardieu et al. 2011), market/country prediction (Kostrzewa and Grabczynski 2024), cover and style transfer (MusicGen, JASCO, Demerle et al., Stable-Audio-Control, MuLaCover), singing voice conversion and translation (FreeSVC, PolySinger, LTAG), cross-cultural bias in generation (Music for All; Bias beyond Borders), conditioning by year (Meade et al. 2019).
3. Data and read-outs: Discogs-VI subset (A decades, B markets), Discogs-VI metadata, probes and their limits (single-family risk, language reading), independent instruments (tagger, CLAP, FD, KL).
4. Era axis: encoder, adapter, inference-time selection, vocal and mix-bus processing.
5. Market axis: what carries the market (stems, language erasure), routes tried and their outcomes, the lyric route (cover, lego, stem swap), scale-up, erasure.
6. Two-axis composition and the 2x2 design.
7. Pair analysis with the original Discogs-VI (pending data): metadata counts (T-1), proxy pairs, planned cover-vs-original test.
8. Limitations: probe-based evaluation, ASR transcripts of 30 s excerpts, one seed for most runs, no human study, legal limits on publishing outputs.
9. Reproducibility: code, commands (`REPRODUCE.md`, `repro/RUNS.md`), result files; the source audio and the lyric texts are not released.
