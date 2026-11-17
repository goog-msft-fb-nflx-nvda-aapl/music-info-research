# Diagram specs (draw.io). Boxes short (≤ 4 words). Colours: Measured = solid blue, Hypothesis = dashed orange outline, external/existing = grey.

**D1 Positioning map (slide 2).** 2×2: x = "controls melody/lyrics", y = "controls era/market". Dots: MusicGen-Melody (melody only), SongEcho (melody F0), MPEcho (melody + phoneme), MuseControlLite (melody, dynamics, rhythm), Ours (melody + era + market). Only "Ours" occupies the upper-right cell; caption "Reported from the papers' abstracts/tables; none evaluates era or market control" [Reported].

**D2 Proposed pipeline (slide 6).** Left to right: Source song → (vocal F0 via RMVPE | phonemes + durations | chroma) → condition stream; Coordinate (era, market) → embedding; both → "IA-EiLM adapter" → frozen ACE-Step DiT (blocks) with a side box "layer-8 alignment to MERT (H2)" → decoded cover. Mark H1 = adapter, H2 = alignment, H5 = phoneme/duration stream as hypotheses (dashed). Source of structure: SongEcho Fig./Eq. (EiLM before FFN), MPEcho Fig. 2.

**D3 The 2D control plane (slide 4 or 6).** Square: x = market (original, es, de), y = era (1960s → 2010s). Cursor dot; arrows labelled "ω_market" (horizontal) and "ω_era" (vertical); a third dial "ω_melody" fixed. Caption: "Guidance scales per axis (H3, hypothesis)". Next to it the screenshot `figures/plane.png` (prototype, discrete) [Measured: prototype exists].

**D4 Evaluation panel (slide 7).** Four columns: Content (RPA/RCA/OA, chroma, song-ID) · Control (≥2 independent probes, dose Spearman, interaction) · Quality (FD, KL, CLAP, FAD_inf*) · Language (Whisper LID, PER). Footnote: "* does not rank our systems consistently [Measured]"; "no listening test (scope)".

**D5 Milestones (slide 8).** Horizontal timeline, five milestones each with its tangible artefact: M1 baselines on common protocol (done) · M2 H1 adapter + dose–response · M3 H3 continuous plane demo · M4 H4/H5 · M5 paper, experiments frozen 20 Dec 2026. Done = solid, others dashed.
