# Direction review (2026-11-17): should the project change direction?

Inputs: `final_proj_pitch.pdf` (what a pitch must contain), `final_project_ideas.pdf` (topic 1 text), lecture 11b/09b/06/08b deep dive (`01_course_deep_dive.md`), our results (`03_progress_snapshot.md`). Statements about what *would* work are hypotheses; none has been run.

## 1. What the pitch has to show (final_proj_pitch.pdf, 5 minutes)
1. Technology innovation: what it is, where it comes from, why it fits the course.
2. Technical objectives and challenges: why hard, how approached, goals.
3. Feasibility: reference papers, datasets, codebases, GPU need, milestones with tangible intermediate results, quick pilots.
4. The team: ML experience, music background, lab.
Slides are sent to the TAs one day before; the TA laptop is used.

## 2. Verdict
**Keep topic 1 and the research question, change the technical core.** Reasons:
- The guide's topic 1 explicitly lists "era-controlled cover generation" as the extension; the course's cover-generation lecture (11b) calls the task content-preserving style transfer and the lecturer invites cover-generation projects. The direction fits.
- But our current core (an era adapter on MusicGen-Melody + probe-based analysis) lacks *method novelty*: the course's own state of the art for covers is **SongEcho → MPEcho on ACE-Step** (adapter with element-wise modulation; phoneme/duration conditioning; multi-condition guidance with APG), and the lab that gave the course is the group that wrote MPEcho. Neither paper conditions on era or market. That gap is where a contribution fits.
- What is already strong and should be kept as the evaluation backbone: nuisance-controlled probes (LEACE erasure of language, independent read-out families), the finding that selecting against one probe inflates that probe only, replicated dose–response over three training seeds, baselines on a common protocol.
- What is weak and should be fixed, not defended: (a) the generator is not the course SOTA (ACE-Step LoRA pilot gave 1990s/2000s ≈ 0 and low song identity); (b) the market axis is carried only by the *sung language* and the translation step has no rhythm/syllable constraint; (c) FAD / MuQ-FAD does not rank adapter vs control consistently, so "more realistic audio" cannot be claimed, only "same decade at lower dose".

## 3. Proposed technical core ("two-axis cover control on ACE-Step")
Hypotheses are numbered H1–H5; each has the experiment, the metric, and a pass rule fixed before the run.

**H1. Coordinate-conditioned element-wise modulation beats label/prefix conditioning.** Replace our one-hot+prefix bias adapter by a SongEcho-style IA-EiLM adapter whose condition is the 2-D coordinate (era ordinal, market embedding) concatenated with the melody stream. Experiment: train on the same pairs as before (Discogs-VI-derived era labels), evaluate dose–response on MERT/MuQ/EffNet probes at 3 training seeds. Pass: middle decades (1980s, 1990s) ΔP with CI above 0 on both probe families, melody RPA/RCA within 0.03 of SongEcho-melody-only. Compute: SongEcho-scale (49 M trainable; paper: 3×A100, 30 k steps).

**H2. Alignment to the base model's own SSL space at layer 8 helps era control.** ACE-Step aligns its 8th DiT layer to MERT/HuBERT (lecture 09b slide). Add a REPA-style auxiliary loss pulling pooled layer-8 features of the generated latent toward the MERT feature of a *real clip of the target era* (not of the same song), with an ordinal/era-probe head as the alignment target. Pass: higher ΔP at equal melody metrics than H1 alone on the three seeds. Risk: same-family probe inflation, so also report the independent EffNet and held-out MuQ read-outs (we already showed MERT-selected words inflate MERT-family probes only).

**H3. Decomposed multi-condition guidance gives a continuous, interaction-free 2D plane without retraining.** Replace dose interpolation by separate guidance scales (ω_era, ω_market, ω_melody) following MPEcho Eq. 4 / MuseControlLite Eq. 9 with APG. Experiment: grid ω_era × ω_market, read era index and P(language/market); pass = monotone response in each axis (Spearman ≥ 0.9) and cross-axis interaction within ±0.05 (the interaction metric we already use: `two_dose_grid`). A first MusicGen-token version of the dose × era-guidance grid has been generated (scoring pending); the diffusion version is the real test.

**H4. DPO with automatic rewards amortises inference-time search.** We measured that best-of-N and probe-guided SMC raise the effect (MuQ 1980s .348 → .565 at N=16; SMC .407); inference cost is the limitation. Build preference pairs from existing best/worst-of-N generations with a **nuisance-controlled reward** (era probe after LEACE, Whisper language-ID, melody similarity, SongEval) as in LeVo/JAM; one DPO run on the adapter. Pass: the DPO adapter at N=1 reaches ≥ 70 % of the best-of-16 gain at equal melody preservation, on held-out songs and on the *independent* probes.

**H5. Duration-aware translated lyrics make the market axis intelligible.** Use MPEcho's phoneme+length-regulator idea: translate lyrics under a syllable/duration budget taken from the source vocal (ASR word timestamps), supply target phonemes with durations. First step is the S2 measurement (rhythm fit and PER of the current translated vocals), then a syllable-constrained translation variant. MPEcho is Mandarin-only, so multilingual phoneme conditioning is an extension, not a claim. Pass: lower PER and better onset alignment than the unconstrained translation at equal language-ID hit rate.

Beyond-language market style (S5) stays a stated open problem; a BOSSA-style audio style embedding is the candidate, not part of the five hypotheses.

## 4. Evaluation protocol (to be shown on a slide; the lecture says evaluation is the field's open problem)
Four-part panel, every number tied to a stored result file:
1. **Content preservation:** RPA / RCA / OA on the vocal F0 (mir_eval, as SongEcho) + chroma/chord similarity + song-ID retrieval.
2. **Control:** era index and P(target market/language) from ≥ 2 independent probe families + EffNet tags; dose–response Spearman; cross-axis interaction.
3. **Quality / realism:** FD_openl3, KL_passt, CLAP, SongEval/Audiobox; FAD_inf with MERT and MuQ (reported with the caveat that it does not rank our systems consistently).
4. **Language:** Whisper language ID, PER/WER of the lyrics.
No listening test (decision recorded in `CLAUDE.md`); say so on the slide as a scope statement, not a hidden limit.

## 5. Feasibility facts for the pitch's "Feasibility" slide
- Reference papers read in full: SongEcho, MPEcho, MuseControlLite. Codebases: ACE-Step / ACE-Step 1.5, SongEcho and MPEcho (links on their pages), MuseControlLite, av-benchmark, our own repo (`goog-msft-fb-nflx-nvda-aapl/music-info-research`, scripts, exact commands, result JSONs).
- Data we hold: PA1 subset (A: 1,290 recordings, six decades; B: 1,002 recordings, six markets; licensed for the course, not redistributed), Discogs-VI metadata (33,578 cliques with a different-artist pair with era and market move; 3,172 US–Spain), TA request pending for Discogs-VI audio/LargeSHS/MTG.
- Compute: a shared multi-GPU server (4× H200 class; at the last check one GPU free); adapter training at SongEcho scale fits one GPU. (Course sheet asks for GPU need; the project's public-page rule is not to mention GPUs, so decide whether this line goes on the slide.)
- Milestones with tangible intermediates: M1 baseline reproduction on the common protocol (done for MusicGen side: `gen_metrics_baselines_*`); M2 H1 adapter on ACE-Step (adapter + dose–response); M3 H3 guidance plane (continuous plane demo page, which already exists in discrete form); M4 H4/H5; M5 paper (target ICML 2027; experiments frozen by 20 Dec 2026).
- Pilot already running/ran: dose–response replicated on 3 seeds; discrete 2D plane page; baselines table; S2 rhythm measurement is next.

## 6. Risks and fallbacks
- ACE-Step era signal could stay weak (the LoRA pilot did). Fallback: keep MusicGen-Melody as the controlled testbed for H3/H4 (method papers can use the smaller model) and report ACE-Step as transfer.
- Multilingual phoneme conditioning has no public recipe (MPEcho is Mandarin); fallback: H5 reduced to the measurement + syllable-constrained translation with the existing ACE-Step vocal route.
- Probe circularity: mitigated by independent probe families, LEACE, and by reporting held-out MuQ/EffNet; a human check is out of scope by decision.
- Data legal limits: vocal-containing outputs from licensed recordings stay internal; public demos are accompaniment only.
- Novelty collision: FlexCover (NeurIPS 2026 poster) and MPEcho target melody/lyrics control; none of the papers opened evaluates era/market control. Re-run the search before the paper (TODO item "repeat survey").
