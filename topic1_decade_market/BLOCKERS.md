# Blockers, evidence and research ideas (Time Traveler: era x market covers)

Maintained 2026-11-03. Each blocker lists the evidence (result files), the reading of the evidence (interpretation, marked), literature found by search (titles found by web search, to be opened and verified before citing), new ideas, and the experiment IDs in `TODO.md`. Only measured numbers appear as evidence.

## B1. The market axis does not move through the accompaniment or the singer's timbre (partly resolved 2026-11-04 by the lyric route, see Status)
Status 2026-11-04: translating the lyrics and singing them in the target language (cover route) while keeping the original accompaniment (stem swap) moves the vocal market probe toward the target market: Delta P(target) vs the English control Spain +.19, Brazil +.53, Germany +.27, Italy +.33, CIs above 0, 12 clips (WORKLOG 2026-11-04, T-4i/T-4j); replicated on 77 clips of all source languages and five targets, Delta .19 (Spain) to .42 (Brazil), all CIs above 0 (T-4k). What it shows: the language cue moves; open: scale, seeds, arrangement cues, composition with the era axis.
Evidence
- Accompaniment-only market encoder after erasing style, statistics and language: validation accuracy .22-.25 (unerased accompaniment .33 MERT, .28 MuQ; chance .167): WORKLOG 2026-11-03 (M-3), `results/market_encoder_*`.
- Vocal market accuracy .608 falls to .392 after erasing language (Whisper large-v3 groups): WORKLOG 2026-10-27.
- FreeSVC timbre conversion to a market voice: target-market accuracy .163 / .175 (chance .167); the accuracy loss on the original label equals the loss of the self-voice control (.451 vs .446 MERT; .196 vs .203 MuQ): WORKLOG 2026-11-03 (T-4a), `results/svc_market_eval*.json`.
- Earlier text-prompt market covers: weak (Spain, UK not significant): WORKLOG 2026-10-26 entry 9.
Reading (interpretation): the market cue lives in the language and delivery of the vocal, not in accompaniment texture or voice timbre.
Literature to open: Music for All, representational bias and cross-cultural adaptability of music generation models (arXiv 2502.07328, NAACL 2025 (arXiv comments line): adapters on MusicGen and Mustango for non-Western traditions); Bias beyond Borders, GlobalDISCO (arXiv 2510.01963: quality and alignment gaps across 79 countries); a cross-cultural evaluation of taste-sound correspondences in AI-generated music (arXiv 2608.03433; it does not attribute bias to the text encoder, an earlier search summary claiming that was wrong).
Ideas and experiments
- I1 language route (T-4b): translated lyrics sung in the target language. Pilot done: language moves (top-1 hit .33-.58) but song identity is lost (B2). Running: cover-strength sweep.
- I2 singable translation: translate with melody constraints (syllable count, tones). Literature to open: LTAG, translating lyrics jointly with melody alignment (arXiv 2303.15705, EMNLP Findings 2023); GagaST, song translation for tonal languages (arXiv 2203.13420); PolySinger, singing voice to singing voice translation English to Japanese (arXiv 2407.14399, ISMIR 2024). Experiment T-4f: re-time the translated lyrics to the source syllable grid before synthesis.
- I3 metadata-derived market descriptors instead of country names: Discogs release styles per (country, decade) from 1.9M versions give a data-grounded text condition (T-4e: `src/113_market_styles.py`, `results/market_styles.*`), evaluated with the Discogs-EffNet style tagger on generated audio.
- I4 compose language and style: lyric route for the language cue plus style-descriptor conditioning for the arrangement cue; test additivity in the 2x2 design (T-6).

## B2. The lyric route loses the song
Evidence: control (English lyrics, same pipeline) already has VINet cosine .496 to the original mix and top-1 0 of 12; translated .435-.483; the accompaniment-only ACE cover of 2026-10-28 also kept less of the source than chroma-conditioned MusicGen-Melody (WORKLOG 2026-10-28).
Reading (interpretation): ACE-Step cover mode at strength 0.5 regenerates melody and arrangement.
Ideas and experiments
- Higher cover strength (0.8, 0.95): done (WORKLOG 2026-11-04): cosine to the original rises by .03-.07 (.49-.52), identity top-1 stays <= .33, language hit up to .75; the control has cosine .50 at every strength.
- Vocal-only regeneration (T-4g, ACE-Step base model lego mode, original accompaniment kept): done (WORKLOG 2026-11-04): identity restored (cosine .81-.83, top-1 .83-.92) but the language is realized less often (hit .17-.33 vs .50-.75 for the cover route): the routes trade language against identity. Open: louder or longer vocal track, guidance sweep (running), language read-out on the isolated track, melody-driven synthesis with the original F0 so that identity and language are decoupled (T-4h).
- Melody-conditioned synthesis: extract the vocal F0/MIDI (RMVPE, already inside FreeSVC) and drive a lyric-conditioned synthesizer; check the licences of candidates first.

## B3. Same-song pairs with year and country need audio we do not have
Evidence: PA1 manifest hides artist, title and Discogs ids; Discogs-VI metadata gives 33,578 cliques with a different-artist pair with era and market move (WORKLOG 2026-11-03, T-1) but the audio is not distributed (SURVEY4.md).
Ideas: TA's answer (request pending); MTG features on request; licensed-audio control corpora (FMA, MTG-Jamendo) for method development; metadata-only analyses (T-1, T-4e) in the meantime.

## B4. No training data labeled on both axes
Evidence: PA1 A has decades (US), B has markets (1980s only): no (decade, market) combinations.
Ideas: factorized conditioners (era adapter, market via lyrics/style) and an additivity test on generated combinations (T-5, T-6): the interaction term of the 2x2 design is the evidence of independence.

## B5. Evaluation depends on probes trained on the same data family
Evidence: era and market probes are MERT/MuQ logistic heads trained on PA1; held-out encoder differences are reported (paired best-of-N vs SMC, WORKLOG 2026-11-03).
Ideas: independent families (production-feature baseline after Tardieu et al., DAFx 2011, to open; Discogs-EffNet style tags; CLAP text score); metadata-grounded style tagger read-out (T-4e); no human test (decision), so report agreement across families.

## B6. The era axis changes the accompaniment only
Evidence: the vocal is kept from the original in all era-axis runs; vocal stems carry decade information (dataset A vocals .39 accuracy, WORKLOG 2026-10-25).
Ideas: era-specific vocal processing chain (bandwidth, saturation, compression, reverb as parametrized DSP) applied to the original vocal, scored with the era probes on the remix (T-9).

## B7. The lyric-route market move is language only
Evidence: after LEACE erasure of the language group from the vocal market probe the move disappears (Delta P(target) within +-.03 for all five targets; WORKLOG 2026-11-04, T-4l), while the plain probe moves by +.19 to +.42 (T-4k).
Reading (interpretation): language is one cue of the market; the cover route changes the language and keeps the arrangement of the original accompaniment, so the remaining market cues are not moved.
Ideas and experiments
- I5 metadata-derived style descriptors for the accompaniment (T-4e/o/p/q, done): only Brazil moves on two independent instruments; selecting words against one instrument inflates that instrument only (WORKLOG 2026-11-04).
- I5b (open) a third, selection-independent instrument for market-style claims (for example CLAP text-audio similarity to a market caption) and a larger evaluation set.
- I6 reference-audio style transfer: ACE-Step takes a reference audio for style; use real clips of the target market (dataset-B TRAIN clips of the market and decade) as the reference with the original melody kept (T-4m); read-outs: Discogs-EffNet style tags, accompaniment market probe (M-3), language-erased vocal probe.
- I7 vocal delivery: accent and singing style are not isolated by a language swap; test a conversion that keeps the language but changes the singer to a native singer of the target market (FreeSVC was not effective with a mean-embedding voice, T-4a; a per-reference nearest-voice choice is untested): T-4n.
- I8 market-conditioned arrangement from the Discogs-VI metadata: condition the accompaniment adapter on the style distribution of (market, decade) instead of a label; needs audio of Discogs-VI-YT (B3).
