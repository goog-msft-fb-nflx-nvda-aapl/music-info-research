# Progress snapshot (2026-11-17): what exists, what the numbers say, what is not supported

All numbers are measured and copied from `results/` (file names given). Interpretation lines are marked "Interpretation". Probes: MERT and MuQ linear probes on dataset A (decades), Discogs-EffNet tagger as an independent family; dataset B probe for market. Evaluation is automatic only (no listening test, by decision).

## System built so far
- Generators: MusicGen-Melody (frozen, adapter + melody/chroma conditioning) as the controlled testbed; ACE-Step 1.5 cover/lego/repaint and an era LoRA as second generator.
- Era axis: one-hot + 16-token prefix bias adapter with a dose α ∈ [0,1] interpolating between a 1960s prototype and a 2010s prototype; a control adapter (era-free conditioning) as comparison.
- Market axis: translated lyrics sung in a target language (Spanish, German) via ACE-Step vocal regeneration + stem swap with the original accompaniment.
- Evaluation tooling: probes (MERT, MuQ, EffNet), LEACE erasure of language, FAD_inf, R-Precision/MM-Dist/Diversity, Whisper language-ID.
- Public deliverables: project page (GitHub Pages) with discrete 2D plane tables and an interactive plane (accompaniment-only audio; vocal versions internal), weekly slides + video.

## Era axis: dose–response, three training seeds (MERT expected decade index, 0 = 1960s … 5 = 2010s; mean over seeds; `results/weekly_tables.json`, `dseed_summary.*`, `dseed3_summary.*`, `dfine_summary.*`)
| dose | a0 | a0.2 | a0.3 | a0.4 | a0.5 | a0.6 | a0.7 | a0.8 | a1 |
|---|---|---|---|---|---|---|---|---|---|
| adapter | .17 | .40 | .84 | 1.77 | 3.25 | 4.01 | 4.23 | 4.36 | 4.56 |
| control | .22 | .38 | .60 | 1.14 | 2.15 | 3.17 | 3.93 | 4.25 | 4.53 |
- Spearman along dose +1.00 in every seed; end points equal; the adapter sits above the control at a0.3–a0.7 on MERT and EffNet in all three seeds (a0.5 MERT adapter 3.35/3.39/3.00 vs control 2.41/1.88/2.14; EffNet 3.10/3.08/2.64 vs 2.39/2.10/2.08).
- Not consistent across seeds: FAD_inf / MuQ-FAD to real 2010s at a0.5 (adapter 36/68/51 vs control 47/39/44). **Supported claim: same decade at a lower dose; not supported: "more realistic audio".**
- Interpretation: the jump between a0.4 and a0.6 skips the 1980s–1990s (intermediate decades are the weak point, matching the G1 gap in `MASTER_PLAN.md`).

## Baselines on the generation-metric protocol (MERT R-Precision top-1 for 1960s / 1980s / 2010s; `gen_metrics_baselines_{mert,muq}.*`)
| system | 1960s | 1980s | 2010s |
|---|---|---|---|
| ours adapter | .972 | .792 | .389 |
| ours control | .986 | .597 | .236 |
| text-only MusicGen | .333 | .986 | .667 |
| MusicGen-Melody + text prompts | .597 | .986 | .681 |
| MuseControlLite | .889 | .111 | .083 |
| SongEcho (Stable Audio) | .847 | .153 | .167 |
| ACE-Step era LoRA | .688 | .896 | .292 |
| best-of-32 | .997 | .678 | .262 |
| SMC N=8 β=2 | 1.000 | .792 | .306 |
- Interpretation: R-Precision alone favours melody-less text-only generation; it has to be read with content preservation (the lecture's point: high similarity ≠ good cover, and generic metrics are unsuited). Steering collapsed (diversity ≈ 1.2).

## Inference-time search (probe-based, held-out MuQ 1980s; `best_of_n_*`, `smc_*`)
Best-of-N: .348 (N=1), .471 (8), .565 (16), .542 (32). SMC (8 particles, β=4): .407. Best-of-8 higher than SMC in 11 of 12 cells. Interpretation: search helps but costs N× inference → motivates amortisation (H4 in `02_direction_review.md`).

## Market axis: what carries it (`scale_market_probe*`, `market_encoder_*`, `lyric_route_eval*`)
- Accompaniment-only market encoder after erasing style/statistics/language: validation accuracy .22–.25 (chance .167).
- Voice conversion (FreeSVC) toward a market voice: no move.
- Translated lyrics sung in the target language (stem swap, 77 clips): vocal-probe ΔP(target) Spain +.188, Brazil +.421, Germany +.314, Italy +.408, US+UK +.325 (all CIs above 0; seed 1 replication Spain +.212, Brazil +.488, Germany +.308, Italy +.381, US+UK +.239). With language erased from the probe all five are within ±.03 of 0 → **the move is the language, not a musical market style.**
- Syllable-matched translation: no measurable change in the probe. Reference-audio style transfer: no consistent move.
- Interpretation: market control beyond sung language is **not achieved**; this is the open technical challenge (S5) and the motivation for a language-free market descriptor.

## Both axes: discrete 2D grid (16 songs × 3 seeds; `two_dose_grid.*`)
- Era dose effect on the era index of the *full mix*: +0.69 [+0.40, +1.08] (Spanish and German targets; at f = 0 %), i.e. the vocal compresses the range (the index sits at 3.4–3.7 even at dose 0 versus ≈ 0.2 for accompaniment-only). Language effect f = 100 % − 0 % on P(language): about +0.72 to +0.78; on P(market): Spanish +0.284 [+0.195, +0.378], German +0.315 [+0.170, +0.466]. The effect of language share is identical at every era dose (to the printed precision) and the era effect holds at every f.
- Interpretation: the two axes act separately (no measurable interaction), but the full-mix era read-out is weak when vocals are present → reading era from the accompaniment stem is the cleaner instrument, and an era-aware vocal chain (T-9) is part of the answer.

## Methodological findings worth a slide
- Same-family inflation: selecting words/particles against one probe inflates that probe only (T-4q: probe-selected words inflate the probes, Germany +.19/+.12, but not the tagger −.006).
- PA1 contains almost no same-composition pairs (proxy-pair analysis); real cover-pair questions need the original Discogs-VI audio (TA request pending).
- Lyric route trade-off: language realisation vs song identity (cover mode identity lost, VINet cosine .5; lego keeps identity .81–.83 but language hit .17–.33; stem swap identity .80–.84, language hit .25–.88).

## Not supported / not done
Market change beyond language; real cover-pair results; any human-perceived claim; "more realistic" claim; ACE-Step middle-decade control (LoRA pilot: 1990s/2000s zero); rhythm fit of translated lyrics (not yet measured).

## Running / pending (server)
Dose × era-guidance interaction (MusicGen-token adapter, guidance 5 and 15 at doses a0.3/0.5/0.7) finished generating on 2026-11-17 (433 files per guidance setting) and awaits scoring; no result yet.
