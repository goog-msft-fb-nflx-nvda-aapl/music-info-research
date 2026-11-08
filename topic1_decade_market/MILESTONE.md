# Milestone consolidation, 2026-11-08 (draft for the page update)

All numbers are copied from `WORKLOG.md` entries of 2026-11-07/08 and `RESULTS_SUMMARY.md`; interpretation is marked. Probe-based read-outs share training data with the controls; independent read-outs are named.

## 1. Where the audio is
- GitHub (`docs/era_market/audio`, `audio_index.json`, 88 files, 15 MB): accompaniment-only demo clips already on the page. Wavs are git-ignored; only OGG accompaniment-only files are published.
- Server `~/era_market/demo_keep/<config>/` (OGG + `manifest.json`, 48 MB in total, 16 configs: adapter dose / dose2 / hpcp / gridA / gridB, ctrlc hpcp, six steering sets, three controls). Not yet copied to the repository or Drive.
- Google Drive (restricted folder, backup via rclone server -> Drive): licensed source audio and vocal-containing material stay there only. Final backup and credential revocation are pending (TODO).
- Vocal market demos (translated lyrics sung over the original or generated accompaniment) contain material derived from commercial recordings: internal and presentation use only until the university's written determination.

## 2. Demonstrable discrete points (era x market)
Era: neutral, 1960s, 1980s, 2010s (hero examples), dose a = 0, .3, .5, .7, 1 (continuous path 1960s -> 2010s), 1970s/1990s/2000s available but weak. Market: original, Spanish, German (also pt, it for the vocal route); language share 0/25/50/75/100 %. Combined grid: 5 era doses x 5 language shares x {es, de}, 16 songs x 3 seeds (`results/two_dose_grid.*`). Per-clip demos: era x language demo clips are not yet rendered as a page table (TODO: render from `ace_a_swap*_voc` + `gen_..._hpcp` for the three hero songs; accompaniment-only for the public page).

## 3. Milestone findings (measured)
Era axis (adapter, one-hot+16 prefix, HPCP chroma):
- Dose response monotone: MERT expected decade .10 -> 4.54, MuQ .36 -> 4.35, Discogs-EffNet (independent, probe val acc .523) .64 -> 3.62; EffNet adapter .782 vs ctrlc .704.
- Per-cell FAD (MERT layer 22, PCA 32, n=72, sample-size extrapolation): 1960s 13.7 vs neutral 22.9; dose crossover between a0.3 and a0.5 (to real 1960s 14.1 -> 17.3 -> 29.1, to real 2010s 33.3 -> 17.8 -> 18.4). FAD does not separate 1970s-1990s from neutral.
- R-Precision top-1 (MERT, chance .167): 1960s .958, 1970s .847, 1980s .889, 1990s .722, 2000s .278, 2010s .389; MuQ 2010s .236.
- Diversity and MultiModality lower than neutral under conditioning (MERT diversity 6.4-7.6 vs 8.6).
Market axis (language is the carrier): language-erased probe removes the move (T-4l); language dose de P(Germany) .201 -> .517, es P(Spain) .083 -> .322; mix-level P(language) +.71 to +.78.
Both axes: era effect +.62 to +1.00 expected-decade index at every language share, language effect +.71 to +.78 at every era dose (CIs exclude 0, `two_dose_grid`); 3-seed grid: 20 of 24 interactions contain 0. Training-free steering (difference in means) is a null baseline for this model (content top-1 .06-.16 vs .958).

## 4. How generation quality and performance are assessed, and compared with previous work
Evaluation protocol (existing metrics only, `EVAL_PLAN.md`):
1. Controllability: probe hit rate and gain vs neutral (MERT/MuQ codec-matched probes), independent Discogs-EffNet, R-Precision and MM-Dist in embedding space, dose monotonicity (Spearman), cross-axis effects with song-bootstrap CIs.
2. Distribution quality: FAD per cell against real clips of the target decade (MERT, MuQ, CLAP), extrapolated to infinite sample size (Gui et al., arXiv 2311.01616), real-half floor as reference; FD-VGG / KL-PaSST / CLAP score (PA3 av-benchmark, existing results).
3. Diversity and MultiModality (Guo et al., CVPR 2022 and its reference [12]).
4. Cover constraint: content preservation (song-id top-1, own-chroma similarity, chord agreement, note F1).
Comparison with previous work, same protocol and same 24 songs (all measured, `RESULTS_SUMMARY.md`): text-prompt-only MusicGen (no melody), MusicGen-Melody with text decade prompts, MuseControlLite (Stable Audio Open, melody adapter; MERT 2010s +.171, 1960s +.015, 1980s +.014), ACE-Step 1.5 with era LoRA (MERT 1960s +.16-.24, 1980s +.42-.45, 2010s +.12-.16; song identity VINet top-1 .02-.19 vs .85), training-free steering (null), best-of-N and SMC selection. No published system exposes release era and market as joint controls, so a direct numeric comparison with the papers is only possible on shared sub-parts: FAD (embedding choice and reference reported), CLAP/KL/FD from the PA3 benchmark, and our re-runs of the available open systems (MuseControlLite, ACE-Step LoRA) under the same probes. Published numbers from other papers are not comparable (different data, embeddings, reference sets) and are not mixed into our tables.
Gaps: one adapter and 24 songs; read-outs share the encoder family with the probes; no listening test (decision). Open: MuQ-FAD with matched references and more seeds, Da-TACOS cover table (real cover pairs), MusicRFM/TADA-style steering re-runs.

## 5. Page update checklist (tomorrow)
- [ ] add sections: two-axis grid (table of es/de dose effects), dose response figure (MERT/MuQ/EffNet + FAD_inf), metric table (R-Precision/MM-Dist/Diversity/MultiModality), baselines table (section 4)
- [ ] copy accompaniment-only demo OGGs (era dose, adapter vs control) into `docs/era_market/audio`, update `audio_index.json`
- [ ] no course/assignment name, no GPU/Mac mention, no bug history
- [ ] rebuild with `src/92_make_page.py`
