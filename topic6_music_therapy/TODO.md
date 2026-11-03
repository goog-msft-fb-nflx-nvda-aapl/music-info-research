# Topic 6 TODO

## Verification (before any experiment)
- [x] NMED-T and NMED-E Stanford pages checked (WORKLOG 2026-11-03). NMED-H / NMED-M and the ISMIR paper still to check.
- [x] Both EEG-to-music papers exist (Postolache ICASSP 2025; Shen TMM 2026). Still to do: read both in full (data, metrics, code); find the DEAP subject-identity paper or drop the claim.
- [x] Stable Audio Open 1.0, AudioLDM2-music checked. Still to do: MERT-v1-95M, official CBraMod/LaBraM weights and licenses, DEAM/PMEmo, fadtk.
- [ ] DEAP access: EULA / download route; confirm preprocessed 128 Hz version.
- [ ] Decide with team: NMED-T audio sourcing and public-demo rights (no redistribution of copyrighted audio).

## Experiments (in order, each commits scripts + launch commands + results)
- [ ] E1 DEAP LOSO VA regression, grouped windowing, metrics MAE/RMSE/Pearson/CCC with subject-level CIs; also pooled-split run to quantify leakage. Server dir `~/music_therapy`.
- [ ] E2 Montage test on DEAP itself (drop/remap channels, measure degradation).
- [ ] E3 Audio-side VA evaluator (MERT features + DEAM/PMEmo regressor), cross-corpus check.
- [ ] E4 NMED-family transfer audit: distribution of predicted VA, uncertainty (ensemble / MC dropout).
- [ ] E5 Generator: Stable Audio Open adapter with VA (and later EEG) conditioning and modality dropout; leave-song-out; shuffled controls.

## Open decisions (team)
- Primary design: original three-phase pipeline vs fallback (DEAM/PMEmo generator, DEAP EEG-to-VA bridge).
- Whether NMED-T stays in the generation phase.

## Doable without DEAP (DEAP in maintenance as of 2026-11-03; email on hold)
- [x] Postolache 2025 read (WORKLOG pass 2). [ ] Shen 2026 (TMM) full text still open.
- [x] CBraMod, LaBraM, MERT, PMEmo checked; DEAM partly. [ ] LaBraM weight download, DEAM official page/license, fadtk, JASCO still open.
- [x] DREAMER (Zenodo request), AMIGOS (request), MAHNOB-HCI (mahnob-db.eu) access routes found; none is open. [ ] SEED-family, NMED-H/NMED-M not checked.
- [ ] E3 audio-side VA evaluator on the server: MERT features + regressor on DEAM/PMEmo, cross-corpus check.
- [ ] Download NMED-T cleaned EEG only (~7 GB for 10 files) to `~/music_therapy/data`, check shapes and metadata alignment (disk is ~99% full: check free space first).
- [ ] Generator smoke test: Stable Audio Open 1.0 loads, audio-to-audio from a DEAM clip, LoRA/adapter dry run with a VA token.
