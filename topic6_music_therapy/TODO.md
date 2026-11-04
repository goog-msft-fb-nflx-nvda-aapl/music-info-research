# Topic 6 TODO

## Verification (before any experiment)
- [x] NMED-T and NMED-E Stanford pages checked (WORKLOG 2026-11-03). NMED-H / NMED-M and the ISMIR paper still to check.
- [x] Both EEG-to-music papers exist (Postolache ICASSP 2025; Shen TMM 2026). Still to do: read both in full (data, metrics, code); find the DEAP subject-identity paper or drop the claim.
- [x] Stable Audio Open 1.0, AudioLDM2-music checked. Still to do: MERT-v1-95M, official CBraMod/LaBraM weights and licenses, DEAM/PMEmo, fadtk.
- [ ] DEAP: needs a permanent-staff signer (supervisor); service reported down. On hold; revise `drafts/email_deap_access.txt` if resumed.
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

## From access check (2026-11-03)
- [x] MET inspected (WORKLOG): same 20 songs for all, 66 ch, labels mostly song-level, no audio. [ ] Identify MET song source (contact author Weng Lam Cheang via OpenNeuro / paper) and audio rights. [ ] Download EEG (13 GB) to `~/music_therapy/data` after disk check.
- [x] ds002721 inspected: 19 ch, audio from Eerola & Vuoskoski set 1 (Jyvaskyla site), no rating file found. [ ] Check where the participant responses are; check stimulus audio terms.
- [ ] Open ds002722, ds002724, ds003774, DAAMEE (Zenodo 16738766) pages ourselves.
- [ ] Decide dataset for Phase 1/2 (MET vs DEAP vs DREAMER) after the inspections.

## Plan B without MET audio (2026-11-03, MET author mail not sent)
Audio is not needed for the EEG-to-valence/arousal step, so the bridge design does not depend on MET stimuli.
- [x] B1 DONE 2026-11-03 (WORKLOG): chance level under loso/loso_song/strict for ridge and CNN. [ ] B1 follow-ups: pretrained EEG encoder (CBraMod/LaBraM) features, per-subject calibration, 2 s windows / longer context. Original item: EEG -> valence/arousal on MET (66 ch, ratings.tsv), leave-song-out and leave-subject-out folds, per-song-label caveat (song explains 50%/67% of rating variance). Metrics: CCC, Pearson, RMSE; report chance level (shuffled labels).
- [ ] B2 Generator: Stable Audio Open 1.0 adapter conditioned on valence/arousal, trained on DEAM/PMEmo audio with their labels (open audio).
- [ ] B3 Join: MET-EEG-predicted valence/arousal -> generator; independent evaluator (MERT features + regressor trained on a different corpus).
- [ ] B4 If B1 is near chance, report that as a finding and test ds002721 / SEED (needs supervisor signature) as the EEG source.
- [ ] Optional: ask Prof. Yang to sign the DEAP/SEED agreements (permanent-staff signer required).

## After B1 (2026-11-03)
- B1 gave chance-level subject-independent prediction on MET, so B3 (EEG-predicted valence/arousal as generator input) has no usable signal yet. Options, in order: (1) B1 follow-ups above; (2) B4: DREAMER/SEED/DEAP via supervisor signature; (3) report B1 as a negative result and use valence/arousal ratings (not EEG) as conditioning, with EEG used only as an ablation.
- [ ] B2/E3 do not depend on B1: DEAM official page unreachable (cvml.unige.ch); find the download route (Zenodo/Mendeley/other) and verify license.
- [ ] Add per-fold progress logging to `met_eval.py` cnn mode (no progress output during the 32 min run).

## State after 2026-11-04 experiments (details in WORKLOG.md)
Done: B1 (MET EEG to valence/arousal: ridge, handcrafted, frozen and fine-tuned CBraMod/LaBraM, CNN; all at chance for unseen subjects/songs), B5/B6 (BCMI ds002722, ds002724: chance across participants; weak within-participant association that shrinks across sessions), E3 (MERT-ridge audio evaluator on PMEmo), Phase 3 prototype (VA-conditioned Stable Audio Open LoRA: arousal r about 0.8, valence r 0.2 to 0.33 by the evaluator; controls at 0), memorisation check, 9 demo clips.
Open, in order of expected value:
- [ ] Obtain a labelled EEG-emotion dataset with unseen-subject signal known from the literature (DEAP, DREAMER, SEED family; all need a signature from a permanent staff member = supervisor) and run the same protocols; MET/BCMI results suggest the bottleneck is the EEG data/labels, not only the model.
- [ ] Resolve the BCMI (ds002722) music-code to clip mapping from Daly et al., Scientific Data 2020 (doi 10.1038/s41597-020-0507-6); only then use its audio or the clip-level analyses.
- [ ] Generator: improve valence control (more training audio with valence labels: DEAM host unreachable on 2026-11-04, try again or find authorised mirror; EMOPIA / other open sets; longer training; text prompts), a second independent evaluator (e.g. CLAP-based or a model trained on a different corpus), FAD with fadtk on enough clips, statistics over more seeds.
- [ ] Connect EEG: replace labels by EEG-predicted valence/arousal (currently uninformative; report as negative end-to-end result) or use per-participant calibrated predictors with a fresh session.
- [x] LaBraM and CBraMod fine-tuning on BCMI done (chance).
- [ ] Contact MET authors only if the team decides to; the mail draft stays in drafts/ (not sent).
- [ ] Repository hygiene before going public: confirm licenses (PMEmo data use, demo audio, BCMI CC BY vs CC0).


## Status for teammates (2026-11-04, end of day)
Where we are: EEG -> valence/arousal does not work subject-independently on MET (66 ch) or BCMI ds002722 with ridge, EEGNet, frozen/fine-tuned CBraMod and LaBraM, covariance/alignment features (WORKLOG entries B1, B5, "Failure investigation"). The audio side works (VA-conditioned Stable Audio Open LoRA, arousal r about 0.8, valence 0.2-0.33 on a MERT evaluator). Open facts: MET ds008701 has 66 channels and no audio (the "8 channels" figure likely comes from the IEEE DataPort copy); BCMI has audio but its music-code-to-clip mapping is unresolved; DREAMER needs a Zenodo access request (no signature stated), DEAP/AMIGOS/MAHNOB/SEED need a permanent-staff signature (see drafts/email_deap_access.txt).
- [ ] Positive control is the key gap: the song cannot be identified from MET EEG across subjects even with lag tolerance. First check whether MET EEG is time-locked to the music (needs song audio or identity; the MET authors or the IEEE DataPort page may have it).
- [ ] Ask DREAMER access on Zenodo (https://zenodo.org/records/546113; name, email, institution, position, reason); run the same protocols (loso, strict, shuffled control) there.
- [ ] Per-subject few-shot calibration (train on 19 subjects + a few trials of the test subject) on MET and BCMI.
- [ ] Decide as a team: either (a) keep the pipeline with EEG as an input that is reported as a negative result, plus the working audio side, or (b) reframe the contribution on the audio side (valence control, second evaluator, FAD).
- [ ] Resolve BCMI clip mapping from Daly et al., Sci Data 2020 (doi 10.1038/s41597-020-0507-6).
- [ ] Survey round 4: send `survey4/survey_prompt.txt` to ChatGPT/Perplexity/Gemini, save as `survey4/response_*`, verify sources, then pick experiments (new-bottleneck framing in WORKLOG "Proposed new bottleneck").
