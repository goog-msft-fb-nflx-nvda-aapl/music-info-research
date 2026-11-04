# Topic 6 work log

## 2026-11-03. Survey round 1 (deep research, two responses)

Files: `survey1/survey_prompt.txt` (prompt), `survey1/response_chatgpt.md`, `survey1/response_gemini.md`. No experiment run. Nothing below is verified by us yet; "claimed" = stated by a response, source not opened.

### Agreement (both responses)
- DEAP has one 1-9 self-rating per 60 s trial; window-level regression repeats the same target. Segment/trial-level CV leaks subject identity; LOSO is the primary protocol.
- NMED-T: 20 listeners, 10 full songs, 128 ch, cleaned data ~125 Hz. No valence/arousal labels. Stimulus audio is not distributed (metadata only).
- Phase 1 (DEAP LOSO regression): go. Phase 2 (DEAP -> NMED-T pseudo-labels): weakest link, label validity untestable without NMED-T VA ratings. Phase 3 on 10 songs: song-identity shortcut and memorisation risk.
- Required controls: shuffled-EEG, leave-song-out, independent audio-side emotion evaluator (DEAM/PMEmo-trained), FAD with MERT/CLAP embeddings (fadtk), subject-level statistics.
- Backbone: Stable Audio Open 1.0 (adapters/LoRA). EEG encoder candidates: CBraMod, LaBraM (BIOT, EEGPT also named by ChatGPT).
- Fallback ranking (both): audio-side VA corpora (DEAM/PMEmo) for generator training, EEG-derived VA as the bridge.

### Disagreements to resolve (check primary sources)
| Item | ChatGPT | Gemini | Action |
|---|---|---|---|
| Prior EEG/NMED-T generation work | Claims ICASSP 2025 "Naturalistic Music Decoding from EEG Data via Latent Diffusion Models" (trained on NMED-T) and IEEE TMM June 2026 "Echoes of the Brain" | Claims no prior work | Verify both titles; decides novelty framing |
| NMED-E (pleasantness/arousal ratings) | Claims 23 listeners x 4 stimuli with VA-like ratings, usable as transfer-validation bridge | Lists NMED family as lacking robust emotion labels | Check Stanford repository |
| Raw NMED-T size | ~39 GB marked UNVERIFIED | ~39 GB raw, ~700 MB cleaned (claimed) | Check before download |
| AudioLDM2-music license / size | CC BY-NC-SA 4.0, ~1.1B | CC BY 4.0, 300M | Check model card |
| Stable Audio Open size | 1B | 1.2B (1.05B DiT) | Check model card |
| DEAP regression numbers | LOSO RMSE 1.902 val / 1.769 aro (one study); a "<0.16 RMSE" study flagged as protocol-dependent | RMSE 1.90-2.11 on 1-9 scale; <0.16 attributed to min-max normalisation (interpretation) | Check the cited studies |
| DEAP subject-identity confound | Claims a 2026 re-analysis: AUC 0.689/0.711 pooled vs 0.493/0.447 participant-independent | not mentioned | Verify (PubMed) |
| Audio source for NMED-T | Redistribution not assumed; public demos need legally redistributable audio | Claims Japanese Copyright Act Art. 30-4 permits scraping the audio | Gemini's legal claim not accepted: team and use are not shown to fall under Japanese law, and it covers analysis, not public release of generated audio. Needs a real rights decision |
| Channel mapping | Ablate nearest / 10-20 subset / spline / native adapter | Abandon interpolation, use channel-agnostic model | Keep as ablation |

### Citations
ChatGPT's links are opaque tool tokens, not URLs. Gemini's list has URLs but some are secondary (ResearchGate, arXiv listing pages). Every paper used in the report must be opened and checked before citing. Gemini's table entries marked "Yes" for verified were not independently checked (e.g. LaBraM "96-97% accuracy on DEAP", TSception LOSO F1, EEG Conformer on DEAP).

### Data access still unknown
DEAP requires a signed EULA (original host status unclear). NMED-T/NMED-E/NMED-H: Stanford PURL jn859kj8079 (T). The exact NMED-T audio files have to be sourced by us; not available now.

## 2026-11-03. Verification pass 1 (web, primary pages opened)

| Item | Result | Source |
|---|---|---|
| NMED-T | 20 adults, 10 full commercial songs, dense-array EEG (channels/rate not on page; ISMIR paper says 128 ch), 55 files ~39 GB total: cleaned per-song aggregates 656-721 MB each, raw 753-861 MB each (two per participant). Behavioral: tapping (35 s excerpts), familiarity and enjoyment per song. No valence/arousal. Stimulus audio not included, metadata links only. License CC BY 3.0. | purl.stanford.edu/jn859kj8079 |
| NMED-E | Is the Elgar dataset, not an emotion dataset: 24 adults, one stimulus (Elgar Cello Concerto, 1st mvt) plus a phase-scrambled control, 128 ch. Post-stimulus ratings: pleasantness, arousal, interest, predictability, familiarity (one ratings file, <1 KB); continuous engagement reports. CC BY 4.0. ChatGPT's "23 x 4 stimuli" was wrong. It has arousal ratings but only 1 piece, so it cannot validate a DEAP regressor across music. Audio availability not stated on page. | purl.stanford.edu/pp371jh5722 |
| Other Stanford sets | NMED-H (Hindi, 48 adults), NMED-M (Minimalism); not inspected. | purl pages |
| Postolache et al., "Naturalistic Music Decoding from EEG Data via Latent Diffusion Models" | Exists. ControlNet on AudioLDM2 conditioned on EEG, trained on NMED-T, arXiv 2405.09062 (v6 Jan 2025), listed as ICASSP 2025 by a university repository (proceedings entry not seen). No code link seen on the arXiv page. Gemini's "no prior work" is wrong. | arxiv.org/abs/2405.09062 |
| "Echoes of the Brain" (Shen et al.) | Exists as IEEE Trans. Multimedia 2026, DOI 10.1109/TMM.2026.3704863, in press; EEG-music latent alignment then EEG-conditioned DiT ControlNet. Only an institutional listing seen; abstract-level only; dataset not confirmed. | pure.bit.edu.cn |
| Stable Audio Open 1.0 | ~1B params (F32), Stability AI Community License, 44.1 kHz stereo, up to 47 s. Model card has no fine-tuning instructions. Gemini's 1.2B not confirmed. | HF model card |
| AudioLDM2-music | CC BY-NC-SA 4.0 (Gemini's CC BY 4.0 wrong), 1.1B total with 350M UNet; samples shown at 16 kHz. | HF model card |
| CBraMod (braindecode) | Code BSD-3-Clause; HF page says no pretrained weights distributed there. Weights from the official repo (wjq-learning/CBraMod) not yet checked. | HF braindecode/CBraMod |
| DEAP | Access by signed EULA emailed to QMUL; preprocessed Python version is 32 EEG ch at 128 Hz, raw BDF 512 Hz (via search snippets and TorchEEG docs; the official page could not be fetched, certificate error). | search snippets |
| 2026 DEAP subject-identity paper (AUC 0.689/0.711 vs 0.493/0.447) | Not found. Treat as unverified; do not cite. | search |

### Interpretation
- Novelty of EEG-conditioned diffusion music generation on NMED-T is taken: two papers precede it. Remaining distinct parts: emotion (VA) as intermediate variable, DEAP-to-NMED transfer, uncertainty filtering, leave-song-out/shuffle controls.
- No public dataset found so far pairs music-listening EEG with per-song or continuous VA across many pieces except DEAP (music videos). Other candidates remain unchecked (DREAMER/AMIGOS/MAHNOB are film clips).

### Still unverified
Echoes of the Brain full text and data; CBraMod/LaBraM official weights; ChatGPT's DEAP LOSO RMSE 1.902/1.769 study; Gemini's LGGNet/LaBraM/Conformer DEAP numbers; DEAM/PMEmo licenses and sizes; fadtk; JASCO; official DEAP page.

## 2026-11-03. Pipeline bottleneck assessment (interpretation, from verified facts above)
Original three-phase pipeline blockers, in order of severity:
1. Phase 2 label validity: NMED-T has no valence/arousal ratings (only familiarity, enjoyment), so DEAP-model pseudo-labels on NMED-T cannot be checked against any ground truth. DEAP differs from NMED-T in stimuli (music videos vs 10 audio-only songs), cap (32 vs 128 ch), lab and task.
2. Phase 3 data: 10 songs x 20 listeners; EEG is time-locked to the song, so EEG can encode song identity (a shortcut) and the generator can memorise 10 tracks. NMED-T audio is not distributed; the exact recordings must be sourced separately, and public demos need redistributable audio.
3. Novelty: EEG-conditioned diffusion on NMED-T already published (Postolache 2025; Shen 2026), so the claim has to rest on the valence/arousal bridge and controls.
4. Phase 1 risk (not a blocker): DEAP labels are one per 60 s trial; needs strict LOSO.
Fallback that avoids 1-2: generator trained on audio with valence/arousal labels (DEAM/PMEmo), DEAP EEG-to-valence/arousal model as the bridge; NMED-T only for a transfer audit.

## 2026-11-03. Drafts
`drafts/email_deap_access.txt` (DEAP access request, placeholders for names/affiliation/supervisor email) and `drafts/message_to_teammates.txt`. Not sent. Team asked first whether anyone already has DEAP.

## 2026-11-03. Verification pass 2 (papers, weights, datasets)

| Item | Result | Source |
|---|---|---|
| Postolache et al. 2025 (arXiv 2405.09062v6) | NMED-T, 124 of 128 ch (face channels dropped), EEG 1 kHz, 3.5 s chunks, minimal preprocessing. ControlNet on `audioldm2-music`, text fixed to "Pop music". Split: 9 songs split by time within each song (train 0-80%, val 80-90%, test 90-100%); 1 song ("Trigger 21") held out as out-of-distribution. "All"-subject models evaluated on subject 2 only. Test FAD (CLAP) ControlNet-2 0.38 vs pretrained no-EEG 0.80; CLAP score 0.60 vs 0.38; EnCodec metrics near no-EEG baseline (authors call EnCodec unsuitable); OOD-track results mostly not significant. No code repository stated; audio examples at emilianpostolache.com/brainwave. Stimulus audio source not described. | arxiv.org/html/2405.09062v6 |
| Shen et al. 2026 TMM | Still abstract-level only (not re-checked). | pure.bit.edu.cn |
| CBraMod | Code MIT (GitHub wjq-learning/CBraMod); weights on HF `weighting666/CBraMod`, license apache-2.0 on the HF card; input (batch, channels, segments, 200-sample patches); parameter count and pretraining data not stated on the pages read. | GitHub, HF |
| LaBraM | Code MIT (GitHub 935963004/LaBraM); input resampled to 200 Hz, 0.1-75 Hz bandpass, channel order list required; checkpoint `labram-base.pth` referenced, download location not given on the page; pretraining ~2,500 h from ~20 datasets (not named on page). | GitHub |
| MERT-v1-95M | 95M params, 24 kHz, 75 Hz features, 12 layers x 768, license **cc-by-nc-4.0** (Gemini's Apache 2.0 wrong; non-commercial). | HF |
| DEAM | 1,802 songs/excerpts, per-second valence/arousal, CC-licensed audio (FMA, Jamendo, MedleyDB), PLOS ONE 2017, doi 10.1371/journal.pone.0173392; one manual copy says BY-NC; official page (cvml.unige.ch) unreachable, so license and download not confirmed. | PMC5345802, search |
| PMEmo | 794 songs, static labels plus dynamic labels per 0.5 s (first 15 s removed), chorus MP3 excerpts only (no full songs), ~1.3 GB on Google Drive (2018, 2019 versions), repo MIT (dataset license not stated). | github.com/HuiZhangDB/PMEmo |
| DREAMER | Zenodo record 546113 is restricted: access by request form. | zenodo.org/records/546113 |
| AMIGOS | Username/password via request on QMUL page. | eecs.qmul.ac.uk/mmv/datasets/amigos/download.html |
| MAHNOB-HCI | Via mahnob-db.eu; terms not confirmed. | search |

### Interpretation
- Postolache's test is a within-song time split, so its good scores do not show generalisation to new songs; our leave-song-out protocol would be stricter. Their approach is a strong baseline to reproduce or compare (ControlNet on audioldm2-music, but that checkpoint is non-commercial CC BY-NC-SA).
- No dataset found yet that gives open (no-EULA) EEG plus music emotion ratings; DEAP, DREAMER, AMIGOS, MAHNOB all need requests. If DEAP stays down, DREAMER (Zenodo request) or AMIGOS is the substitute; both use film clips, not music.
- Still open: Shen 2026 full text, LaBraM weight download, DEAM official page, JASCO, fadtk.

## 2026-11-03. Access check
DEAP (maintenance), DREAMER Zenodo URL and AMIGOS QMUL page unavailable when tried by the user. Prompt for a second deep-research round: `survey2/access_check_prompt.txt`. Not sent yet.

## 2026-11-03. Access check results (survey2: response_perplexity.txt, response_chatgpt_v2.md)
Both responses are third-party; items marked (ours) were opened by us.

Agreement: DEAP/AMIGOS/MAHNOB-HCI need a signed EULA by a person with a permanent academic/research position (up to five same-institution researchers can be named, institutional email and webpage required); students cannot be the signer. DEAP page down/timeout, AMIGOS 502, MAHNOB HCI-Tagging path 404. SEED family: application form works, supervisor name/email requested for students, academic use only, no redistribution. DREAMER: Zenodo restricted record, request form (name, email, institution, position, reason); ChatGPT could not open the page today, Perplexity could. No owner-authorized mirrors found for DEAP/AMIGOS/MAHNOB/SEED/DREAMER; Hugging Face/Kaggle copies not authorized.

Open datasets (no agreement), music listening + affect ratings:
| Dataset | Content | License | Size | Check |
|---|---|---|---|---|
| MET, OpenNeuro ds008701 | 20 participants x 20 songs (one per session), 120 s music, SAM valence 1-9, arousal 1-9, likeability, familiarity, HVHA/HVLA/LVHA/LVLA/N labels; 128 ch at 1 kHz recorded, 66 ch released after preprocessing | not shown on GitHub README; Perplexity/ChatGPT say CC0 | 13.06 GB (ChatGPT) | (ours) GitHub README: content and ratings confirmed; audio inclusion not stated; license not confirmed |
| ds002721 | 31 adults, 40 film-score clips ~12 s (Eerola & Vuoskoski 2010 stimulus set), 1 kHz, reported emotional responses, resting runs | CC BY 4.0 (ours, GitHub README Terms of Use; ChatGPT's CC0 wrong) | 3.35 GB (ChatGPT) | (ours); rating scale not described; audio not included; page says 20 s vs 12 s epoch, check |
| ds002724 | 16 participants, EEG/GSR/ECG, 40 s clips, valence/arousal | CC BY 4.0 (Perplexity) | n/a | not opened by us |
| ds002722 | EEG/GSR/ECG, valence/arousal-targeted clips | CC0 (ChatGPT) | n/a | not opened by us |
| MUSIN-G, ds003774 | 20 participants, 12 songs, familiarity/enjoyment 1-5 | CC0 (Perplexity) | n/a | not opened by us |
| DAAMEE, Zenodo 16738766 | scalp EEG + cEEGrid, music, behavioural emotion responses | LICENSE file present, text not read | 32.8 GB (ChatGPT) | not opened by us |
| Mendeley 9fm94df3x2, xjtxnn7bx7 | spectral EEG features only (no raw EEG), valence/arousal | CC BY 4.0 | small | not opened by us |

### Interpretation
- DEAP cannot be requested by students alone; an email from us as drafted would not meet its terms. A permanent staff member (supervisor) has to submit/sign. With the service reported down, DEAP is no longer the fastest start.
- MET (ds008701) has EEG, music listening and per-song valence/arousal ratings in one open dataset; it could replace DEAP for emotion modelling and also serve as the paired EEG-music set. Whether audio is obtainable (needed for the generator), the license, and whether 20 songs per participant are the same songs across participants are unchecked.
- ds002721 stimuli are a published research stimulus set (audio not in dataset).

## 2026-11-03. Dataset inspection: MET (ds008701) and ds002721 (files read from GitHub mirrors, no EEG downloaded)
Script: `scripts/met_ratings_summary.py`; output: `results/met_ratings_summary.txt`.

MET (ds008701), ours:
- License field in dataset_description.json: CC0. Author: Weng Lam Cheang. 20 participants (21-29 y), 400 rating rows (2 flagged missing).
- 20 distinct songs (ids within t01-t30), each rated by 19-20 participants; every participant heard the same 20 songs in random order. Song categories 4 each of HVHA, HVLA, LVHA, LVLA, N.
- Released EEG: 66 EEG channels + 1 EOG, 1000 Hz, `.set` files, 130 s per song (120 s music), `bad` segments marked in events.tsv. Electrode positions provided per recording.
- Music audio not in the dataset; song names not given in the README, only ids t01-t30. Audio source must be identified separately (unresolved).
- Ratings: valence 1-9, arousal 1-9 (histograms in results file). Share of rating variance explained by song: valence 0.50, arousal 0.67; by participant mean: 0.06 and 0.05.
- Interpretation: labels are mostly song-level, so a subject-independent EEG-to-valence/arousal model on MET is confounded with song identity unless evaluated leave-song-out, and only 20 songs exist.

ds002721 (BCMI-MIdAS), ours:
- License conflict inside the source: dataset_description.json says CC0, README Terms of Use says CC BY 4.0 (University of Reading rights-holder). Use CC BY 4.0 attribution to be safe (interpretation).
- 31 participants, 19 EEG channels, 1 kHz, EDF, 6 runs (runs 1 and 6 resting). Events: music onset code 788, clip code 301-360 = mp3 `<code-300>.mp3` from the Eerola & Vuoskoski (2010) soundtrack set 1, downloadable from the University of Jyvaskyla page (reachable stimulus audio; terms not checked). Clip length: README says 12 s and 20 s in different places.
- No per-participant rating file found in the BIDS tree (README says it includes reported emotional responses; location unknown). Not usable as a valence/arousal target source until located.

Open: license/redistribution terms of the Jyvaskyla stimulus audio; source of MET stimuli; ds002722, ds002724, ds003774, DAAMEE not inspected.

## 2026-11-03. Draft to MET author
`drafts/email_met_author.txt`: asks for song ids -> titles/excerpt times and audio availability. Not sent; author's contact email not verified (placeholder).

Prompt to locate the MET author contact and song list: `survey3/find_met_contact_prompt.txt` (for ChatGPT / Perplexity / Gemini).

## 2026-11-03. MET author contact (survey3: r3_chatgpt.rtf, r3_perplexity.rtf, r3_gemini)
- No email for Weng-Lam Cheang found anywhere. ORCID 0009-0007-9415-4308 (ChatGPT, Gemini; not opened by us).
- datacite.yml (ChatGPT): six creators: Cheang (Researcher), Weng-Lam Chan, Yiting Zhu, Xinwen Jiang (DataCollectors), Cheng-Teng Ip (Supervisor), Minglei Zhang (Sponsor). IEEE DataPort lists the six authors at University of Macau (Perplexity).
- Ours: PubMed 40419149 (2025, Chan et al.) prints "Electronic address: chengtengip@um.edu.mo" with Centre for Cognitive and Brain Sciences, University of Macau. ChatGPT and Gemini give the same address; ChatGPT also lists ICI.CogBrainSci@um.edu.mo (not verified by us).
- Ours: IEEE DataPort author page lists only "MET: Music-Induced EEG Dataset" (a separate version with 8 channels, ~400 CSV files, per ChatGPT-independent page text); private messaging requires login. Different file format from OpenNeuro (66 ch .set).
- No MET data paper found; song titles/audio source not found by any response. GitHub issue tracker (OpenNeuroDatasets/ds008701/issues) named as alternative channel (not checked by us).
- Draft `drafts/email_met_author.txt` redirected to Prof. Ip with a forward request. Not sent.

## 2026-11-03. B1: EEG -> valence/arousal on MET (ds008701), grouped cross-validation
Scripts (all in `scripts/`): `download_met.sh` (public S3, 256 parallel connections), `prep_met.py`, `met_eval.py`, `met_probes.py`. Server: `~/music_therapy` (env `~/music_therapy/env`, torch 2.6.0+cu124, mne 1.12.1). Results: `results/b1/` (summary_ridge.json, summary_cnn.json, probes.json, logs).

Commands (server, in `~/music_therapy`):
```
bash scripts/download_met.sh data/ds008701
env/bin/python -I scripts/prep_met.py data/ds008701 data/met_win4s.npz 4 128
env/bin/python -I scripts/met_eval.py data/met_win4s.npz results/b1 ridge 4
env/bin/python -I scripts/met_eval.py data/met_win4s.npz results/b1 cnn 4      # 80 folds over 4 H200, 1922 s
env/bin/python -I scripts/met_probes.py data/met_win4s.npz results/b1/probes.json
```

Data used: 399 EEG files downloaded; 1 unreadable (sub-12 ses-t05, file size equals the S3 object size, so the source file is corrupt), 1 trial flagged missing in ratings.tsv; 397 recordings, 396 trials with usable EEG and ratings (20 subjects, 20 songs), 10,799 non-overlapping 4 s windows (6-124 s of each recording, windows overlapping annotated bad segments dropped), 66 EEG channels (EOG removed), 1-45 Hz band-pass, resampled to 128 Hz, per-recording per-channel z-score.

Models: ridge regression on log band power (delta, theta, alpha, beta, gamma x 66 channels; alpha 0.1/1/10, table shows alpha 1); EEGNet-style CNN (30 epochs, fixed hyperparameters, no tuning on test folds). Window predictions averaged per trial. Metrics at trial level (n=396): Pearson r / CCC / RMSE (1-9 scale). Protocols: loso = leave one subject out (20 folds); loso_song = leave one song out (20); strict = unseen subjects and unseen songs (5 x 4 = 20 folds); random = random window 5-fold (leaky reference, not a valid protocol). Shuffled-label control: ridge trained on trial labels permuted across training trials.

| protocol | target | ridge r / CCC / RMSE | ridge shuffled-label r / CCC | CNN r / CCC / RMSE |
|---|---|---|---|---|
| loso | valence | 0.00 / 0.00 / 2.23 | -0.08 / -0.05 | -0.08 / -0.07 / 2.81 |
| loso | arousal | -0.01 / -0.00 / 2.68 | -0.06 / -0.03 | -0.01 / -0.01 / 3.30 |
| loso_song | valence | -0.07 / -0.04 / 2.27 | -0.13 / -0.07 | -0.10 / -0.10 / 2.81 |
| loso_song | arousal | -0.09 / -0.05 / 2.73 | -0.21 / -0.12 | 0.05 / 0.04 / 3.10 |
| strict | valence | -0.10 / -0.07 / 2.37 | -0.16 / -0.11 | -0.08 / -0.08 / 2.80 |
| strict | arousal | -0.02 / -0.02 / 2.75 | -0.08 / -0.05 | -0.02 / -0.02 / 3.34 |
| random | valence | 0.42 / 0.21 / 1.97 | 0.00 / 0.00 | 0.92 / 0.89 / 0.90 |
| random | arousal | 0.38 / 0.20 / 2.38 | -0.03 / -0.01 | 0.94 / 0.91 / 0.96 |

Reference predictors without EEG (trial level, r / CCC / RMSE): song mean from the other subjects (knows the song id): valence 0.67 / 0.63 / 1.59, arousal 0.79 / 0.78 / 1.55. Subject mean from other songs: valence 0.04 / 0.02, arousal 0.01 / 0.01. Global mean of other subjects: r is negative by construction (leave-one-out mean), CCC 0.

Probes (ridge/logistic regression on the same band-power features, leave-one-subject-out): song identity from EEG: window accuracy 0.042, trial accuracy 0.043 (chance 0.05). Residual rating (rating minus training-subject song mean): valence r 0.01, CCC 0.003; arousal r 0.07, CCC 0.017.

Findings (measured): with band-power ridge and an EEGNet-style CNN, valence and arousal were not predicted above chance when the test subject (loso), the test song (loso_song) or both (strict) were unseen; CCC between -0.12 and 0.05, shuffled-label control similar. Random window splits give r 0.38-0.42 (ridge) and r 0.92-0.94, CCC 0.89-0.91 (CNN), i.e. the leakage gap is large; those numbers come from windows of the same trial appearing in training and test. Band-power features did not recover song identity or the subject-specific residual of the ratings across subjects.
Interpretation (not tested): the random-split scores reflect trial/subject-specific signal that does not carry over to unseen subjects; 20 subjects and 20 songs with song-driven labels (see inspection above) give little room; other features (e.g. pretrained EEG encoders, connectivity, per-subject calibration) were not tried. These results do not show that MET EEG contains no emotion information.
Caveats: one hyperparameter setting per model; the 4 s window and 1-45 Hz preprocessing were fixed a priori; bootstrap CIs resample subjects (files in summary json).

## 2026-11-04. B1 follow-ups on MET (all measured, ridge on window features, trial-level CCC; best of loso / loso_song / strict over alpha 1, 10, 100)
Scripts: `prep_met.py ... 200 uv100`, `extract_encoder.py`, `make_band_feats.py`, `met_eval_feat.py`; results `results/b1f/*.json` (c0 = raw features, c1 = per-subject mean-centred features, no labels used).
- Handcrafted features (log band power, relative band power, Hjorth, left-right alpha asymmetry, all combined): best CCC 0.00-0.04 (valence), 95% subject-bootstrap CIs include 0; random-window split CCC up to 0.32 (leaky).
- Frozen CBraMod (global mean/std of encoder output, 400-d) and frozen LaBraM (global mean 200-d, cls token 200-d): best held-out CCC between -0.01 and 0.003; random-split CCC <= 0.045.
- Per-channel pooled CBraMod/LaBraM features (13,200-d, ridge) were still running when this entry was written (see below if filled).
- Environment note: installing `timm` upgraded torch to cu130 (driver 12.4 cannot run it); reinstalled torch 2.6.0+cu124 and cudnn 9.1.0.70. The B1 CNN run was before this change.
- Data note: one MET recording unreadable (sub-12 t05, corrupt at source).

## 2026-11-04. New open dataset found: BCMI-MIdAS calibration set (OpenNeuro ds002722), inspected and downloaded
Source files (ours): README, events.json, EDF channel list, stimuli folder in the GitHub mirror.
- 19 participants, 5 runs x 18 trials, 40 s synthetic-music clips (first 20 s one targeted affect state, the rest another; events mark 21 s), 32 EEG channels at 1 kHz + GSR + ECG; stimuli folder contains the 108 synthetic clips as .wav (89 distinct clips were presented in the data we parsed); target affect on a 3x3 valence x arousal grid; reported valence and arousal stored as continuous channels VA1, VA2 (1-9 scale, start 4.5; 0 outside trials) inside the EDF. License: README says CC BY 4.0, dataset_description.json says CC0 (attribute to be safe). Same group has ds002724 (training sessions, 16 participants x 3 sessions, 8.6 GB, downloaded), ds002721.
- Parsed with `scripts/prep_bcmi.py` (trial = first 20 s after onset; 4 s windows; targets = trial means of VA1/VA2 over samples > 0, plus window means): 94 recordings, 1,645 trials, 8,315 windows, 19 participants. Reported valence mean 4.47 (sd 1.46), arousal 4.16 (sd 1.30).
- Reference (no EEG), trial level: song(clip)-mean of other participants predicts reported valence r 0.69 CCC 0.65, arousal r 0.64 CCC 0.59.
- B5 ridge band-power (`results/b5/summary_ridge.json`): leave-subject-out valence CCC -0.002 (arousal -0.003); leave-clip-out valence CCC 0.034 (CI -0.0003..0.066), arousal 0.019 (CI 0.002..0.038); unseen subject and clip -0.012 / -0.014; random-window split 0.051 / 0.037; shuffled-label controls about -0.01.
- Interpretation: the audio (synthetic, with known generator parameters) predicts the reported ratings across listeners, while band-power EEG did not, again. Open audio + EEG + reported valence/arousal in one CC-licensed set gives a data basis for the generator step (audio conditioning) that MET lacks.

## 2026-11-04. B1/B5 additions, E3 (audio-side evaluator), PMEmo
Scripts: `bcmi_extra.py`, `extract_mert.py`, `e3_audio_va.py`; results `results/b5/extra_ridge.json`, `results/e3/e3_audio_va.json`, `results/b1f/*.json`.

Per-channel pooled foundation-model features (13,200-d, ridge), MET: CBraMod raw: best held-out CCC 0.009 (valence); CBraMod subject-centred: arousal CCC 0.032 (95% CI -0.01..0.07, loso) and 0.030 (strict); LaBraM subject-centred valence CCC 0.028 (CI -0.02..0.08); LaBraM raw run incomplete at time of writing. All intervals include 0.

BCMI (ds002722) extra protocols, ridge on band power (alpha 1): within-subject trial-blocked 5-fold (train on the same participant only): valence r 0.118 (sd across 19 participants 0.164), CCC 0.069 (0.094); arousal r 0.077 (0.163), CCC 0.050 (0.100). Leave-one-subject-out with window-level continuous targets: valence CCC -0.003, arousal 0.003; with trial-averaged features: -0.002 / -0.002. Interpretation: a small positive within-participant association (mean r about 0.1; participant-level t about 3 for valence if participants are independent) while nothing transfers across participants. Within-participant folds share recording session and slow drifts, so this is not evidence of emotion decoding by itself.

PMEmo (downloaded from the owners' Google Drive link in their GitHub README, 794 chorus MP3 clips, annotations for 767 songs; license for the data not stated on the page, repository code MIT; used only locally for analysis, not redistributed): MERT-v1-95M (cc-by-nc-4.0) layer-6 time-mean features + ridge, 10-fold CV over songs: valence r 0.776, CCC 0.753, RMSE 0.102; arousal r 0.857, CCC 0.845, RMSE 0.095 (labels in 0-1 range). These are measured on PMEmo itself (same corpus, random song folds).
DEAM official host (cvml.unige.ch) unreachable from server and Mac (connection timeout, 2026-11-04).

BCMI stimulus mapping problem (unresolved): events.json says to subtract 100 from the music code and look the file up in the listed stimulus list. Applying that rule (list parsed from events.json), the clip name's affect label matches the trial's target code in only 383 of 1,663 trials (chance about 11%). The same music code co-occurs with many different target codes across trials. Therefore music codes are not confirmed to identify the same audio file across participants; the "song" grouping and song-mean reference used for BCMI above (and the E3-to-BCMI comparison: MERT-PMEmo predictor vs clip-mean reported valence r 0.53, arousal r -0.56 over 89 codes) are not valid evidence about audio. Reported valence/arousal showed no association with the target code (Spearman |rho| < 0.1 for trial mean, last value, second-half mean; 504 trials of sub-01..06). Needs the dataset paper (Daly et al., Scientific Data 2020, doi 10.1038/s41597-020-0507-6; page not retrievable here) before using ds002722 audio.

## 2026-11-04. Phase 3 prototype: valence/arousal-conditioned Stable Audio Open 1.0 (LoRA) on PMEmo, evaluated with an independent MERT evaluator
Scripts: `sao_va.py` (train/gen), `gen_eval.py`, `extract_mert.py`, `e3_audio_va.py`. Results: `results/gen/gen_eval_v1.json`, `results/gen/*/train_loss.json`.
Model facts (loaded by us): stable-audio-open-1.0 via diffusers 0.41: DiT 1.057 B parameters, VAE 156 M, T5 encoder 110 M, 44.1 kHz stereo; HF access with the account token works.
Method: encoder tokens = [T5("music"), seconds_start, seconds_end] + 2 VA tokens (MLP of (valence, arousal) in 0-1; replaced by a learned null embedding with probability 0.15 = modality dropout for classifier-free guidance); LoRA rank 16 on all DiT attention projections (8.26 M trainable) + VA MLP; EDM/v-prediction objective of the diffusers CosineDPM scheduler (sigma_data 1, u ~ U(0.02,0.999), sigma = tan(u pi/2)); batch 8, 3,000 steps (~11 min on one H200), 10 s crops (2,218 crops of 652 training songs; 15% of songs (115) held out). Train loss 0.749 -> about 0.706 in all runs.
Runs (3,000 steps, seed 0): A_true (true PMEmo labels), B_shuf (labels permuted across training songs: control), C_none (no VA, null tokens only: baseline). Generation: 3x3 grid of target (valence, arousal) in {0.2, 0.5, 0.8}, 6 clips per cell (54 clips), 50 DPM steps, CFG 3.0 on the VA branch.
Evaluator: MERT-v1-95M layer-6 time-mean features + ridge trained on the 652 training songs (PMEmo labels). On real held-out songs (n 115, 30 s chorus clips): valence r 0.80 / CCC 0.77, arousal r 0.87 / CCC 0.85.
Measured (target vs evaluator-predicted VA on generated clips, n 54): A_true valence r 0.29 (CCC 0.26, slope 0.18), arousal r 0.50 (CCC 0.46, slope 0.33); B_shuf valence r -0.24, arousal 0.05; C_none valence -0.05, arousal -0.02. Frechet distance (PCA-16 of MERT features, to held-out real clips): A 0.90, B 0.84, C 0.78 (n 54 vs 115, indicative only).
Interpretation: the VA tokens steer the evaluator-predicted arousal (and weaker valence) in the intended direction, controls do not; effect sizes are small relative to real audio (slopes 0.18/0.33 vs 0.62/0.73 for real music). Single seed, 54 clips, no confidence intervals yet; the evaluator and the generator's labels come from the same corpus (PMEmo) and the evaluator is a ridge model on MERT features, so alignment shows consistency with that evaluator, not human-perceived emotion (no listening test by design). Next: longer training (8,000 steps, 2 seeds of true labels, shuffled control), more clips per cell (12), CFG 5, bootstrap CIs.

## 2026-11-04. Phase 3 prototype, round 2: 8,000-step runs, 2 seeds, 12 clips per cell, CFG 3 and 5, bootstrap CIs
Commands (server, `~/music_therapy`): `env/bin/python -I scripts/sao_va.py train <run> <gpu> <true|shuffled|none> <steps> <seed>` (A8k_s1: true, 8000, seed 1; A8k_s2: true, 8000, seed 2; B8k_s1: shuffled, 8000, seed 1; 3k-step runs from round 1 re-generated with 12 clips per cell), `env/bin/python -I scripts/sao_va.py gen <run> <gpu> 12 50 <cfg>`, `env_mert/bin/python -I scripts/gen_eval.py results/gen/gen_eval_v2.json 3 <runs...>` (env_mert: torch 2.6.0+cu124, transformers 4.44.2, needed because MERT remote code returns no hidden states under transformers 5; main env: transformers 5.20.2, diffusers 0.41.0, huggingface_hub 1.33).
Generation: 3x3 target grid (valence, arousal in {0.2, 0.5, 0.8}), 12 clips per cell = 108 clips, 10 s, 50 DPM steps; evaluator and reference as in round 1. Pearson r between target and evaluator-predicted VA with bootstrap 95% CI over clips (1000 resamples; clips treated as independent), slope of predicted on target, Frechet distance (PCA-16 of MERT features, to 115 held-out real clips; indicative).

| run | n | valence r [95% CI] | valence slope | arousal r [95% CI] | arousal slope | FD |
|---|---|---|---|---|---|---|
| A8k_s1_n12 | 108 | 0.20 [0.01, 0.39] | 0.15 | 0.81 [0.73, 0.86] | 0.68 | 0.83 |
| B8k_s1_n12 | 108 | -0.05 [-0.24, 0.14] | -0.02 | 0.07 [-0.10, 0.23] | 0.04 | 0.87 |
| A8k_s2_n12 | 108 | 0.26 [0.09, 0.43] | 0.19 | 0.82 [0.77, 0.87] | 0.75 | 0.62 |
| A8k_s1_cfg5_n12 | 108 | 0.30 [0.13, 0.46] | 0.24 | 0.74 [0.65, 0.82] | 0.65 | 0.95 |
| B8k_s1_cfg5_n12 | 108 | -0.03 [-0.22, 0.15] | -0.01 | -0.05 [-0.24, 0.14] | -0.03 | 0.98 |
| A8k_s2_cfg5_n12 | 108 | 0.33 [0.17, 0.48] | 0.29 | 0.80 [0.72, 0.86] | 0.87 | 0.73 |
| A_true_n12 | 108 | 0.25 [0.08, 0.42] | 0.14 | 0.50 [0.33, 0.66] | 0.35 | 0.78 |
| B_shuf_n12 | 108 | -0.16 [-0.34, 0.02] | -0.07 | -0.04 [-0.22, 0.14] | -0.02 | 0.75 |
| C_none_n12 | 108 | -0.08 [-0.26, 0.15] | -0.03 | -0.07 [-0.27, 0.13] | -0.04 | 0.75 |
| real held-out songs (reference) | 115 | 0.80 | 0.62 | 0.87 | 0.73 | - |

Measured: with 8,000 training steps and true labels the evaluator-predicted arousal follows the target (r 0.81 and 0.82 in two seeds; slope 0.68 and 0.75, versus slope 0.73 for the evaluator on real held-out music), valence follows weakly (r 0.20 and 0.26; CFG 5: 0.30 and 0.33; slopes 0.15 to 0.29 versus 0.62 on real music). Shuffled-label control (B8k_s1) and the no-VA baseline (C_none) show no association (all r CIs include 0, except B_shuf 3k valence -0.16 [-0.34, 0.02]). 3,000-step runs: arousal r 0.50, valence r 0.25. CFG 5 raised valence correlation and slope slightly and did not improve arousal. Frechet distances of true-label runs (0.62 to 0.95) are in the same range as controls (0.75 to 0.98); no fidelity loss visible at this sample size.
Interpretation (not tested): steering works much better along arousal (energy) than valence for this evaluator and generator; this matches the known difficulty of valence in music emotion recognition and of the evaluator itself (real-music valence r 0.80). The alignment holds as far as the MERT-ridge evaluator is concerned; it is trained on the same corpus (PMEmo labels) as the generator's conditioning labels, so it is not independent of label noise or of that corpus. No human listening test (by design). The EEG part of the pipeline is not connected to these generations yet (EEG-derived VA were at chance, see B1/B5).

## 2026-11-04. B5 CNN on BCMI (ds002722), 80 folds on 4 GPUs (`results/b5/summary_cnn.json`)
EEGNet-style CNN, same protocol code as MET (trial level, n 1,645): leave-subject-out valence r 0.054, CCC 0.017 (CI -0.006..0.041); arousal r -0.014, CCC -0.006. Leave-clip-code-out (clip identity not verified, see above) valence r 0.169, CCC 0.067 (CI 0.014..0.107), arousal r 0.139, CCC 0.047 (0.006..0.095). Unseen subject and clip code: valence CCC -0.011, arousal -0.016. Random window split: CCC 0.080 / 0.064 (not leaky-high here, unlike MET: windows are 4 s of a 20 s trial and the CNN did not memorise). Interpretation: subject-independent decoding is near chance again; the leave-code-out positive CCC (up to 0.067) is small and its meaning depends on the unresolved code-to-clip mapping.
Fine-tuning CBraMod end-to-end on MET (loso, loso_song, strict; 6 epochs) started with `scripts/finetune_fm.py` (results to `results/b1ft/`).

## 2026-11-04. End-to-end fine-tuning of EEG foundation models on MET (`scripts/finetune_fm.py`, `results/b1ft/`)
Command: `env/bin/python -I scripts/finetune_fm.py data/met_win4s_200.npz results/b1ft <cbramod|labram> loso,loso_song,strict 4 6` (4 s windows at 200 Hz, microvolt/100; mean pooling + linear head; backbone lr 5e-5, head 1e-3, 6 epochs, fixed, 60 folds on 4 GPUs: 822 s for CBraMod). Trial-level CCC [95% subject-bootstrap CI], valence / arousal:
- CBraMod: loso -0.037 [-0.086, 0.002] / -0.023 [-0.049, 0.004]; loso_song -0.013 [-0.047, 0.021] / -0.062 [-0.078, -0.050]; strict -0.079 [-0.119, -0.040] / -0.026 [-0.056, 0.004].
- LaBraM: loso -0.029 [-0.090, 0.019] / -0.061 [-0.113, -0.008]; loso_song 0.014 [-0.084, 0.113] / -0.042 [-0.104, 0.007]; strict -0.140 [-0.197, -0.084] / -0.025 [-0.088, 0.030].
Measured: fine-tuned CBraMod and LaBraM did not predict MET valence/arousal for unseen subjects or songs (all CCC <= 0.014; several slightly negative). Negative CCC in strict folds is the sign of a shift of the predictions relative to the held-out cell, not information.
Summary over MET so far (all with unseen subjects/songs): band-power ridge, handcrafted features, subject-centred features, frozen CBraMod and LaBraM features, EEGNet-style CNN, fine-tuned CBraMod and LaBraM: no method above chance.

## 2026-11-04. B6: BCMI training sessions (OpenNeuro ds002724), within-participant across sessions
Scripts `prep_bcmi_train.py`, `bcmi_train_eval.py`; results `results/b6/ridge_train_sessions.json`. Downloaded 96 EDF recordings = 10 participants (6 with 3 sessions x 4 runs, 2 with 2 sessions, 2 with 1 session; the README speaks of 16 participants), 3,456 trials, 17,280 windows (4 s, 128 Hz, 32 ch), same VA1/VA2 reporting channels as ds002722; no clip identity (music generated in real time; license of the sub-sets as in ds002722: README CC BY 4.0, dataset metadata CC0).
Ridge on log band power (alpha 1), trial-level r / CCC, valence | arousal:
- leave-one-participant-out: r -0.003, CCC -0.001 | r 0.036, CCC 0.006.
- within participant, leave-one-session-out (8 participants with >= 2 sessions): r 0.058 (SD across participants 0.092), CCC 0.017 | r 0.074 (0.056), CCC 0.036; shuffled-label control: r -0.055 (0.027) | -0.065 (0.064).
- within participant, trial-blocked 5-fold inside the same sessions (10 participants): r 0.120 (0.121), CCC 0.037 | r 0.152 (0.128), CCC 0.052.
Interpretation: the within-participant association is about twice as large when train and test trials come from the same session as when they come from a different session, and it is near the shuffled-label level across sessions; this fits slow within-session drift shared between EEG and ratings rather than stable emotion information (not tested directly). Nothing transfers to unseen participants.

## 2026-11-04. Memorisation check and demos
`scripts/memorization_check.py` (MERT layer-6 time-mean features, cosine similarity after train-set standardisation, nearest PMEmo training song): generated clips (108 each) A8k_s2 CFG 3: mean max-cosine 0.44 (max 0.56); A8k_s2 CFG 5: 0.43 (max 0.60); A8k_s1 CFG 5: 0.41 (max 0.56). Held-out real songs to training songs: mean 0.50, 95th percentile 0.64, max 1.00 (one held-out song has an identical copy among the training songs, i.e. PMEmo contains a duplicate). Measured: no generated clip is closer to a training song than typical held-out real songs are; this is one embedding and says nothing about perceptual copying.
Demos: `demos/` (9 clips, 10 s, A8k_s2 CFG 5, one per target cell), README with the exact command. Rights of the demo audio (trained on PMEmo clips of unclear dataset license) have to be settled before the repository is made public.

## 2026-11-04. Fine-tuned foundation models on BCMI ds002722 (`results/b5ft/`)
`finetune_fm.py data/bcmi_win4s_200.npz results/b5ft <cbramod|labram> loso,strict 4 6` (8,150 windows of 4 s at 200 Hz, 1,630 trials; trials cut by the recording end dropped). Trial-level CCC [CI] valence | arousal:
- CBraMod: loso valence CCC -0.004 (r -0.020), arousal CCC 0.014 [-0.010, 0.041] (r 0.033); strict -0.017 | -0.007.
- LaBraM: loso -0.031 [-0.060, -0.004] | 0.008 [-0.014, 0.036]; strict -0.024 | -0.001.
Measured: no above-chance prediction for unseen participants (and unseen clip codes) with either fine-tuned model; consistent with ridge and CNN on the same data.

## 2026-11-04. Failure investigation on MET (why EEG -> valence/arousal stays at chance); scripts `met_diag_a.py`, `met_diag_b.py`, `met_diag_c.py`, results `results/diag/`
Launch (server, `~/music_therapy`, env `env`, one H200 each): `env/bin/python -I scripts/met_diag_a.py data/met_win4s.npz results/diag/met_diag_a.json`; `... met_diag_b.py ... results/diag/met_diag_b.json`; `... met_diag_c.py ... results/diag/met_diag_c.json`. Same 4 s windows / 66 channels / 396 trials as B1.

Measured:
- A. Rating structure (additive subject + song + residual, share of variance; valence / arousal / like / familiarity): song 0.50 / 0.67 / 0.36 / 0.25, subject 0.06 / 0.05 / 0.09 / 0.44, residual 0.44 / 0.28 / 0.55 / 0.31. Split-half reliability of the 20 song means over random halves of subjects (Spearman-Brown): 0.96 / 0.98 / 0.92 / 0.94. Mean correlation of a subject with the leave-one-out song mean: 0.71 / 0.82 / 0.59 / 0.64.
- B. EEG band-power features (330 per trial, subject x song table): mean variance share subject 0.29, song 0.04 (max over features 0.085), residual 0.67.
- C. Mass-univariate (subject-centred trial band power vs subject-centred rating, 5,000 within-subject permutations, family-wise by max |r| over 330 features): max |r| valence 0.128 (C1 theta), arousal 0.144 (AFz delta), like 0.132 (P3 delta), familiarity 0.180 (CP4 alpha); family-wise p 0.53 / 0.30 / 0.44 / 0.035. Only familiarity crosses 0.05 (four targets tested, uncorrected across targets).
- D. Stimulus-locking positive control (leave-one-subject-out CorrCA, 3 components, template = other subjects' projected series of the same song, nominal onset alignment): song identification accuracy 0.0375 (chance 0.05); matched minus mismatched correlation 0.00003; category (5 classes) accuracy from predicted song 0.149 (chance 0.2). Per-trial matched score vs subject-centred rating: valence r 0.001, arousal r 0.107 (p 0.03, 2,000 within-subject permutations, uncorrected), like -0.03, familiarity 0.03.
- F. Same control made tolerant to onset jitter: CorrCA with lag search +-3 s: song-id accuracy 0.058, matched minus mismatched 0.0001; slow band-power envelope template (66 ch x 5 bands, 4 s steps): accuracy 0.056, matched minus mismatched 0.005 (chance 0.05).
- E. Covariance tangent-space features (5 bands, 66x66 covariance per trial, log map; per-subject re-centring using that subject's unlabeled trials, i.e. transductive; kernel ridge, leave-one-subject-out, trial level; shuffled-label control): best CCC valence 0.051 (r 0.061, 95% CI -0.06..0.17), arousal 0.058 r with subject-centred target (CI 0.01..0.11, CCC 0.024); without re-centring valence r -0.04..-0.08, arousal r 0.04..0.05; shuffled-label r between -0.07 and 0.07. Song category (5 classes) accuracy 0.197 aligned / 0.151 unaligned (chance 0.2); song id accuracy 0.035 / 0.050 (chance 0.05).
- Source of the "8 channels" claim: the IEEE DataPort page of MET (found by web search, page not opened by us) is reported to list 8 channels (FP1, FP2, F3, F4, F7, F8, T7, T8); the OpenNeuro release ds008701 that we use has 66 EEG channels. No audio files are in either release as far as we saw (0 audio files in ds008701).

Interpretation (not proven):
1. The labels are learnable in principle only through the song: song explains 50-67% of valence/arousal variance and the song-mean predictor has r 0.67/0.79, while the EEG features carry almost no song variance (4%) and the song cannot be identified from EEG across subjects by any of three methods, including the one used for inter-subject correlation work.
2. Because the positive control (matching the song across subjects) fails, the failure is upstream of emotion: either the MET EEG is not time-locked to the music at the precision these methods need (the dataset has no trigger channel, `TriggerChannelCount` 0, and the README itself warns of playback/hardware delays), or music-driven EEG is too weak at 1 s-4 s resolution with 66 channels after the dataset's own preprocessing. Our tests cannot separate these two.
3. Within-subject noise: 67% of feature variance is residual and 29% is subject; the labels have 20 values per subject, so a subject-independent regressor sees 19 x 20 trials.
4. Published near-chance leave-one-subject-out results for auditory EEG emotion exist (about 30% for four classes, few-shot calibration helped; see sources) and Khan et al. 2025 report that window-level k-fold splits inflate DEAP accuracy (our random-window CNN CCC 0.89 matches that pattern). Neither source tests MET.

Sources opened via search only (summaries, not full papers; verify before citing): arXiv 2508.02417 (Khan et al., DEAP review process failures), Sensors 2026, 26(15):4971 (auditory EEG emotion, LOSO near chance, few-shot helps), Madsen et al., Sci. Rep. 2019, doi 10.1038/s41598-019-40254-w (inter-subject correlation vs preference), FACE arXiv 2503.18998 (few-shot adapter), Euclidean-alignment evaluation arXiv 2401.10746 (motor imagery only).

Not tried yet (in TODO): onset verification against audio (needs MET song audio / identity), per-subject few-shot calibration, ISC-based engagement features on a dataset with triggers (NMED-T has triggers but no valence/arousal), second dataset with confirmed positive control.
