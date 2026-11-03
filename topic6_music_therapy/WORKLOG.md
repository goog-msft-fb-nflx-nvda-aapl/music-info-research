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
