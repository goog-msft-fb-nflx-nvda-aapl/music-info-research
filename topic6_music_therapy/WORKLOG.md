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
