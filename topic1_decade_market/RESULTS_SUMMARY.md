# Time Traveler: results summary (living document; numbers copied from `WORKLOG.md`, files named in each row)

Last updated 2026-11-04. Every row has its configuration and numbers in `WORKLOG.md` under the date and title given; interpretation is marked there. Plans: `TODO.md`; blockers and ideas: `BLOCKERS.md`; requirements checklist: `TODO.md` (section "Requirements checklist").

## Axis 1, time (era)
| ID | What | Main measured result | Files |
|---|---|---|---|
| era adapter | z-conditioned bias adapter on MusicGen-Melody (ctrlc, ge 10) | MERT Delta P(target decade) 1960s-2010s +.741 / +.591 / +.484 / +.389 / +.225 / +.517; rho_E .957 | `results/six_decades*.txt`, WORKLOG 2026-11-01/02 |
| N-5a | best-of-N with a nuisance-controlled selector | held-out MuQ 1980s .348 (N=1), .471 (8), .565 (16), .542 (32) | `results/best_of_n_ax_ctrlc_bon{,16,32}.*` |
| N-5b | probe-guided SMC (8 particles) | held-out MuQ 1980s .407 (beta 4); scoring only later stages does not help (.372) | `results/smc_ctrlc_N8_*` |
| N-5 paired | best-of-8 vs SMC beta 4 | MuQ: best-of-8 higher in 11 of 12 cells, one cell with CI above 0 (erased 1980s +.109 [+.011,+.201]) | `results/paired_bon_smc.*` |
| T-9 / T-9c / T-9e | decade-specific vocal processing chains | vocal probe 1980s +.32, 2010s +.18 (tuned), 1960s +.10 (tape cues); mix-level 1980s +.13, 2010s +.04, 1960s -.008 | `results/vocal_era_dsp.*`, `vocal_chain_tune{,2}.*` |
| T-9f | era chains on the whole mix (mix-bus) | mix-level gain on the era-axis remix: 1960s +.062, 1980s +.084, 2010s +.092 | `results/mixbus_chain.*` |
| T-9d | vocal chain x era accompaniment additivity | 2010s additive (interaction -.007); 1980s super-additive (+.101 [+.063,+.141]) | `results/vocal_acc_additivity.*` |
| FD / paired metrics | FD-VGG, KL-PaSST, CLAP of the new sets | ctrlc FD 1980s -.72, 2010s -.81, 1960s +.60; CLAP 1980s +.145 | `results/fd_boot_*`, `pair_*` |

## Axis 2, market
| ID | What | Main measured result | Files |
|---|---|---|---|
| M-3 | market encoder on the accompaniment | validation accuracy .22-.25 after erasing style, statistics and language (chance .167) | `results/market_encoder_*` |
| T-4a | FreeSVC timbre conversion to a market voice | no move toward the target market; accuracy loss equals the self-voice control | `results/svc_market_eval*.json` |
| T-4b / T-4g / T-4i | lyric route: translated lyrics sung in the target language | cover mode: language hit .42-.75, song identity lost (VINet cosine .5); lego: identity kept (.81-.83) but language hit .17-.33; stem swap: language hit .25-.88 with identity .80-.84 | `results/lyric_route_eval*.*` |
| T-4k | scale-up of the stem swap, 77 clips, 5 targets | vocal market probe Delta P(target) Spain +.188, Brazil +.421, Germany +.314, Italy +.408, US+UK +.325 (all CIs above 0) | `results/scale_market_probe.*` |
| T-4k seed 1 | replication | Spain +.212, Brazil +.488, Germany +.308, Italy +.381, US+UK +.239; erased probe within +-.03 | `results/scale_market_probe_seed1*` |
| T-4l | same, language erased from the probe | all five targets within +-.03 of 0 (Spain +.023): the move is language | `results/scale_market_probe_leace.*` |
| T-4e / T-4o / T-4p | style descriptors from Discogs-VI metadata | tag score: Brazil +.055 / +.049 / +.063, others small; tagger-selected words raise Germany/Italy tags but the independent probes read them as less German/Italian; independent probes move for Brazil (MERT/MuQ +.145/+.490 lift, +.117/+.313 selected); T-4q swap: probe-selected words inflate the probes (Germany +.19/+.12) but not the tagger (-.006): selection inflates its own instrument | `results/market_styles.*`, `market_style_eval.*`, `mstyle_probe_*` |
| T-4n | FreeSVC with a voice-matched reference | no move (target accuracy .178 / .193, effects within +-.01) | `results/svc_market_eval_nn.*` |
| T-4m | reference-audio style transfer | no consistent move across encoders (Brazil +.105 / +.075 only at strength 0.5) | `results/ace_mref*_eval.*` |

## Both axes
| ID | What | Main measured result | Files |
|---|---|---|---|
| T-1 | Discogs-VI metadata pair counts | 33,578 cliques with a different-artist pair with era and market move (3,172 US-Spain) | `results/discogs_vi_pairs.*` |
| T-10 | era-axis accompaniment x lyric language, 16 songs | era gain the same with source and translated vocal (interactions within +-.05, 1 of 12 CIs excludes 0); language reading independent of the accompaniment | `results/two_axis.*` |

## What the results support, and what they do not
- Supported (probe-based, same-family probes, one seed unless stated): the era axis moves the accompaniment and the vocal-chain adds to it; the language component of the market moves with translated lyrics at scale and composes with the era axis without a measurable interaction.
- Not supported: a market move beyond language (T-4l, T-4m, T-4e), real cover-pair analysis (needs the Discogs-VI audio, pending the TA's answer), a human check (decision: quantitative only).
