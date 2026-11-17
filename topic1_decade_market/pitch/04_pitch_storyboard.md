# Pitch storyboard (5 minutes, ~8 content slides, matching the four blocks in `final_proj_pitch.pdf`)

Speaker time ≈ 35 s per slide. Slides are sent to the TAs one day before; the TA laptop is used (no live demos that need the network, embed audio/screenshots).

| # | Slide | Content (use the numbers in `03_progress_snapshot.md`; mark plans as plans) | Time |
|---|---|---|---|
| 1 | Title | "Time Traveler: two-axis (era × market) cover generation" · team · topic 1 of the project guide | 15 s |
| 2 | Innovation (what + origin) | One fixed song, a 2-D control plane (release era × release market); the cover changes as the point moves. Origin: guide topic 1 "era-controlled cover generation"; lecture 11b: cover generation = content-preserving style transfer, adaptation includes translating into another language. Gap: SongEcho and MPEcho control melody/lyrics but not era or market | 40 s |
| 3 | Why it fits the course | Uses lecture techniques: adapters (EiLM, MuseControlLite), multi-condition guidance (APG), REPA (ACE-Step layer 8 ↔ MERT), DPO with automatic rewards (LeVo/JAM), phoneme/length regulator (MPEcho) | 30 s |
| 4 | Where we are (pilot evidence) | Dose–response table (adapter vs control, 3 seeds: a0.5 3.25 vs 2.15 MERT index); discrete 2D plane page; baselines on the common protocol. Honest limits: market only via sung language; FAD not consistent | 45 s |
| 5 | Technical objectives & challenges | Why hard: (1) content vs style has no agreed definition, (2) era/market labels are confounded with language, genre, bandwidth, (3) market style beyond language has no descriptor, (4) evaluation is "a mess" (lecture 11b). Goals: H1–H5 from `02_direction_review.md` | 50 s |
| 6 | Approach (the new part) | Diagram: melody/F0 + phoneme stream (MPEcho) + coordinate (era, market) → ACE-Step with IA-EiLM adapter; REPA-style layer-8 alignment; decomposed guidance ω_era, ω_market, ω_melody → continuous plane; DPO with probe rewards (amortised search) | 50 s |
| 7 | Evaluation protocol | 4-part panel: content (RPA/RCA/OA, chroma, song-ID), control (independent probes, dose Spearman, interaction), quality (FD, KL, CLAP, FAD_inf with caveat), language (Whisper LID, PER). Baselines table. No listening test (scope statement) | 35 s |
| 8 | Feasibility & milestones | Papers read in full; code available (ACE-Step, SongEcho, MPEcho, MuseControlLite); data held (PA1 subset, Discogs-VI metadata) and pending (TA: audio); compute: 4 x NVIDIA H200 NVL (143,771 MiB each, nvidia-smi spec only; no server name); adapter training at SongEcho/MPEcho scale fits one GPU; milestones M1–M5 with intermediate artefacts; pilots done; risks + fallback | 45 s |
| 9 | Team | Three members, department/major only: James Christian (NTU-EE); Liang-Yu Hsia and Jefferson Sy Dionisio (**department/major to be filled by the team**). No ML/music background, no lab | 15 s |

Rules for the slide-maker: no Claude/Anthropic mention; no claim beyond `03_progress_snapshot.md`; mark H1–H5 as hypotheses; cite papers by name + venue + arXiv id exactly as in `01_course_deep_dive.md` and say "abstract only" where applicable; do not show vocal-containing audio or lyric text; no bug history.
