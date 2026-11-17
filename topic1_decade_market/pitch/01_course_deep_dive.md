# Deep dive: techniques from the generation lectures that bear on era- and market-controlled cover generation

Status key: **[read]** = full text or full HTML/PDF opened this session; **[abstract]** = abstract or project page only; **[slides]** = stated on a lecture slide, paper not opened. Nothing below is a result of ours unless marked "ours".

Lecture sources (in `course_materials/`): lecture 11b (cover generation, Chih-Pin Tan), lecture 10 (cover generation, earlier copy), lecture 09b (song generation), lecture 06/08b (text-to-music diffusion), final_project_ideas, final_proj_pitch.

## 0. How the course frames our task
- Cover generation = **content-preserving style transfer**: content = melody + chord (+ structure), style = everything else (genre, instrument, arrangement). Adaptation also covers translating a song into another language (lecture 11b, slides "Song adaptation", "Problem formulation").
- Stated open problems on the slides: paired data are hard to get; "sounds similar" has no measurable definition; **"Evaluation - what a mess"**: style is ill-defined, metrics borrowed from general generation are unsuited, high similarity ≠ good cover. The lecturer invites contact for cover-generation work.
- Datasets named: POP909, LargeSHS (SecondHandSongs), Discogs-VI, Pop2Piano, HookTheory. Our topic (guide topic 1) uses Discogs-VI and lists "era-controlled cover generation" as the possible extension.

## 1. The cover-generation papers on the slides (what each does, what we can take)

### SongEcho (Li et al., ICLR 2026, arXiv 2602.19976) [read, ~80 % of the HTML]
- Built on **ACE-Step** (DiT, flow matching). Condition = vocal **F0 contour at 100 Hz (RMVPE)**, encoded by 1D convs, interpolated to the DiT length.
- **EiLM** (element-wise linear modulation): h' = γ(c) ⊙ h + β(c) with γ, β having the *same shape as the hidden state* (per element, per time step), zero-initialised ((γ+1)⊙h + β). Contrast: FiLM = one scale/shift per channel. Injected **before the FFN** of every block, not before self-attention (ablation: before-attention worse on melody metrics, RPA 0.619 vs 0.708).
- **IACR** (instance-adaptive condition refinement): c_i = tanh(L_h(h)) ⊙ tanh(L_m(m)), so the modulation depends on the current hidden state (ablation: EiLM without IACR RPA 0.680, FD 75.3 vs full 0.708 / 42.1).
- Trainable: 49.1 M (3.07 % of ACE-Step+SA-ControlNet 1.6 B; 14.8 % of ControlNet+LoRA 331 M; 26.0 % of MuseControlLite 189 M). 3×A100, 30 k steps, batch 12. Data: **Suno70k** (69,469 AI-generated songs, ~3,000 h, tags from Qwen2-Audio).
- Main table (Suno70k test): RPA .7080, RCA .7339, OA .6952, CLAP .3243, FD_openl3 42.06, KL_passt .1123, PER .2951 (ACE-Step PER .4168; MuseControlLite RPA .5205).
- **Limits stated**: no paired original–cover data; vocal timbre control only by gender; no era/market/genre-control evaluation. Tag-conflict only shown on the demo page ("prioritises the source melody"). **No style/era control is evaluated → our axis is not covered.**
- Take-away for us: this is the course's SOTA cover baseline and its adapter is the natural place to add an *era/market coordinate*; its metrics (RPA/RCA/OA, CLAP, FD, KL, PER) are a ready-made content-preservation panel we can report next to our probe-based control metrics.

### MPEcho (Lee, Yeh, Hu, Tan, Tsai, Yang, ISMIR 2026, arXiv 2607.26698) [read, full PDF text]
- Adds to SongEcho a **phoneme encoder (4-layer FFT, d=256) + length regulator** (FastSpeech-style) so each sung phoneme gets explicit onset/offset; phoneme timings come from **Phonsa**, a Whisper-based phoneme aligner (MAE 32.6 ms vs MFA 233.9 ms on GTSinger). Trainable: 53.3 M IA-EiLM + 0.33 M melody encoder + 12.1 M phoneme encoder; single GPU, 50 k steps.
- Data: **internal Chinese (Mandarin) songs, 13,045 tracks, ~1,427 h**; Phonsa trained on Mandarin SVS corpora (M4Singer, Opencpop). Stated limit: single singer; **multilingual phoneme modelling listed as future work**.
- Results (Table 2): SongEcho PER .4562 → MPEcho M+P .1865 with RPA .5764 (melody-only SongEcho RPA .5779). Phoneme-only collapses melody (RPA .0667): melody and phoneme conditions are complementary. JAM-style word-level arrangement PER .7125 (worse): *structural clarity of the condition matters for a small adapter*.
- **Multi-condition guidance (Eq. 4)**: separate scales for text (ω_t), lyrics (ω_l), and time-varying controls (ω_a), built from differences of conditional/unconditional branches plus **APG** (adaptive projected guidance, Sadat et al. ICLR 2025) to avoid saturation. Best (15, 7.5, 5.0): RPA .6241, PER .1793; MOS 3.88/3.59/3.37/3.57 (25 listeners).
- Take-away for us: (i) a lab-internal, directly relevant paper that supports "lyrics" as a controllable axis through explicit phoneme+duration conditioning; (ii) its guidance decomposition is exactly the tool for **independent scales on era, market and melody**; (iii) it is Mandarin-only, so cross-language/translated lyrics (our market route) is an open extension, not covered.

### FlexCover / "MuLaCover" (NeurIPS 2026 poster; project github.com/HeartMuLa/MuLaCover; the slide title is "MuLaCover: Flexible Cover Song Generation via Symbolic Tonal Control") [abstract, poster page only]
- Conditions a pretrained text-to-song model on a **symbolic lead sheet (melody + chords)**, lyrics and style through text; alignment-free (outputs may change tempo/structure); curriculum with partially mismatched audio–symbolic pairs. No arXiv link, base model or data stated on the page; no numbers → **cannot cite numbers**; treat as unverified beyond the abstract.
- Take-away: symbolic lead-sheet condition (instead of chroma/F0) as the content channel (our TODO T-12).

### Pop2Piano, PiCoGen/PiCoGen2, Etude, AccoMontage, BOSSA (symbolic / piano routes) [slides; Etude arXiv 2509.16522 seen in search only]
- Cover = extraction (SheetSage / transcription) + generation; PiCoGen2 syncs training data by beat; BOSSA learns a **style embedding from audio via a BLIP-2 Q-Former** and conditions a symbolic generator (NeurIPS 2025 AI4Music workshop). Take-away: **BOSSA's cross-modal style bootstrap is a template for a language-free "market/era style vector" learned from audio** (our S5), instead of one-hot labels.

## 2. Adapter / control mechanisms (lectures 06, 08b)

### MuseControlLite (Tsai et al., ICML 2025, arXiv 2506.18729) [read, full HTML]
- Decoupled cross-attention branch with **RoPE on queries, keys AND values**; text cross-attention untouched; output through a **zero-init 1D conv**; K/V weights copied from the pretrained ones. Ablation: w/o RoPE melody accuracy 10.7 % vs 58.6 %.
- Conditions: melody (CQT top-4 pitches), dynamics, rhythm (beat/downbeat), audio (inpainting/outpainting with a separate adapter + complementary masks). 85 M trainable; 40 k steps on one RTX 3090; MTG-Jamendo 1.7 K h (vocals removed).
- **Multi-condition guidance**: ∇log p = ∇log p(x) + Σ_i λ_i[∇log p(x|c_≤i) − ∇log p(x|c_<i)] with λ_text 7, λ_attr 2, λ_audio 1.
- Take-away: we already use it as an external baseline (steering evaluation); the guidance form is a **sequential, order-dependent decomposition** that fits "text → era → market → melody".

### Audio Prompt Adapter (Tsai et al., ISMIR 2024, arXiv 2407.16564) [abstract]
- AudioMAE features through attention adapters into AudioLDM2; timbre/genre transfer, accompaniment generation; 22 M trainable. Take-away: **reference-audio style conditioning** as the alternative to label conditioning (our T-4m reference-audio test found no consistent market move, so the *encoder* of the reference is the open question).

### Music ControlNet (Wu, Donahue, Watanabe, Bryan, TASLP 2024, arXiv 2311.07069) [abstract]
- Time-varying controls (melody, dynamics, rhythm) with partial-control masking; 35× fewer parameters than MusicGen. Take-away: masking partial controls = **partial era/market control (e.g. era only in the chorus)**.

## 3. Training-time and inference-time techniques from lectures 06/08b/09b

### REPA (Yu et al., ICLR 2025, arXiv 2410.06940) [abstract] and its use in ACE-Step [slides, lecture 09b]
- REPA pulls hidden states of a diffusion transformer toward clean-sample features of a frozen self-supervised encoder; aligning only the first few blocks is enough (abstract).
- **Lecture 09b, slide "ACE-Step": ACE-Step aligns an intermediate Linear-DiT layer (the 8th of 24) to MERT and HuBERT embeddings; REPA is "critical" for lyric–audio alignment.** (Slide statement; the ACE-Step paper itself was not opened.)
- **Why this matters to us (interpretation):** our era read-outs are MERT/MuQ probes. If ACE-Step layer 8 already lives in MERT space, an *era-aware alignment term at that layer* (pull the layer-8 features of a generated clip toward the MERT era-probe direction of the target era) is a principled, cheap training signal that uses the same representation the generator was trained with.

### MIDI-REPA (Yu, Tsai, Yang, arXiv 2610.20967, 14 Oct 2026) [abstract and project page]
- Aligns a DiT's intermediate features with a **piano roll from automatic transcription (MuScriptor)**; best = early layer, whole flow trajectory; FluxAudio-S 122 M; trains in ~31 h on one RTX 3090; releases JamendoMIDI (51,077 instrumental tracks, CC BY-NC-SA 4.0). Loss, layer index and numbers not on the page. Take-away: "symbolic supervision as an auxiliary alignment signal" → our T-12 (MIDI vs chroma melody condition) and an **auxiliary MIDI/chord alignment to protect content preservation while the era/market signal pushes style**.

### Classifier-free guidance variants, APG (Sadat et al., ICLR 2025, arXiv 2410.02416) [abstract]
- Decomposes the CFG update into parallel/orthogonal parts relative to the conditional prediction; the parallel part causes oversaturation; adds rescaling and momentum. Used inside MPEcho/SongEcho. Take-away: **guidance with separate, bounded scales per axis** is the cleanest route to a *continuous* 2D plane without retraining.

### DPO / preference post-training (LeVo, NeurIPS 2025; JAM, 2025) [slides, lecture 09b]
- LeVo: preference data from (a) lyric alignment via ASR, (b) prompt consistency via MuQ-MuLan/CLAP, (c) musicality via ratings/SongEval; DPO post-tuning. JAM uses SongEval as DPO reward. Take-away: **our automatic read-outs are exactly such reward signals** (era probe, Whisper language ID / lyric PER, melody similarity, SongEval/FD). We already show inference-time search (best-of-N, SMC) raises the effect; DPO would *amortise* that search into the adapter.

### Lyrics control in song generators: YuE (ICLR 2026), DiffRhythm, ACE-Step 1.5, JAM [slides]
- Sentence-level (DiffRhythm), word+phoneme-level (JAM) timing; ACE-Step 1.5 = current SOTA open song model per the lecture. Take-away: precision of **timing** is what separates intelligible from garbled lyrics; our translated lyrics currently have no timing constraint.

## 4. Mapping: lecture technique → gap in our current system

| Gap in our results | Course technique | Why it could help | Cost |
|---|---|---|---|
| era effect weak on middle decades, ACE-Step LoRA pilot weak (1990s/2000s ≈ 0) | EiLM+IACR adapter (SongEcho); REPA-style alignment at ACE-Step layer 8 | state-dependent, per-time-step modulation; alignment to the representation the base model already uses | adapter training, ≤ 1 day on one GPU (SongEcho-scale) |
| 2D plane is built by interpolating a conditioning vector (dose) | multi-condition guidance with APG (MPEcho Eq. 4, MuseControlLite Eq. 9) | separate scales ω_era, ω_market, ω_melody → continuous plane at inference, no retraining | inference only |
| inference-time selection (best-of-N, SMC) is expensive | DPO with probe rewards (LeVo, JAM) | amortise search into the weights | preference pairs from existing generations + one DPO run |
| market axis = sung language only; translated lyrics have no rhythm constraint | phoneme+duration conditioning with a length regulator (MPEcho/Phonsa); syllable-constrained translation (GagaST, MAVL, Lyra: see SURVEY6.md) | explicit phoneme/duration slots → intelligible translated singing at the original melody's timing | needs a duration-aware translation step + ACE-Step adapter retrain with multilingual phonemes (MPEcho is Mandarin-only) |
| market beyond language: no descriptor | BOSSA-style audio style embedding (Q-Former) | learn a language-free market/era style vector from audio instead of a label | one encoder + adapter |
| content preservation measured by one song-ID retrieval | SongEcho metric panel (RPA, RCA, OA, CLAP, FD, KL, PER) | comparable to the SOTA tables; lecture says evaluation is the field's open problem | scripts only |
| chroma vs symbolic content channel | FlexCover lead sheet; MIDI-REPA; Piano2Pop/MIDI-SAG (unverified) | stronger identity at higher style change | T-12 |

## 5. Verification status
Opened in full: SongEcho (HTML), MPEcho (PDF), MuseControlLite (HTML). Abstract/project page only: FlexCover, MIDI-REPA, REPA, APG, Audio Prompt Adapter, Music ControlNet. Slides only (papers not opened): ACE-Step layer-8 MERT/HuBERT alignment claim, LeVo/JAM DPO details, YuE, DiffRhythm, BOSSA, PiCoGen2, Etude. **Open these before citing in the paper.** Earlier verified entries (GagaST, PolySinger, MAVL, LOAF-M2L, Music for All) are in SURVEY6.md.
