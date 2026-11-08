# Survey round 6 (2026-11-05): style control for music generation (verified abstracts)

Question: which state-of-the-art methods give continuous, composable style control (a 2D era x market plane) on a music generator?

Verified (abstract opened 2026-11-05):
- Tsai et al., MuseControlLite, ICML 2025, arXiv 2506.18729: lightweight conditioners (RoPE in decoupled cross-attention) on Stable Audio Open for time-varying attributes and reference audio; separate guidance terms per condition (per search summary; the guidance detail is not in the opened abstract, check the paper before citing it).
- Facchiano et al., Activation Patching for Interpretable Steering in Music Generation, arXiv 2504.04479: difference-in-means steering vectors for binary concepts (tempo, timbre) added to activations of a text-to-music model.
- Narashiman et al., Genre Controlled Music Generation via Activation Steering, arXiv 2506.10225 (v2 May 2026; v1 title "Fine-Grained control over Music Generation with Activation Steering"): MusicGen activations steered with linear-probe weights for genre control and blending.
- ACE-Step Concept-Slider LoRA "mood" (HF `lukasz-staniszewski/ace-step-cs-mood-r8-all`, TADA steering collection): scalable LoRA slider on ACE-Step; the model card has no paper link or description, so no claims beyond existence.
Search result not yet opened (do not cite): Composer Vector (arXiv 2604.03333, symbolic), Audio Conditioning for Music Generation (arXiv 2407.12563, double CFG for text + audio), Music ControlNet (arXiv 2311.07069), ACE-Step 1.5 (arXiv 2602.00744), Stylus (arXiv 2411.15913).
Not found: any method conditioning on release decade or market, or a 2D control plane for music (standard search, two rounds; consistent with SURVEY4 round 5).

Interpretation: closest design to the target is per-axis guidance (MuseControlLite-style multiple CFG terms) with an activation-steering or slider alternative; our z-conditioned adapter with three-branch CFG is already this pattern for era.

## Cross-modality methods for two-parameter style control (added 2026-11-05; the novelty claim stays "no prior work on era x market cover generation", transfer of methods is allowed)
Abstracts opened 2026-11-05 (verified):
- Gandikota et al., Concept Sliders, arXiv 2311.12092 (no peer-reviewed venue on the page): LoRA low-rank direction per concept, continuous scale, sliders composable.
- Zhu et al., CompSlider, arXiv 2509.01028, ICCV 2025: conditional prior over several attribute values with disentanglement and structure losses so simultaneous edits do not interfere; base model untouched.
- Brooks et al., InstructPix2Pix, arXiv 2211.09800: two conditions (input image, instruction); the abstract does not define the two guidance scales (read the paper before describing them).
- Buchheim et al., Controlling Human Shape and Pose in Text-to-Image Diffusion Models via Domain Adaptation, arXiv 2411.04724: SMPL shape and pose parameters injected into ControlNet, CFG-vector composition with another control; evaluated against 2D-pose ControlNet.
- Watcharasupat and Lerch, Evaluation of Latent Space Disentanglement in the Presence of Interdependent Attributes, arXiv 2110.05587 (ISMIR 2021 LBD): MIG misleads when attributes depend on each other; dependency-aware MI metric (relevant because decade and market are correlated in Discogs).
Seen in search results only (not verified, do not cite): All-in-One Slider (arXiv 2508.19195, SAE on text embeddings), Measured Sliders (arXiv 2609.05234), Composer (arXiv 2302.09778, subset-weighted CFG), 4DiM multi-guidance (arXiv 2407.07860; its abstract does not describe multi-guidance), Controlla (arXiv 2605.16603), MoViE (arXiv 2412.06578, distilling two-scale guidance), Shape Conditioned Human Motion Generation (arXiv 2405.06778).

Mapping to our problem (interpretation):
- Two axes with independent guidance scales = InstructPix2Pix/Composer pattern: train with random dropping of each condition, sample with one guidance term per axis (our three-branch CFG covers era only; add a market branch).
- Concept-Slider / CompSlider: attribute-specific low-rank or prior module plus an explicit disentanglement loss; our LEACE nuisance control and one-hot prefix tokens correspond to the conditioning side, and a cross-axis leakage loss is the missing piece.
- Parametric-body control (Buchheim): a continuous parameter vector as the condition, with an off-domain problem solved by composing CFG vectors; our era z plays the role of the SMPL parameters, the market axis lacks a clean parameter (language is the carrier).
- Evaluation transfer: dependency-aware disentanglement metrics for correlated attributes, plus per-axis regression error and cross-axis leakage on a grid (CONSOLIDATION.md).

## Replies to SURVEY5_PROMPT.md (2026-11-07; files in `survey5/`: claude.md, gemini.md, qwen.md)
Abstracts opened 2026-11-07 (verified existence, authors, date, one-line content):
- Ezra et al., FreeSliders, arXiv 2511.00103 (Oct 2025): training-free sliders for image, audio and video; extends the Concept Sliders benchmark to audio and video, three evaluation properties with new metrics (names not in the abstract), saturation detection and reparameterised traversal.
- Gaudi, Sreekumar, Boddeti, CoInD, arXiv 2503.01145: conditional diffusion models violate attribute independence, worse when only some combinations are observed; Fisher-divergence regulariser.
- Xia et al., Factored Classifier-Free Guidance (arXiv 2506.14399, v1 titled "Decoupled CFG"): per-attribute guidance from a causal graph.
- Sadat et al., ICG, arXiv 2407.02687, ICLR 2025: independent condition guidance without special training.
- Almudévar and Ortega, Rethinking Disentanglement under Dependent Factors of Variation, arXiv 2408.07016 (v3 Feb 2026): disentanglement definition and metric valid for dependent factors (the TMLR venue stated in the reply was not on the arXiv page; unverified).
- Zhao et al., MusicRFM, arXiv 2510.19127: Recursive Feature Machine directions in frozen MusicGen activations (ICLR 2026 stated in the reply is not on the page; unverified).
- Staniszewski et al., TADA!, arXiv 2602.11910: concepts routed through few consecutive attention layers in audio diffusion; localized activation steering compared with prompt, score-space and weight-space interventions on a benchmark and user study.
- Tong et al., Causal-Adapter, arXiv 2509.24798: structural-causal adapter on frozen text-to-image diffusion with a conditioned token contrastive loss.
- Lertpetchpun et al., Accent Vector, arXiv 2603.07534: task vectors scaled and blended to control accent in multilingual TTS.
Reply defects found: the Qwen reply cites 2026 papers that exist but do not do what it says (BiasReducer, arXiv 2609.32720, edits a reward-model head, it has no cross-attribute leakage metric; MUtE, arXiv 2609.11253, is concept erasure for text); attributes arXiv 2407.12563 to two different titles (it is MusicGen-Style, Rouard et al.); several entries are marked "Unverified Baseline"; do not use the Qwen list. The Gemini reply mixes verified papers with entries labelled "Anonymous" or "Unverified Baseline" and an invented "Audio-CompBench"; use only items confirmed above or in the earlier section.
Synthesis (interpretation), consistent across the Claude and Gemini replies:
- Separate guidance per condition with independent dropout (InstructPix2Pix, MusicGen-Style double CFG, Factored CFG), plus a regulariser for dependence (CoInD), is the standard route; our three-branch CFG covers era and lacks the market branch and the dependence regulariser.
- Training-free steering (MusicRFM on MusicGen, TADA on ACE-Step, FreeSliders) is the cheap baseline for both axes, and TADA's finding (few consecutive layers carry concepts) can localise our adapter injection.
- Confound handling: LEACE (already used) plus an explicit quality/loudness/bandwidth condition so the era vector does not carry them (QA-MDT idea; arXiv 2405.15863 not opened); language is a separately supplied input in ACE-Step (lyrics), so market x language becomes a design choice (reply claim, not verified here).
- Evaluation: independent read-outs plus confound probes, per-axis monotonicity and range/smoothness/preservation (FreeSliders), leakage matrix (CompSlider), dependency-aware disentanglement on read-out embeddings (Almudévar and Ortega, DMIG), FAD with per-era reference sets (Gui et al., arXiv 2311.01616, not opened), read-out validation by rank agreement on real recordings.
- Novelty wording from both replies: no explicit release-decade or market control found; related work is era recognition from audio, cultural adaptation of music generators and accent control in TTS; wording "first to treat release era and market as explicit, jointly controllable axes" (era ordinal, market categorical, not a metric plane).

## Verified 2026-11-08 (abstracts read)
- Gui, Gamper, Braun, Emmanouilidou, "Adapting Frechet Audio Distance for Generative Music Evaluation", arXiv 2311.01616: FAD distorted by sample-size bias, weak embeddings and biased reference sets; extrapolation to infinite sample size; per-song FAD; toolkit. Used in `src/158_fad_inf.py`.
- Li et al., "Quality-aware Masked Diffusion Transformer for Enhanced Music Generation", arXiv 2405.15863: quality-aware training on quality-imbalanced data, masked diffusion transformer, three-stage caption refinement; quality control. (Relevant to the open idea of a quality condition next to z.)
- Lee et al., "Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music", arXiv 2608.10980: Korean songs of the 1960s-1980s classified 4-5 years earlier than the Billboard era, 2-3 years from the 1990s.
- Guo et al., CVPR 2022 (text-to-motion): see EVAL_PLAN.md.
