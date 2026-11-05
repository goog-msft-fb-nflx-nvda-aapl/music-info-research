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
