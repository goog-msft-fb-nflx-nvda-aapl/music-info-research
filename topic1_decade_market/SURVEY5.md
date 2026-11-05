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
