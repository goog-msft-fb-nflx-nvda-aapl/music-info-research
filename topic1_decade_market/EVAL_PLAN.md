# Evaluation plan for the era × market grid (draft, 2026-11-07, revised)

Scope decision (user, 2026-11-07): no new-metric contribution. The paper evaluates with existing generation metrics (FAD as the audio counterpart of text-to-motion FID, R-Precision, MM-Dist, Diversity, MultiModality, content preservation); the focus stays on the two-axis generation. Section B items are descriptive diagnostics already computed in the experiments (hit rate, monotonicity, leakage), reported as analysis and not claimed as new metrics. Whether motion-field metric critiques apply to music is not argued here.

Principle (user): reuse generic generation metrics from text-to-motion and text-to-music as the base, add new metrics only for what they do not cover (controllability along two axes).

## A. Generic metrics reused (map from text-to-motion / text-to-music practice)
| Generic metric (source field) | Here |
|---|---|
| Fréchet Audio Distance, FAD (Kilgour et al., arXiv 1812.08466, verified) | per-target-era and per-target-market FAD of generated vs. real clips of that cell (embedding: MERT/MuQ/CLAP; report which); also generated vs. unconditioned baseline for quality drift |
| R-Precision / top-k retrieval accuracy (text-to-motion) | condition-retrieval: rank the true (era, market) cell among all 36 cells by a frozen classifier; top-1/top-3 |
| MM-Dist (text-to-motion) | distance between embedding of generated clip and the prototype embedding of its target cell |
| Diversity (text-to-motion; TEMOS abstract names diversity, arXiv 2204.14109, verified) | mean pairwise embedding distance across seeds/clips in one cell |
| MultiModality (text-to-motion) | same song, same cell, different seeds: within-cell spread |
| CLAP score / KL (text-to-music) | optional text-condition agreement; not the main axis |
| Content preservation (cover task) | chroma/melody similarity to the source song (existing `13_content_metrics.py`: top-1 / cosine) |

Citation status: R-Precision, MM-Dist, Diversity, MultiModality definitions originate in the text-to-motion literature (Guo et al., CVPR 2022 and later); exact reference not yet verified, to be opened before citing.

## B. Diagnostics (descriptive, not claimed as new metrics)
1. Hit rate and gain vs. neutral with bootstrap CI (existing `145_grid_metrics.py`).
2. Adjacent vs. far-decade mass and Spearman monotonicity along the era axis (existing).
3. Dose-response smoothness: monotone, smooth change of classifier probability as the control moves continuously (FreeSliders-style range/smoothness/preservation, arXiv 2511.00103, verified).
4. Cross-axis leakage: change of the era read-out when only the market control moves, and the reverse; disentanglement score (CoInD / dependency-aware idea, arXiv 2503.01145, verified).
5. Independence from the training probe: read-out by a different classifier (Discogs-EffNet already used).
6. Real-reference check: era classifier accuracy on real clips of the cell (ceiling reference label only as measured value).

## C. Known risks
- All read-outs are classifiers trained on the same Discogs-style data; the independent EffNet read-out and per-cell FAD against real audio reduce circularity.
- Market read-out is language-dominated; report language-erased controls.
- No listening test (standing rule); quantitative only.

## D. Next steps (also in TODO.md)
Implement per-cell FAD, MM-Dist, Diversity/MultiModality and leakage in `src/145_grid_metrics.py` once this round's grids are scored.
