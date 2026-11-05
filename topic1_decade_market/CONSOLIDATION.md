# Consolidation (2026-11-05): goal, findings, bottlenecks, measurement

Goal. One control plane with two axes: release decade (era) and release market. The user picks a point on the plane; the generated cover keeps the song (melody, chords) and changes only the era and market cues. Classifier-free-guidance-style composition (one guidance term per axis) is the intended interface.

## What works (measured, numbers from WORKLOG)
- Era axis on MusicGen-Melody: z-conditioned adapter with three-branch CFG; target-decade hit (24 clips x 6 eras x 3 seeds = 432) MERT .819, MuQ .764; one-hot+16 prefix tokens MERT .847-.889, MuQ .787-.829 over three training seeds; independent Discogs-EffNet read-out supports 1990s (.53) and 2000s (.60) moves for MusicGen (chance 1/6) but not for the ACE-Step LoRA.
- Content: top-1 .854-.873, own-clip similarity .879-.893, chord agreement .347-.398 (baseline vs HPCP chroma). HPCP chroma (roll by 3) replaces the processor chroma without loss (MERT .854 / MuQ .824).
- Era on ACE-Step LoRA: 1960s, 1980s, 2010s move; 1990s and 2000s do not, also with year-augmented captions (3 epochs).

## Bottlenecks
1. Market axis does not move with timbre, accompaniment or style words; language is the carrier (LEACE language erasure removes the vocal move; M-3 accompaniment encoder .22-.33; CultureMERT-95M no better than MERT-v2).
2. Lyric route changes language but loses singer identity; SoulX-Singer restores identity for English only (no es/pt/de/it).
3. Market read-out instruments read language, not arrangement; no instrument separates the two yet.
4. Pair data for the guide question: Da-TACOS gives year gaps (4,968; 1,548 >= 20 y) and MusicBrainz countries (6,821 ids) but no audio; Discogs-VI audio awaits the TA.
5. 1990s/2000s are weak on the ACE route and on every probe (probe reads them near zero for ACE).
6. Single training seeds almost everywhere; probes are trained on the same data distribution used to read out (partly mitigated by the Discogs-EffNet read-out).

## How to measure a 2D plane (proposal, items in TODO)
- Per-axis hit and monotonic response along each axis with the other axis fixed.
- Cross-axis leakage: change in the market read-out when only era changes, and the reverse.
- Content preservation (top-1, own-clip similarity, chord agreement, VINet identity) at every grid point.
- Two independent read-outs per axis (codec-matched probe + independent tagger/embedding), never only the instrument used for selection.
- Audio quality with reference-free and reference-based scores (FAD, CLAP text score) on the grid; no listening test.
