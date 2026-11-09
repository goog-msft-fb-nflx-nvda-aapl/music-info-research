# Lecture 10 (Synthesis and MIDI-to-audio generation, 2026 edition): relevance to topic 1

Source: `lecture10_midi-to-audio.pdf` (course slides, read in full on 2026-11-09). All references below are as cited on the slides; none has been opened by us yet, so they are not used as citations in the paper until verified (see the rule in `CLAUDE.md`).

## Verdict
Partly. The synthesis part of the lecture (DSP synthesizers, sample-based synthesis, WaveNet autoencoder, GANSynth, InstrumentGen) is not relevant to era/market-controlled covers. Three later parts are relevant, as a source of ideas and baselines, not as a replacement of the current method.

## Relevant items
1. **Symbolic melody condition instead of chroma (slides 49-55: Piano2Pop, MIDI-REPA, multi-track MIDI to audio).** Piano2Pop (Cheng et al., arXiv 2026, as cited on the slides) conditions an ACE-Step 1.5 DiT on a pianoroll encoder and trains on transcribed MIDI to bridge to real MIDI. Our cover system conditions MusicGen-Melody on chroma of the original mix (HPCP substitute) and ACE-Step on lyrics plus audio. A transcribed-MIDI melody condition is a candidate replacement of chroma for the content-preservation arm and for the ACE-Step era route; it gives an editable melody. Experiment item: transcription of the original vocal/accompaniment, MIDI-conditioned generation with the same era dose, same read-outs (T-12 in `TODO.md`).
2. **Singing voice synthesis route for the market axis (slides 29-30, 47).** The lecture states that monophonic note-to-audio is the SVS problem (duration predictor, note transitions). Our market axis currently swaps the vocal stem produced by ACE-Step cover mode. A melody-driven SVS in the target language (open item T-4h-b: Spanish-capable synthesizer) fits this description. The slides give no concrete multilingual SVS model, so this does not close the open item.
3. **Low-level controls (slides 34-46: MIDI-DDSP, DDSP-Piano, guitar flow matching).** F0 and loudness as separate controls. Possible use: keep F0 and loudness of the original while the era dose changes timbre and production; relates to the existing quality/loudness/bandwidth confound (TODO item: condition on loudness and bandwidth next to z). Not a near-term experiment.
4. **Accompaniment generation from MIDI (slide 50: MIDI-SAG, arXiv 2026.02 as cited).** Closest published setting to ours (generate an accompaniment around a given vocal). To be read as a baseline candidate for the cover system; checking code and weights is a TODO item.

## Not relevant
DSP synthesis types, PSOLA, sample-based libraries, WaveNet and autoencoder single-note models, InstrumentGen, DDSP-Piano as instrument models.

## Actions (added to TODO.md)
- open and verify the cited papers (Piano2Pop, MIDI-SAG, MIDI-REPA, MIDI-DDSP) before any use in the paper;
- test MIDI (AMT transcription) as melody condition against chroma on the same 24 songs, same era dose read-outs.
