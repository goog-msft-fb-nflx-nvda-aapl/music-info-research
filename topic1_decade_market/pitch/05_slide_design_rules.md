# Slide design rules (the previous decks were criticised for too many words and unsupported claims)

## Words
- One message per slide, stated as the **slide title (a full sentence, max 12 words)**. Body = one diagram or one small table, plus at most 3 short bullets of max 10 words each. Hard cap 40 words of body text per slide (title and captions excluded). Everything else goes into speaker notes.
- No paragraphs, no sub-bullets, no sentences that repeat the title. Numbers appear in tables or on the diagram, not in prose.
- 5 minutes ≈ 9 slides; if a slide needs more than 35 s to say, split the diagram, not the text.

## Claims
- Every number, comparison or "works / does not work" statement carries an **evidence tag** in small grey text, e.g. `[E3]`; the tag resolves in `claim_ledger.csv` (claim → evidence file → status). A claim without a ledger row must not appear.
- Status vocabulary is fixed: **Measured** (our result file), **Reported** (from a paper we opened; name the paper), **Slide-only** (stated on a lecture slide), **Hypothesis** (not run), **Plan**. Show the status as a coloured chip next to the claim (Measured = solid, Hypothesis = outlined).
- Forbidden wording: "state of the art" (unless the paper says it and is cited), "novel" without naming what exists (SongEcho, MPEcho, MuseControlLite), "proves", "significantly" (without CI), "realistic audio", "ceiling", "best possible", any market-style claim beyond sung language.
- The honest limits stay on the slides as one line each (market = sung language only; no listening test; FAD inconsistent), not hidden.
- Finish with a **self-audit**: list every number on the deck and check it against `03_progress_snapshot.md`; report mismatches.

## Visuals (use draw.io for all diagrams)
Diagrams to draw (specs in `06_diagram_specs.md`): D1 positioning map vs existing cover systems; D2 pipeline/architecture of the proposed system; D3 the 2D control plane with the two guidance axes; D4 evaluation panel; D5 milestone workflow. Tables only for numbers. One accent colour, grey for context, status chips as in the claim rules. Font ≥ 18 pt on slides. Export diagrams as SVG/PNG at 1920 px width and place them full-width.

## Workflow for the assistant
1. Build `claim_ledger` check: read `claim_ledger.csv`; every slide claim maps to a row.
2. Draw D1–D5 with the draw.io connector/plugin; show them to the owner as inline previews if the connector supports it.
3. Assemble the deck (PPTX skill or slides artifact), images from step 2, notes per slide.
4. Self-audit (numbers, status chips, word cap per slide), then report the word count per slide.
