# Instructions for the assistant that builds the pitch slides

You receive a folder with everything needed to produce the **5-minute project pitch** for the course final project (topic 1, "Time Traveler: era × market cover generation"). Do these in order:

1. Read `01_pitch_requirements.md` (pitch format from the course) and `final_proj_pitch.pdf`.
2. Read `03_progress_snapshot.md` (measured results, limits), `02_direction_review.md` (proposed technical core H1–H5, evaluation, feasibility, risks) and `01_course_deep_dive.md` (paper-by-paper technique notes, with verification status).
3. Follow `04_pitch_storyboard.md` for the slide list and timing. Build 9 slides (title, innovation, course fit, pilot evidence, objectives/challenges, approach, evaluation, feasibility/milestones, team). Build the deck in **Canva** (Canva connector; no PowerPoint tool is available). Diagrams D1-D5: draw them with the draw.io connector, export as PNG/SVG, and upload/place them in the Canva slides. Put speaker notes in each slide's notes. If the Canva connector cannot import images, give the owner the draw.io exports and the slide-by-slide text instead.

**Read `05_slide_design_rules.md`, `06_diagram_specs.md` and `claim_ledger.csv` before building. The previous decks were criticised for too many words and unsupported claims: respect the 40-word body cap and the evidence tags. Draw all diagrams (D1-D5) with the draw.io connector; if it is not enabled, tell the owner which connector to enable and stop before drawing.**

Hard rules (from the project owner):
- Use only the numbers in `03_progress_snapshot.md` and the verified paper facts in `01_course_deep_dive.md`. Label H1–H5 as hypotheses/plans, never as results. Say "abstract only" for papers marked [abstract]/[slides]. Do not invent datasets, results, team experience, or compute.
- Keep every claim the evidence supports and no more: market control beyond sung language is **not achieved**; FAD/MuQ-FAD does not consistently rank our systems; no human listening test was run (scope decision).
- Do not include vocal-containing audio, lyric text, or any recording from the licensed dataset; only derived numbers and accompaniment-only material. The dataset is licensed for the course and not redistributed.
- No Claude/Anthropic/AI-assistant attribution on the slides. No bug history. No word "solo" (the team has three members).
- Team slide: three members, department/major and lab only (no ML/music background). James Christian: NTU-EE. Liang-Yu Hsia: master's student, GICE, iDSSP lab. Jefferson Dionisio: PhD student, GICE, iDSSP lab (https://www.idssp.ee.ntu.edu.tw/members). Do not add anything else about them.
- Compute (the sheet asks for GPU need): state hardware as the nvidia-smi spec only: "4 x NVIDIA H200 NVL, 143,771 MiB (about 141 GB) each". Do not mention the server name, host, account, location or that it is shared. Adapter training at the scale of SongEcho (49 M trainable parameters) and MPEcho (53 M) fits one such GPU.
- Style: clean academic deck (16:9), one message per slide, tables from the snapshot as small native tables, a simple architecture diagram for slide 6 (conditions → ACE-Step with adapter and guidance). Include `plane.png` (screenshot of the interactive 2D plane prototype) on slide 4. English text.
- Cite as: SongEcho (Li et al., ICLR 2026, arXiv 2602.19976); MPEcho (Lee et al., ISMIR 2026, arXiv 2607.26698); MuseControlLite (Tsai et al., ICML 2025, arXiv 2506.18729); APG (Sadat et al., ICLR 2025, arXiv 2410.02416); REPA (Yu et al., ICLR 2025, arXiv 2410.06940); MIDI-REPA (Yu, Tsai, Yang, arXiv 2610.20967); LeVo / JAM (lecture 09b); Lecture 11b (C.-P. Tan).

Folder map:
- `01_pitch_requirements.md`, `final_proj_pitch.pdf`, `final_project_ideas.pdf` — what the pitch must contain and the topic text.
- `01_course_deep_dive.md` — techniques from the lecture papers and how they map to our gaps.
- `02_direction_review.md` — direction decision, hypotheses H1–H5, protocol, feasibility, risks.
- `03_progress_snapshot.md` — measured progress and limits.
- `04_pitch_storyboard.md` — slide-by-slide plan.
- `results/` — source JSON/TXT of the numbers (weekly tables, two-dose grid, baselines).
- `figures/` — `plane.png`, `weekly1_slides.pdf` (previous deck, for visual style only).
- `lectures/` — lecture 11b, cover generation (lecture 10 copy), song generation (09b).
- `papers/` — SongEcho, MPEcho, MuseControlLite, MIDI-REPA, REPA, APG, Audio Prompt Adapter, Music ControlNet (open-access PDFs).
- `repo_notes/` — RESULTS_SUMMARY, PAPER_OUTLINE, SURVEY6, LECTURE10_NOTE, INTRO_THREAD.
