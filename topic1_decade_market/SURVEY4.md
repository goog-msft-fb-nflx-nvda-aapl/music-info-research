# Survey 4 synthesis (open melody-conditioned accompaniment / cover systems), 2026-11-01
Raw replies: `survey4/response_{claude,chatgpt,gemini}.md`. Rule: only items re-checked against the source in this session count as verified.
## Reliability
- `response_claude.md`: specific, marks unverified items; consistent with our own checks (Stable Audio Open ControlNet of Hou et al. links no weights: project page and our fetch of it; MuLaCover is arXiv-independent HeartMuLa release).
- `response_chatgpt.md`: careful, separates verified/partial/unverified; adds MIDI-SAG, MusiConGen, ACE-Step 1.5 cover checkpoints, YuE2 (not re-checked here).
- `response_gemini.md`: states checkpoints for Music ControlNet and Stable Audio Open ControlNet exist; contradicted by the other two replies and by our own fetch of the project page (no weights linked). Not used.
## Verified by us this session (source opened)
- Hou et al., Stable Audio Open ControlNet (arXiv 2410.05151): project page `stable-audio-control.github.io/web/` links only the paper and datasets, no code or weights. The third-party repo `EmilianPostolache/stable-audio-controlnet` ships audio-conditioned (accompaniment) ControlNets for MusDB / MoisesDB, no melody-control checkpoint.
- Text-to-Music control family (`github.com/fundwotsai2001/Text-to-Music_control_family`, commit de6978b): README lists SongEcho_base (31M trainable), SongEcho_large (198M), MuseControlLite (85M), run with `control_melody_inference.py`, adapters downloaded from the Hugging Face repo `fundwotsai2001/Text-to-Music_control_family`.
## Reported by replies, not re-checked
JASCO melody 400M/1B (10 s, gated, chords + melody salience + drums), MuLaCover (full song, regenerates vocals; style schema without an era field), MPEcho and SongEcho-ACE-Step (regenerate vocals), MIDI-SAG (vocal to backing track, many dependencies), MusiConGen (chords and rhythm, about 30 s), ACE-Step 1.5 cover checkpoints (base, sft, turbo, xl).
## What it changes for us
Same-task baselines (keep the original vocal, regenerate the accompaniment, era text prompt, melody from the mix) with public weights: the three control-family adapters (we ran MuseControlLite only), ACE-Step cover (turbo only, done). Systems that regenerate the vocal (MuLaCover, MPEcho, SongEcho-ACE) answer a different task and are not baselines for accompaniment-only covers. No system reports era-labelled training or era-adherence evaluation (replies), so our decade-control evaluation has no direct external comparison.

## 2026-11-03 — Discogs-VI audio access (first check; to be completed in survey round 4)
Sources: [Discogs-VI paper (arXiv 2410.17400)](https://arxiv.org/html/2410.17400v1), [Zenodo record 13983028](https://zenodo.org/records/13983028), [ISMIR 2024 poster page](https://ismir2024program.ismir.net/poster_166.html), project page mtg.github.io/discogs-vi-dataset and code github.com/raraz15/Discogs-VINet (named in the paper; not opened in this check).
- Discogs-VI-YT: about 493,000 versions of about 98,000 works matched to YouTube URLs of official uploads (paper). The public release gives metadata, YouTube links and precomputed audio representations; audio features beyond the released ones are on request (poster page). The tracks themselves are not distributed; audio would have to be fetched from YouTube by us (terms of use unchecked).
- License: the paper is CC BY 4.0 (UPF repository); the Zenodo metadata record is CC BY-NC-SA 4.0 (as recorded earlier in WORKLOG 2026-10-27); no separate statement for audio found.
- Open: read the project page and Zenodo file list for the exact released features (dimension, model), whether they include the YouTube ids for every version, and the request procedure; decide whether a TA-provided audio copy or our own download is possible.


## 2026-11-03 — Survey round 4 synthesis (replies: `survey4/gemini_v4.md`, `survey4/chatgpt_v4.md`; prompt `SURVEY4_PROMPT.md`)
Status labels: **checked** = I opened the primary page or file myself; **reported** = stated in a reply with a source, not yet checked; **doubtful** = contradicted or unsupported. Interpretation is marked.

**Checked (primary pages, 2026-11-03).**
- [Discogs-VI site](https://mtg.github.io/discogs-vi-dataset/): YouTube IDs only, "We do not share any audio"; a script downloads the files and warns that too many parallel downloads can get the IP banned by YouTube; the extracted features are "available upon request for non-commercial scientific research purposes" (request to the Music Technology Group); metadata CC BY-NC-SA 4.0. The page names CQT features; chroma and HPCP (reported by ChatGPT, from the paper) not confirmed here.
- [Discogs-VINet repository](https://github.com/raraz15/Discogs-VINet) and the local checkout `~/era_market/ext/Discogs-VINet/LICENSE`: **MIT license**; contact recepoguz.araz@upf.edu. Earlier project text calling this code AGPL was wrong (the AGPLv3 statement belongs to the dataset-construction code); corrected in `REPRODUCE.md`, `src/96_clique_proxy.py` and `repro/RUNS.md` (WORKLOG 2026-10-30 entry keeps the old wording).

**Both replies agree (reported).**
- No audio distributed; routes are (a) features by request from MTG, (b) download YouTube audio ourselves from the IDs. Scale: Discogs-VI 1,911,611 versions / 348,796 cliques; Discogs-VI-YT 493,049 versions / 98,785 cliques; Zenodo package main.zip about 1.4 GB compressed, 21 GB uncompressed (ChatGPT).
- The authors relied on the EU DSM Directive Art. 3 (scientific TDM exception). Japan Copyright Act Art. 30-4 (non-enjoyment analysis) is the relevant national rule for training; publishing generated output is a separate question.
- Do not redistribute source audio; release IDs, code, numbers.
- Same-composition grouping: Discogs-VI cliques as backbone, MusicBrainz Work/Recording/Release Group and SecondHandSongs work ids as independent checks; reissue filter via Discogs descriptors (Reissue, Remastered, Repress) and MusicBrainz recording identity; audio version-identification embedding only as a supporting filter.

**New and important (reported by ChatGPT, consistent with the primary pages).**
- Release country is a property of a release, not of a recording: the same recording can appear on US and Spanish releases. A cross-market pair needs distinct recordings linked to one work, with release metadata attached per recording (interpretation: this is the pair definition the project needs; the guide's "exclude reissues" point is the same issue).
- YouTube's terms prohibit downloading except where authorized; the authors' legal basis (EU TDM exception) is separate from YouTube's contract terms. Interpretation: "the authors did it" does not authorize our own crawl; asking MTG for features avoids the question.
- No verified system does "this commercial song rendered as year Y and country C"; closest lines: MusicGen-Melody (melody + text), JASCO (ISMIR 2024), Demerlé et al. style transfer (ISMIR 2024), Stable-Audio-Control (arXiv 2024), Instruct-MusicGen (ISMIR 2025), MuLaCover (arXiv 2026), FreeSVC (ICASSP 2025). Absence of a system is a claim of the reply, not checked.
- Era/market prediction literature: He et al. (arXiv 2024; 54% within +-3 years on MSD), Tardieu et al. (DAFx 2011, production features carry decade), Kostrzewa and Grabczynski (ICCS 2024, 60.03% on 44 country classes, origin not release market). No standard benchmark for release market from audio.
- Evaluation design: 2x2 counterfactual grid (original/target era x original/target market) with main effects, interaction and leakage (change of the non-target probe); control encoder family A, evaluate with families B and C, song identity with an unrelated VI model; no artist, composition or master family crossing train/evaluation.

**Doubtful in the Gemini reply (do not use without checking).** Discogs-VINet code "AGPLv3" (contradicted: MIT); "CLEWS (Discogs-VI-SIREN)" as an acoustic fingerprint tool; fixed CoverHunter similarity thresholds (>0.95 discard, 0.60-0.90 keep) presented as method; DALI country field (ChatGPT could not verify it); citations of unrelated papers for "circular evaluation" (GUI-agent and world-model surveys); the assumption that the team sits in Yokohama (legal analysis built on it). Its dataset list (Music4All-Onion, DALI, MusicBrainz canonical metadata) and the MusicBrainz-based pairing idea are plausible and overlap with ChatGPT.
**Doubtful in the ChatGPT reply.** Some entries carry placeholder citation markers instead of URLs; each cited paper must be opened before it enters the paper's reference list.

**Ranked routes (merged; interpretation).**
1. Discogs-VI metadata (download now) + MTG features by request + MusicBrainz/SHS validation for gold pairs. First steps: fetch `main.zip` from Zenodo 13983028 (CC BY-NC-SA, metadata only), compute metadata-only pair statistics (distinct versions in one clique, different decade and/or release country, reissue descriptors excluded); request the features from MTG.
2. MusicBrainz Work graph with legally available audio (Creative Commons catalogs such as FMA, MTG-Jamendo) for a licensed-audio control experiment.
3. Da-TACOS / SHS-100K cover graph enriched with MusicBrainz release date and country.
Open: written confirmation from the university on publishing generated clips of commercial compositions (the project publishes accompaniment-only demos so far).
