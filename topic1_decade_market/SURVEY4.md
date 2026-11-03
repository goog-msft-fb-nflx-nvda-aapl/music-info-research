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
