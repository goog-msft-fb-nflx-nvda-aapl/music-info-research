# Open Melody-Conditioned Accompaniment / Cover Models (2024–2026)

*Compiled 1 November 2026. Each entry was checked on the official GitHub, Hugging Face or project page unless marked otherwise.*

**⚠ = not verified at source.**

## Short version

Six systems have runnable open weights: JASCO, MuseControlLite, two variants of SongEcho, MPEcho and MuLaCover. Music ControlNet and the Tencent Stable Audio ControlNet paper have no checkpoint I could find.

None of them was trained or evaluated on decade or era prompts. Some accept free-text era words and show them in demos, and that is all I could confirm.

---

## 1. JASCO (Meta, 2024)

- **Checkpoint:** yes. There are four checkpoints. The two "melody" ones (`facebook/jasco-chords-drums-melody-400M` and `-1B`) are the ones that matter here, and each generates 10-second samples. The HF page is gated: you must agree to share your contact information to download.
  - <https://huggingface.co/facebook/jasco-chords-drums-melody-1B>
- **Conditioning:** text, plus chords given as symbolic `(chord, start_time)` pairs, plus optional drum audio, plus a precomputed melody salience matrix.
  - The API call is `generate_music(descriptions, chords, drums_wav, melody_salience_matrix)`.
  - You extract the melody matrix from your source song yourself; you don't pass in the audio.
- **Max duration:** 10 s.
- **Era prompt:** free text, not evaluated. A community demo uses "80s pop with groovy synth bass" as an example.
- ⚠ **License:** believed to be CC-BY-NC 4.0, as with other AudioCraft weights. It is not stated on the pages I read.
- ⚠ **VRAM:** not stated on the pages I read.

## 2. MuseControlLite (NTU, ICML 2025)

- **Checkpoint:** yes. The HF repo bundles melody-condition checkpoints for SongEcho_base, SongEcho_large and MuseControlLite. The license is MIT.
  - <https://huggingface.co/fundwotsai2001/Text-to-Music_control_family>
  - <https://github.com/fundwotsai2001/MuseControlLite>
- **Base model:** a fine-tuning method built on `stabilityai/stable-audio-open-1.0`. You therefore also need the gated SAO weights and an HF login.
  - ⚠ I did not re-check SAO's own license (Stability AI Community License) in this pass.
- **Conditioning:** text plus a melody extracted from audio.
  - The melody is a top-4, 128-pitch-class CQT.
  - The v3 checkpoint processes the melody of the two stereo channels separately.
  - Dynamics, rhythm, audio inpainting and outpainting are also supported.
  - The audio-reference adapters have not been released yet.
- **Max duration:** about 47 s. The demo Space asks for 47.5 s or longer input audio.
- **Inference:** 11.44 GB VRAM in fp16, 15.9 GB in fp32, measured on an A6000.
- **Vocal-accompaniment variant:** the project page shows singing-accompaniment generation (vocal melody, rhythm, key and structure in; backing out), but says code will be released soon.
  - ⚠ I found no released checkpoint for it.
- **Era prompt:** free text. A demo prompt reads "A lively 70s style British pop song…". Era adherence was not evaluated.

## 3a. SongEcho, official full-song version (ICLR 2026)

- **Checkpoint:** yes. It is available on Google Drive and at `huggingface.co/lsfhuihuiff/SongEcho`. The GitHub repo carries an Apache-2.0 badge.
  - <https://github.com/lsfhuihuiff/SongEcho_ICLR2026>
  - ⚠ The HF model page returned an error when I fetched it, so I could not check the weights' own license or files.
- **What it does:** it generates new vocals and accompaniment together, conditioned on the original vocal melody and text prompts. It is built on ACE-Step.
- **Conditioning:** vocal F0 extracted with RMVPE at 100 Hz, plus tag-style text and lyrics.
  - Practical catch: the shipped `inference.py` runs over a prepared dataset folder, not a single song.
- ⚠ **Max duration and VRAM:** not stated. The output is full-song length, inherited from ACE-Step.
- **Era prompt:** tags such as "90s" can be passed as text, but this was not evaluated.

## 3b. SongEcho_base and SongEcho_large on Stable Audio Open

- These are instrumental adapters in the same HF repo as MuseControlLite, MIT-licensed.
- They use the same melody input and have the same roughly 47 s limit.
- **VRAM in fp16:** 11.15 GB for base, 11.83 GB for large.

## 4. MPEcho (NTU and Yating, ISMIR 2026)

- **Checkpoint:** yes.
  - <https://huggingface.co/Lonian/MPEcho>
  - <https://github.com/YatingMusic/MPEcho> (code is MIT)
  - ⚠ The HF license field just says "cc" without naming a variant, so the exact weight license is unclear.
- **What it adds:** a phoneme encoder and a length regulator on top of the SongEcho framework, which lowers the phoneme error rate. It also generates vocals and accompaniment.
- **Conditioning:**
  - RMVPE F0
  - phoneme timings from their Phonsa aligner
  - lyrics, transcribed with SongPrep
  - text tags
- ⚠ **Max duration and VRAM:** not documented. The README gives only `inference_all.py --ckpt_path`.
- **Era prompt:** the demo uses prompts such as "fuzzy guitar, moody, heavy, 90s style, grunge" and "reminiscent of 80s dance music". This is the closest thing to evidence of era prompting I found, but it was not evaluated.

## 5. MuLaCover (HeartMuLa / MuLa Labs, released 16 September 2026)

- **Checkpoint:** yes, about 4B parameters, and no access request is required.
  - <https://huggingface.co/HeartMuLa/MuLaCover>
  - <https://github.com/HeartMuLa/MuLaCover>
- **License:**
  - Code: Apache-2.0.
  - Weights: CC BY-NC 4.0 plus the MODEL_LICENSE terms.
  - Outputs: noncommercial without written authorization.
- **Conditioning:** one of two inputs, plus lyrics and style.
  - Reference audio, which is transcribed into melody, harmony and optional drum conditions rather than copied frame by frame.
  - Or melody and chord MIDI directly.
  - Lyrics with section markers.
  - One structured style line, for example `genre:[country]; instrument:[...]; mood:[...]`.
- **Extra models needed:**
  - HeartCodec
  - Qwen3-Embedding-0.6B
  - YourMT3 and ChordNet, for audio input only
- **Inference:** Linux with an NVIDIA GPU and Python 3.10. The only verified setup is a B300 GPU.
  - ⚠ Minimum VRAM and maximum length are not stated; it is presented as a full-song model.
- **Era prompt:** the tag schema has no era field. You would put "80s synth-pop" inside `genre:[...]`, but whether that works is untested. ⚠

## 6. No public checkpoint found

- **Music ControlNet** (Adobe/CMU, TASLP 2024).
  - Hugging Face lists zero models linked to the paper, and users there are asking whether it will be open-sourced.
  - Its melody input is a 12-pitch-class chromagram, and it can also take dynamics and rhythm controls.
  - <https://huggingface.co/papers/2311.07069>
- **Stable Audio Open ControlNet** (Hou et al., Tencent ARC, ICASSP 2025).
  - The project page links only to the paper.
  - It conditions on text plus a top-4 CQT melody, with a 47 s maximum.
  - Its demos include a "70s style British pop" prompt.
  - <https://stable-audio-control.github.io>
- ⚠ **"Ciranni et al., 2025"**, which SongEcho cites as a melody-control method, is a separate SAO ControlNet-style work. I did not locate its repo, so whether it has a checkpoint is unverified.

---

## Summary

| System | Open weights | Melody input | Output | Max length | Weight license |
|---|---|---|---|---|---|
| JASCO melody 400M/1B | Yes (gated) | Salience matrix + chords (+drums) | Instrumental | 10 s | ⚠ (likely CC-BY-NC) |
| MuseControlLite | Yes | Top-4 CQT from audio | Instrumental | ~47 s | MIT (+SAO terms ⚠) |
| SongEcho (SAO adapters) | Yes | Top-4 CQT from audio | Instrumental | ~47 s | MIT (+SAO terms ⚠) |
| SongEcho (ACE-Step) | Yes | Vocal F0 + lyrics + tags | Vocals + accompaniment | Full song ⚠ | ⚠ (repo Apache-2.0) |
| MPEcho | Yes | F0 + phonemes + lyrics + tags | Vocals + accompaniment | ⚠ | "cc" ⚠ |
| MuLaCover | Yes | Audio→melody/chord/drums, or MIDI | Vocals + accompaniment | Full song ⚠ | CC BY-NC 4.0 |
| Music ControlNet | No | Chroma, dynamics, rhythm | — | — | — |
| SAO ControlNet (Hou) | None found | Top-4 CQT | — | 47 s | — |

## Era / decade prompts

- Every system takes free text or tags, so you can write "1970s" or "80s".
- None reports training on era labels or measuring how well the output matches the requested era, so treat it as best-effort.
- The demos that show era prompts are:
  - SAO ControlNet and MuseControlLite: 70s
  - MPEcho: 80s and 90s

## What I left out

I did not check these, which may also be relevant:

- ACE-Step's own edit/repaint mode
- LiLAC (Sony)
- Diff-A-Riff
- the July 2026 full-song framework with a cover mode (arXiv 2607.20253)

---

## Sources

- JASCO model card: <https://huggingface.co/facebook/jasco-chords-drums-melody-1B>
- Tonic JASCO demo: <https://huggingface.co/spaces/Tonic/audiocraft>
- MuseControlLite GitHub: <https://github.com/fundwotsai2001/MuseControlLite>
- MuseControlLite project page: <https://musecontrollite.github.io/web/>
- Text-to-Music control family (HF): <https://huggingface.co/fundwotsai2001/Text-to-Music_control_family>
- MuseControlLite Space: <https://huggingface.co/spaces/manoskary/MuseControlLite>
- SongEcho GitHub: <https://github.com/lsfhuihuiff/SongEcho_ICLR2026>
- SongEcho paper: <https://arxiv.org/abs/2602.19976>
- SongEcho summary: <https://www.emergentmind.com/topics/songecho>
- MPEcho project page: <https://lonian6.github.io/MPEcho.github.io/>
- MPEcho GitHub: <https://github.com/YatingMusic/MPEcho>
- MPEcho weights: <https://huggingface.co/Lonian/MPEcho>
- MuLaCover weights: <https://huggingface.co/HeartMuLa/MuLaCover>
- MuLaCover GitHub: <https://github.com/HeartMuLa/MuLaCover>
- Music ControlNet paper page: <https://huggingface.co/papers/2311.07069>
- Stable Audio ControlNet project page: <https://stable-audio-control.github.io>
