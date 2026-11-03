# Public checkpoints for melody-preserving music generation and cover-song transformation (2024–2026)

Research cutoff: November 1, 2026 · Primary sources: official model repositories, Hugging Face model cards, inference documentation, and project pages.

## Scope and verification criteria

I distinguish four capabilities that are often conflated in the literature:

1. Melody-conditioned generation: Supply a melody, chromagram, F0 contour, or MIDI sequence together with a text prompt to generate music following that melody.
2. Accompaniment generation: Supply a vocal or instrumental source and generate new backing music, ideally without copying the source recording.
3. Cover-song generation: Preserve the source's melodic identity, potentially including lyrics and vocal phrasing, while changing arrangement, instrumentation, or singer.
4. Audio editing/remixing: Transform or extend an existing recording while retaining some original musical content. This does not necessarily preserve melody.

A checkpoint qualifies as publicly released when an identifiable downloadable weight artifact exists. It qualifies as runnable when compatible inference code and the necessary dependencies are also documented. A paper, demonstration, hosted API, or GitHub repository without weights does not qualify.

The tables distinguish:

- Verified: Supported by an accessible primary source.
- Partial: Weights exist, but the complete workflow, licensing, or required dependencies are not fully established.
- Unverified: The source does not establish the claim.

Crucially, accepting a decade prompt is not the same as demonstrating successful decade-specific style transfer. A model may accept `1970s disco` as text without producing recognizably period-authentic instrumentation, rhythm, production, or harmonic vocabulary.

## 1. JASCO — Meta (2024)

Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation

All four official JASCO variants are released as Hugging Face checkpoint repositories. The two melody variants are the relevant ones for this task; the other two support chords and drums but have no explicit melody input. JASCO generates 10-second music clips, and the checkpoint licenses are CC BY-NC 4.0, separate from the MIT-licensed AudioCraft code.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



| Checkpoint                                                                                                        | Public weights | Melody conditioning | Maximum duration |
| ----------------------------------------------------------------------------------------------------------------- | -------------- | ------------------- | ---------------- |
| [chords-drums-400M](https://huggingface.co/facebook/jasco-chords-drums-400M?utm_source=chatgpt.com)               | Yes, gated     | No; chords + drums  | 10 s             |
| [chords-drums-1B](https://huggingface.co/facebook/jasco-chords-drums-1B?utm_source=chatgpt.com)                   | Yes, gated     | No; chords + drums  | 10 s             |
| [chords-drums-melody-400M](https://huggingface.co/facebook/jasco-chords-drums-melody-400M?utm_source=chatgpt.com) | Yes, gated     | Yes                 | 10 s             |
| [chords-drums-melody-1B](https://huggingface.co/facebook/jasco-chords-drums-melody-1B?utm_source=chatgpt.com)     | Yes, gated     | Yes                 | 10 s             |

Input: Natural-language description, timestamped chord progression, optional drum audio, and (for the melody variants) a melody audio-derived condition. Its local-control pipeline should not be confused with direct full-song audio-to-audio remixing.

Output: Generated music audio, rather than an editable multitrack arrangement.

Inference requirements: AudioCraft, PyTorch, an appropriate GPU, the model's chord mapping and preprocessing dependencies. Model access requires accepting Hugging Face conditions.

Decade/era prompt: Yes, as free-form text, but no verified guarantee of chronological style fidelity.

Assessment: Suitable as a short-form controlled-melody baseline, not a full-song cover generator. Its 10-second generation limit is a major constraint for evaluating historical arrangement styles over complete songs.

## 2. MuseControlLite and Stable Audio Open melody-control adapters (2025–2026)

The official [Text-to-Music control family](https://huggingface.co/fundwotsai2001/Text-to-Music_control_family?utm_source=chatgpt.com) contains three downloadable adapter configurations trained around Stable Audio Open 1.0. They are genuine, documented releases, rather than merely claimed model implementations.

| Released adapter                                                                                                                       | Public checkpoint              | Input / output                                      | Inference VRAM | Duration |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------- | -------------- | -------- |
| [SongEcho_base](https://huggingface.co/fundwotsai2001/Text-to-Music_control_family/tree/main/SongEcho_base?utm_source=chatgpt.com)     | Yes, 31M trainable parameters  | Reference audio-derived melody + text → music audio | 11.15 GB FP16  | \~47.5 s |
| [SongEcho_large](https://huggingface.co/fundwotsai2001/Text-to-Music_control_family/tree/main/SongEcho_large?utm_source=chatgpt.com)   | Yes, 198M trainable parameters | Reference audio-derived melody + text → music audio | 11.83 GB FP16  | \~47.5 s |
| [MuseControlLite](https://huggingface.co/fundwotsai2001/Text-to-Music_control_family/tree/main/MuseControlLite?utm_source=chatgpt.com) | Yes, 85M trainable parameters  | Reference audio-derived melody + text → music audio | 11.44 GB FP16  | \~47.5 s |

License: The adapter collection is marked MIT. The required [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0?utm_source=chatgpt.com) base checkpoint has separate Stability AI Community License terms; the MIT adapter license does not supersede them.

Exact conditioning: The documented inference implementation extracts a melody representation from a reference audio file, applies the corresponding adapter and melody encoder, and combines this condition with a natural-language music prompt. It uses stereo WAV output at the base model's 44.1 kHz sample rate.

Duration: The official example uses `2097152 / 44100`, or approximately 47.54 seconds. This is the documented single-pass duration, not a verified full-song generation limit.

Inference: Python 3.11, PyTorch, Diffusers, Stable Audio Open 1.0, the adapter checkpoint, the matching melody encoder, FFmpeg, and CUDA. The documented FP32 VRAM figures are 15.7 GB, 16.5 GB, and 15.9 GB, respectively.

Era prompt: Yes, arbitrary text can be supplied. Actual era preservation or style accuracy remains unverified.

Important qualification: These adapters primarily implement melodic control, not a guarantee of preserving lyrics, exact vocal F0, or note-by-note phrasing. The MuseControlLite authors also state that the public checkpoints were trained on MTG-Jamendo, predominantly electronic music, and advise retraining for other domains.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+2



### MuseControlLite singing-accompaniment demonstration — separate status

The authors additionally demonstrate vocal melody + rhythm + key + structure → instrumental backing, controlled by prompts such as country, reggae, jazz, and orchestral backing. However, their demonstration page says the corresponding code will be released later, and their repository separately indicates that audio-conditioning adapter weights are not yet available.

Therefore, the public melody-adapter checkpoint above must not be reported as a verified release of the exact singing-accompaniment system. That specialized checkpoint/workflow remains unverified.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://musecontrollite.github.io\&sz=32)

musecontrollite.github.io+1



## 3. SongEcho — original ICLR 2026 release

This is a different checkpoint family from the Stable Audio Open adapters named `SongEcho_base` and `SongEcho_large` in Section 2.

| Field            | Verified finding                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Weights          | [Official SongEcho checkpoint repository](https://huggingface.co/lsfhuihuiff/SongEcho/tree/main?utm_source=chatgpt.com) |
| Runnable?        | Partial — official inference code exists, but input preparation is not a simple source-WAV command                      |
| License          | Unverified — no explicit license for the released weights established                                                   |
| Input            | Extracted vocal F0/melody, source-related preprocessing data, and a text prompt                                         |
| Output           | Newly synthesized vocals and accompaniment                                                                              |
| Maximum duration | Unverified                                                                                                              |
| Decade prompt    | Yes, via natural-language text; era accuracy unverified                                                                 |

The released bundle includes `melody.pt` (the IA-EiLM adapter), `melody_encoder.pt`, and `rmvpe_model.pt` for vocal melody extraction. The implementation uses ACE-Step as its generative foundation.

Inference: The [official code](https://github.com/lsfhuihuiff/SongEcho_ICLR2026?utm_source=chatgpt.com) documents Python/PyTorch with CUDA and a command of the following form:

```
python inference.py \
  --output_dir ./val_results \
  --pt_path ./checkpoints/melody.pt \
  --dataset_path /path/to/val_dataset
```

Notice that `--dataset_path` points to prepared data, not directly to an arbitrary MP3. A source-song preprocessing workflow is necessary.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



## 4. MPEcho — ISMIR 2026

| Field            | Verified finding                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| Weights          | [MPEcho checkpoint](https://huggingface.co/Lonian/MPEcho/tree/main?utm_source=chatgpt.com)              |
| Runnable?        | Partial — downloadable checkpoint and inference script exist; preprocessing is manual                   |
| License          | Code: MIT. Weights: unresolved; Hugging Face only specifies `cc`, without a complete license identifier |
| Input            | Vocal F0/melody + phonemes and their timing/alignment + text/music description                          |
| Output           | Re-synthesized cover with vocals and accompaniment                                                      |
| Maximum duration | Unverified                                                                                              |
| Decade prompt    | Yes, through the underlying text-controlled synthesis pipeline; precise era fidelity unverified         |

MPEcho extends ACE-Step-based cover generation with explicit phonetic conditioning. This is particularly relevant when a system must preserve both melody and recognizable lyrics.

Its official implementation relies on SongPrep for lyrics transcription, Phonsa for phoneme alignment, and RMVPE for melody extraction. Inference is documented as:

```
python inference_all.py \
  --ckpt_path /path/to/50000-steps-all.ckpt \
  --exp_name cover_experiment
```

Inference requirements: Python/PyTorch, an appropriate CUDA environment, ACE-Step components, feature preprocessing, the MPEcho checkpoint, and the associated alignment tools. A reproducible minimum GPU-memory requirement is not specified in the source documentation.

The [official GitHub repository](https://github.com/YatingMusic/MPEcho?utm_source=chatgpt.com) is available, but its README does not establish a complete direct MP3-to-cover workflow or a hard maximum duration.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



## 5. MuLaCover — 2026

Public weights and code

Noncommercial checkpoint

| Field            | Verified finding                                                                                                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Weights          | [HeartMuLa/MuLaCover](https://huggingface.co/HeartMuLa/MuLaCover?utm_source=chatgpt.com)                                   |
| Runnable?        | Yes, with multiple additional checkpoints                                                                                  |
| License          | Code: Apache-2.0. Weights: CC BY-NC 4.0 plus `MODEL_LICENSE`; generated outputs have additional noncommercial restrictions |
| Input            | Reference MP3/WAV or melody/chord MIDI, plus lyrics and structured style tags                                              |
| Output           | Newly generated cover-song WAV with vocals and backing                                                                     |
| Maximum duration | Unverified at the primary source                                                                                           |
| Decade prompt    | Partial — style fields accept genres/instruments/moods; a dedicated decade control is not documented                       |

Its textual conditioning is not an unrestricted chat prompt. The documented format is:

```
topic:[Longing]; genre:[country]; instrument:[Strings,acoustic guitar]; mood:[hopeful]
```

Lyrics are provided separately with section markers such as `[Verse]` and `[Chorus]`.

Inference requirements: Linux, NVIDIA CUDA GPU, Python 3.10, PyTorch (officially tested with 2.10/CUDA 13.0), FFmpeg, the MuLaCover weights, HeartCodec, and Qwen3-Embedding. Source-audio conditioning additionally requires YourMT3 and ChordNet checkpoints to produce symbolic conditions; MIDI-only conditioning can omit the transcription components.

The official [repository](https://github.com/HeartMuLa/MuLaCover?utm_source=chatgpt.com) documents a five-shard checkpoint layout and complete installation instructions. Public release is dated September 16, 2026.

Duration caution: A 240-second limit appears in third-party descriptions, but I could not establish it from the official release. It should not be entered into a reproducibility table as a verified maximum.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+2



## 6. MIDI-SAG / ComposerFlow — singing accompaniment (2026)

This is an important additional release beyond the systems named in the question.

The newer [MIDI-SAG repository](https://github.com/fundwotsai2001/MIDI-SAG?utm_source=chatgpt.com) provides an accompaniment-generation pipeline, despite the older MuseControlLite demonstration page still saying that its code will be released later.

| Field                   | Verified finding                                                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Checkpoint availability | Partial — installation script references downloadable MuseControlLite and auxiliary checkpoints on Google Drive; direct weight accessibility was not independently verified |
| Runnable?               | Yes, documented multicomponent pipeline, subject to obtaining all checkpoints                                                                                               |
| License                 | Pipeline code: Apache-2.0; adapter-specific license unverified; Stable Audio Open and other dependencies retain separate licenses                                           |
| Input                   | Vocal audio alone, or vocal audio + vocal MIDI, with style text                                                                                                             |
| Output                  | Generated backing-track WAV; optionally mixed with supplied/synthesized vocals                                                                                              |
| Duration                | Up to 47 s per segment, or `full_song` mode with section-wise generation; no verified overall maximum                                                                       |
| Era prompt              | Yes, `BACKING_TEXT_PROMPT` or per-section prompts                                                                                                                           |

The source provides three modes: `detected` automatically transcribes vocal melody, `ground-truth` uses an existing vocal MIDI, and `ComposeFlow` starts from lyrics and synthesizes singing before adding accompaniment.

Inference requirements: Python 3.9, PyTorch 2.5.1, CUDA-capable GPU, Stable Audio Open 1.0, MuseControlLite adapter, GAME, AccoMontage2, beat tracking, and other mode-specific dependencies. The source does not specify an overall minimum VRAM requirement. The installation script supplies checkpoint download commands.

A crucial implementation detail: in `full_song` mode, every structural segment must be shorter than 47 seconds. Segment concatenation extends the total duration but introduces possible coherence and transition problems.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



This is the most directly relevant released backing-track-only workflow in the audited set.

## 7. MusiConGen — ISMIR 2024

[Public inference weights](https://huggingface.co/Cyan0731/MusiConGen/tree/main?utm_source=chatgpt.com) · [Official inference code](https://github.com/YatingMusic/MusiConGen?utm_source=chatgpt.com)

| Field            | Verified finding                                                                                               |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| Runnable?        | Yes, with source-code configuration                                                                            |
| License          | Code MIT; checkpoint repository tagged MIT, but inherited MusicGen weight restrictions need separate review    |
| Conditioning     | Text description + symbolic chord progression + rhythm/beat control, optionally extracted from reference audio |
| Output           | Generated instrumental/backing music, 32 kHz audio                                                             |
| Maximum duration | Approximately 30 s in the published workflow; hard limit not independently verified                            |
| Era prompt       | Yes, natural-language genre/style description                                                                  |

The source model is MusicGen-based, with additional conditioning for chord and rhythm structure. It does not guarantee preservation of the exact sung melody merely because the chords and beats match.

Inference requirements: Python 3.9, PyTorch 2.0, FFmpeg, AudioCraft-derived implementation, and a GPU with more than 12 GB VRAM recommended.

The published checkpoint has `state_dict.bin` and `compression_state_dict.bin`. Generation uses `python generate_chord_beat.py` from the repository's `audiocraft` directory.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

YatingMusic/MusiConGen · GitHub+2



## 8. Stable Audio Open ControlNet releases: two distinct projects

| Project / checkpoint                                                                                                   | Runnable open checkpoint?                                                        | Exact conditioning                                              | Duration                                              | License / hardware                            |
| ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------- |
| [Editing Music with Melody and Text (ICASSP 2025)](https://stable-audio-control.github.io/web/?utm_source=chatgpt.com) | No verified public weights or runnable source release                            | Input music converted to top-4 CQT melody representation + text | 47 s                                                  | Weights/license/hardware unverified           |
| [MusDB audio ControlNet](https://github.com/EmilianPostolache/stable-audio-controlnet?utm_source=chatgpt.com)          | Partial — checkpoint linked on Google Drive, download not independently verified | Source audio + text → accompaniment/music audio                 | Unverified; based on \~47 s Stable Audio architecture | Adapter weight license unverified; 16 GB VRAM |
| [MoisesDB audio ControlNet](https://github.com/EmilianPostolache/stable-audio-controlnet?utm_source=chatgpt.com)       | Partial, same limitation                                                         | Source audio + text → accompaniment/music audio                 | Unverified                                            | Adapter weight license unverified; 16 GB VRAM |

The ICASSP 2025 system's published examples explicitly state 47 seconds as the maximum supported length, but its project page does not provide a verified open checkpoint. It is therefore not a runnable public-weight entry.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://stable-audio-control.github.io\&sz=32)

stable-audio-control.github.io



The Emilian Postolache repository is a different, independently released implementation. It references two pretrained audio-conditioned ControlNet models and includes separate inference notebooks:

- `notebook/inference_musdb_audio_large.ipynb`
- `notebook/inference_moisesdb_audio_large.ipynb`

The checkpoint links point to Google Drive folders. I could verify that the authors published the links, but not that their model files are currently obtainable through those links. These should be classified as claimed checkpoint releases, availability unverified, not confidently reproducible baselines. Chroma and envelope examples in the same repository are not evidence of additional separately released checkpoint files.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



The Stable Audio Open base weights use the Stability AI Community License; adapter licensing cannot be inferred automatically from that license.

## 9. ACE-Step — released audio-to-audio and cover checkpoints (2025–2026)

### ACE-Step v1, May 2025

| Field            | Verified finding                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------- |
| Weights          | [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B?utm_source=chatgpt.com) |
| Runnable?        | Yes                                                                                         |
| License          | Apache-2.0                                                                                  |
| Conditioning     | Source music audio in audio-to-audio mode + text style description; optional lyric controls |
| Output           | Re-generated music audio, including vocals                                                  |
| Maximum duration | Full-song, multiminute generation documented; specific audio-to-audio maximum unverified    |
| Decade prompt    | Yes                                                                                         |

Inference: Official [ACE-Step v1 implementation](https://github.com/ace-step/ACE-Step?utm_source=chatgpt.com), CUDA/PyTorch, Python environment and model weights. The developers document audio-to-audio support from May 9, 2025 and a reduced 8 GB VRAM inference configuration from May 10.

Melody retention is implicit through audio-to-audio conditioning, not an externally enforced note-sequence constraint.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



### ACE-Step v1.5, released January 2026

The official release supports a dedicated `cover` task. Its documented maximum general generation duration is 600 seconds (10 minutes), though the source does not separately prove that every checkpoint reliably preserves melody for a 600-second cover.

The following are publicly released model variants:

| Model / verified checkpoint                                                                                 | Weights                       | License    | Cover support                                                              |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------- | -------------------------------------------------------------------------- |
| [v15-base](https://huggingface.co/ACE-Step/acestep-v15-base?utm_source=chatgpt.com)                         | Yes                           | MIT        | Yes                                                                        |
| [v15-sft](https://huggingface.co/ACE-Step/acestep-v15-sft?utm_source=chatgpt.com)                           | Yes                           | MIT        | Yes                                                                        |
| [v15-turbo](https://huggingface.co/ACE-Step/Ace-Step1.5/tree/main/acestep-v15-turbo?utm_source=chatgpt.com) | Yes, unified model repository | MIT family | Yes                                                                        |
| [v15-turbo-continuous](https://huggingface.co/ACE-Step/acestep-v15-turbo-continuous?utm_source=chatgpt.com) | Yes                           | MIT        | Advertised family capability; variant-specific cover validation unverified |
| [v15-turbo-shift1](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift1?utm_source=chatgpt.com)         | Yes                           | MIT        | Advertised family capability; variant-specific validation unverified       |
| [v15-turbo-shift3](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift3?utm_source=chatgpt.com)         | Yes                           | MIT        | Advertised family capability; variant-specific validation unverified       |
| [v15-xl-base](https://huggingface.co/ACE-Step/acestep-v15-xl-base?utm_source=chatgpt.com)                   | Yes                           | MIT        | Yes, family API                                                            |
| [v15-xl-sft](https://huggingface.co/ACE-Step/acestep-v15-xl-sft?utm_source=chatgpt.com)                     | Yes                           | MIT        | Yes, family API                                                            |
| [v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo?utm_source=chatgpt.com)                 | Yes                           | MIT        | Yes, family API                                                            |

All use the same fundamental cover conditioning:

- Input: Source WAV/MP3 or other supported audio + natural-language target-style caption.
- Optional: Lyrics and audio-cover strength.
- Output: Generated music audio, with configurable output file format.
- Duration: 10–600 s documented at the family API level.
- Era control: Yes, an unrestricted text caption can include a decade/style description.
- Inference: [Official repository](https://github.com/ace-step/ACE-Step-1.5?utm_source=chatgpt.com) with compatible PyTorch/CUDA, selected DiT weights, audio codec and optional language-model components. Quantization and CPU offloading allow the smaller family to run in under 4 GB VRAM according to the developers; XL models require more memory.

The source makes an important distinction between `src_audio` (the song to cover) and `reference_audio` (a separate style reference). For cover generation, the official example is:

```
params = GenerationParams(    task_type="cover",    src_audio="original_song.mp3",    caption="1970s disco arrangement, strings, funky bass",    audio_cover_strength=0.8,)
```

This is a genuine documented source-audio + target-text cover task. The text caption can express an era directly. Nevertheless, its melody preservation is probabilistic, and a 10-minute generation limit is not proof of accurate long-form melody matching.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+5



The official repository also publishes Diffusers-format conversions of XL-base, XL-SFT, and XL-turbo. These represent alternative packaging of the relevant models, not three fundamentally different cover-generation architectures.

## 10. YuE2 — symbolic melody-preserving covers (2026)

| Field            | Verified finding                                                                   |
| ---------------- | ---------------------------------------------------------------------------------- |
| Weights          | [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B?utm_source=chatgpt.com)             |
| Runnable?        | Yes, with a separate transcription step                                            |
| License          | Code Apache-2.0; model weights CC BY-NC 4.0                                        |
| Conditioning     | Source song → transcribed melody as ABC score; user-supplied lyrics and text style |
| Output           | Newly arranged and sung 48 kHz stereo music                                        |
| Maximum duration | Unverified; released examples include songs of at least five minutes               |
| Decade prompt    | Yes, unrestricted natural-language style                                           |

The publicly released dependency checkpoints include [SheetSage2](https://huggingface.co/m-a-p/SheetSage2?utm_source=chatgpt.com) for transcription and [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae?utm_source=chatgpt.com) for audio decoding.

For covers, the recommended process is to transcribe a recording, remove unwanted chord constraints from the melody ABC score, and generate with `cot="melody"` so the accompaniment can change independently.

Inference: The [official YuE2 repository](https://github.com/multimodal-art-projection/YuE?utm_source=chatgpt.com) documents Python `YuE2Pipeline`, CUDA inference and a 24 GB GPU without quantization. This is not a single-pass MP3-to-MP3 API: source-audio transcription is an explicit upstream component.

Compared with latent audio-to-audio editing, the symbolic representation makes melody constraints inspectable and potentially easier to evaluate automatically.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+2



## 11. Additional released editing checkpoints

These systems accept source audio and text but are weaker matches for a strict melody-preserving cover requirement.

| System / checkpoint                                                                                                                            | Public and runnable? | License                                    | Exact conditioning → output                                       | Max duration | Decade prompt                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------ | ----------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------- |
| [MelodyFlow 1B (2024)](https://huggingface.co/facebook/melodyflow-t24-30secs?utm_source=chatgpt.com)                                           | Yes                  | CC BY-NC 4.0 weights, MIT code             | Input music audio + target text → edited/generated music          | 30 s         | Yes; exact melody preservation unverified                                         |
| [Instruct-MusicGen reproduction (2024–25)](https://huggingface.co/ldzhangyx/instruct-MusicGen/blob/main/finetuned.ckpt?utm_source=chatgpt.com) | Partial              | Apache-2.0 code; weight license unverified | Input music audio + add/remove/extract instruction → edited music | Unverified   | Free-text instructions possible, but decade-specific rearrangement not documented |

MelodyFlow: Uses latent inversion for text-guided musical editing. Inference is through AudioCraft with the released `facebook/melodyflow-t24-30secs` checkpoint. The published workflow supports 30-second clips; AudioCraft documents a roughly 16 GB GPU recommendation for medium-scale inference. It can change musical style but does not guarantee identical note sequences. The official documentation lists one pretrained MelodyFlow release, despite the research paper discussing both 300M and 1B model sizes.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



Instruct-MusicGen: The checkpoint linked above is explicitly identified by its developer as a retrained reproduction using public datasets, not the original Sony-related research checkpoint. Installation uses Python 3.11.7, PyTorch and the repository requirements; its inference script is `python src/inference.py`, with source-audio and configuration preparation required. Its core operations are instrument/stem manipulation rather than unrestricted cover generation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 12. Systems that must not be counted as verified open checkpoints

| System                                                                                                                    | What is established                                                                                 | Why it is excluded or qualified                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [Music ControlNet](https://musiccontrolnet.github.io/web/?utm_source=chatgpt.com) (2024 publication)                      | Text + melody/rhythm/dynamics-conditioned music; project demonstrations                             | No publicly released checkpoint verified; weight license, runtime, and maximum duration unavailable                                |
| [Stable Audio Open melody ControlNet research](https://stable-audio-control.github.io/web/?utm_source=chatgpt.com) (2025) | 47 s text + top-k CQT melody-control demonstrations                                                 | No open checkpoint verified                                                                                                        |
| [MusicGen-Style](https://huggingface.co/facebook/musicgen-style?utm_source=chatgpt.com) (2024)                            | Released 1.5B CC BY-NC 4.0 checkpoint; 1.5–4.5 s audio style reference + text; AudioCraft inference | Audio-style conditioning is not independently verified melody-preserving source-song conditioning                                  |
| [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0?utm_source=chatgpt.com) (2024)           | Public text-to-audio base model, \~47 s                                                             | No source-melody or cover control on its own; requires an appropriate adapter                                                      |
| [ComposerFlow research demonstration](https://composerflow.github.io/web/?utm_source=chatgpt.com) (2026)                  | Full-song MIDI-informed accompaniment demonstrations                                                | Should be counted through the released MIDI-SAG pipeline above, not as an independently verified second set of proprietary weights |
| [ACE-Step Singing2Accompaniment ControlNet roadmap item](https://github.com/ace-step/ACE-Step?utm_source=chatgpt.com)     | Announced as a planned development                                                                  | A roadmap item is not a verified released checkpoint                                                                               |

For Music ControlNet in particular, the first-party page provides paper, video, and audio examples, but not a verifiable download-and-inference route. It is scientifically relevant but does not satisfy the checkpoint-release requirement.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://musiccontrolnet.github.io\&sz=32)

musiccontrolnet.github.io+3



## 13. Which checkpoints actually satisfy the requested use case?

| Intended experiment                                         | Strongest verified candidates                   | Main unresolved limitation                                                     |
| ----------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------ |
| Full-song source audio → stylistically different sung cover | ACE-Step 1.5, MuLaCover                         | No guarantee that every melody note or lyric syllable is preserved             |
| Explicit melody-preserving full-song cover                  | YuE2, MuLaCover MIDI path                       | Requires reliable transcription; styles may alter melodic phrasing             |
| Preserve vocal pitch and lyrics in a new cover              | SongEcho, MPEcho                                | Preprocessing complexity, incomplete license/duration documentation            |
| Original singing → new instrumental accompaniment           | MIDI-SAG / MuseControlLite pipeline             | Multicomponent setup, unverified direct availability of some auxiliary weights |
| Short controlled instrumental reconstruction                | JASCO melody variants, MuseControlLite adapters | 10 s or \~47 s single-pass limits                                              |
| Chord/rhythm-controlled backing with textual style          | MusiConGen, JASCO chord/drum variants           | Harmony/rhythm preservation is not exact melody preservation                   |
| Editing timbre/style while retaining some source content    | MelodyFlow, Instruct-MusicGen                   | Musical identity preservation not explicitly enforced                          |

### Implications for decade/era experiments

All of the runnable text-conditioned families above can either accept free-form style text or documented genre/instrument tags. None of the inspected primary sources establishes a separately calibrated decade-conditioning mechanism that guarantees 1960s, 1970s, 1980s, or 1990s period authenticity.

For a controlled research comparison, separate three outcomes: melody preservation, target-era recognition, and audio quality. A generated cover that sounds good but changes the tune should not receive a high melody-preservation score; likewise, the presence of analog synthesizers or orchestral strings does not, on its own, establish a correct historical release-market style.

The highest-priority reproducible baselines would be ACE-Step 1.5 for direct audio covers, YuE2 for explicit melodic score control, and MIDI-SAG for vocal-conditioned accompaniment, followed by short-form comparison against JASCO and MuseControlLite.

Verification boundary: This is an extensive primary-source audit of identifiable releases, not a mathematically exhaustive enumeration of every third-party LoRA, quantization, or private/ephemeral checkpoint published between 2024 and 2026. Public weights, source availability, licensing, inference readiness, and guaranteed musical behavior are deliberately treated as separate claims.