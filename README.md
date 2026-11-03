# CommE5070 final project (music generation track)

Course project for CommE5070 (Deep Learning for Music Analysis and Generation, NTU). The repository has one folder per topic of the course project guide (eight topics, numbered as in the guide); two contain work so far, each in its own folder with its own README, code, results, work log and plan. People interested in either topic are welcome to join.

| Guide topic | Folder | Status |
|---|---|---|
| 1. Released decade and market classification, era-targeted covers | [`topics/topic1_decade_market/`](topics/topic1_decade_market/) | Stem-wise analysis of release-decade and release-market cues on Discogs-VI clips, and cover generation that keeps the melody and vocals while steering the era. Main work of this repository. |
| 2. Streaming music accompaniment generation | [`topics/topic2_streaming_accompaniment/`](topics/topic2_streaming_accompaniment/) | not started |
| 3. Inference-time optimization and evaluation for text-to-music | [`topics/topic3_inference_time_ttm/`](topics/topic3_inference_time_ttm/) | Independent, significance-tested evaluation of concept-direction steering in MusicGen. |
| 4. MIDI-REPA and ATTM | [`topics/topic4_midi_repa_attm/`](topics/topic4_midi_repa_attm/) | not started |
| 5. Hit song analysis and generation | [`topics/topic5_hit_song/`](topics/topic5_hit_song/) | not started |
| 6. Music therapy | [`topics/topic6_music_therapy/`](topics/topic6_music_therapy/) | not started |
| 7. AI DJ | [`topics/topic7_ai_dj/`](topics/topic7_ai_dj/) | not started |
| 8. Copyright concerns in music generation | [`topics/topic8_copyright/`](topics/topic8_copyright/) | not started |

## Team
- James Christian: MS in Artificial Intelligence, University of Texas at Austin (jameschristian@utexas.edu)
- Liang-Yu Hsia: MS in Communication Engineering, [NTU](https://gice.ntu.edu.tw/en/) (r13942129@ntu.edu.tw)
- Jefferson Sy Dionisio: Ph.D. in Communication Engineering, NTU (jsydionisio@gmail.com)

## Shared
- `docs/` GitHub Pages source (kept at the repository root because Pages serves from `/docs`): `index.html` for topic 3, `era_market/` for topic 1. Pages URL: https://goog-msft-fb-nflx-nvda-aapl.github.io/music-info-research/
- Topic 3 mid-progress video: https://youtu.be/aaZ89FoZe_U
- `LICENSE` Apache License 2.0.

Each topic folder is self-contained: run its scripts from inside that folder.

## License
Code and text in this repository are released under the Apache License 2.0 (`LICENSE`). Datasets, audio and third-party models used by the experiments keep their own licenses and terms (for example the Discogs-VI metadata CC BY-NC-SA 4.0, course-provided audio, and model weights from their authors); none of that audio is redistributed here except the generated demo clips under `docs/era_market/` and `docs/audio/`, which are model outputs released under the same Apache-2.0 license as the code (the licenses of the generators, for example MusicGen-Melody CC-BY-NC 4.0 weights, still apply to their use). Code under AGPL (Discogs-VINet) is used as an external checkout and is not included.
