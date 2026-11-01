# Reproduction guide (living document; finalized when the experiments are frozen)

Every number in the report, page and paper comes from a file in `results/` written by a script in `src/`; the worklog (`WORKLOG.md`) records, per run, the configuration, the data and the measured result. This guide gives the order of the steps and the exact commands. Scripts are run from the working directory `era_market/` (a copy of this folder's `src/` plus the data, embeddings, stems and generated audio, which are not in the repository because the source clips are not redistributable and the generated audio is large).

## 0. Environment
- Python environments and package versions: `repro/environment.txt` (main env Python 3.11; separate envs for Audiobox Aesthetics, the Essentia tagger, ACE-Step, MuseControlLite / Stable Audio Open adapters, av-benchmark (commit `f351b9a6fc6abde746d5f8e1d4c47c883319cb41`), Discogs-VINet (third-party AGPL code, kept outside the repository), Whisper, LEACE (`concept-erasure` 0.2.4)).
- One CUDA device for stems, embeddings and generation (up to about 35 GB memory per generation process; one model load per process); the analysis tables need CPU only.
- Hugging Face access for gated weights (token in a mode-600 file, never in the repository).
- Determinism: MusicGen generation with `torch.manual_seed(seed)` is bit-identical across runs and devices in our environment (WORKLOG, R03).

## 1. Data (not in the repository)
Two public manifests (dataset A: six release decades; dataset B: six release markets in the 1980s), 30 s mono 24 kHz clips of Discogs-VI recordings; the manifests hide artist, title, Discogs id and YouTube id. Place them under `era_market/data/ex/dataset_{A,B}/` (`manifest.csv`, `audio/`). `src/common.py` reads the manifests and defines labels and splits. Test labels are hidden; all accuracies are on the validation split.

## 2. Analysis pipeline (stems, probes, controls)
Order, scripts and usage lines are in `repro/RUNS.md` (generated from the script headers by `src/90_make_runs_index.py`):
1. Stems: `src/run_stems.sh` (`01_separate.py`, HT Demucs); second separator `src/run_bsr.sh`.
2. Embeddings: `02_embed.py` (MERT-v2-30s, conditions mix/vocals/accompaniment raw and -23 LUFS normalized), `02b_embed_muq.py` (MuQ), `src/run_encoders.sh` (other encoders), `src/run_derived.sh` (HPSS, derived stems).
3. Probes: `04_probe.py` (layer and C by 5-fold CV on train, scored once on validation), `src/run_analysis.sh`; shortcut baselines `03_stats.py`, `05_langid.py`, `06_extra_probes.py`; Holm correction `35_holm.py`; ablations `45_probe_models.py`, `62_pooling_embed.py` / `62b_pooling_probe.py`, `63_probe_sensitivity.py`.
4. Interventions: stem swap `50_shuffle_remix.py`; LEACE language `51_leace.py`; style erasure `58_style_erasure.py`; cue ablations `57_cue_ablation.py`, `57b_cue_retrain.py`; gap analyses `49a_tags_real.py`, `49b_gap_analyses.py`; controlled tasks `95_controlled_tasks.py`; split audit `96_clique_proxy.py`.
5. Codec-matched era probes: `src/run_codec.sh` (`12_codec_resynth.py`, then embeddings and probes `probe_codec_{mert,muq}_A.json`).

## 3. Generation
1. Standard generation: `07_generate.py` (MusicGen-Melody, chroma of the original mix, era prompts of `src/prompts_era.py`, seeds 0-2, 24 validation songs listed in `gen/picked.txt`); remix with the original vocal stem.
2. Era adapter (text-decade LoRA): `python src/48_lora_musicgen.py --name v1 --steps 3000 --bs 8 --lr 1e-4 --r 16 --pdrop 0.1 --seed 0` (seeds 1 and 2: `run_lora_seed.sh`); generation with the adapter `src/run_lora_gen.sh`.
3. Three-branch guidance: `src/61_triple_guidance.py` (`LORA=v1 NEUTRAL_SRC=gen_lora python src/61_triple_guidance.py --gm 3 --ge 10 --outdir gen_tri_lora_m3e10 --seeds 0 1 2`); drivers `src/run_tri.sh`, `run_tri_ad.sh`, `run_tri_x.sh`, `run_tri_x2.sh` (extra decades, seed adapters, ge 15 and 20), `run_tri_p90.sh`, `run_tri_p00.sh`, `run_tri_pbw.sh` (prompt wording, bandwidth wording).
4. Other generators: MuseControlLite `27_musecontrol.py`, SongEcho adapters `100_songecho_sao.py` via `src/run_sao.sh`, ACE-Step cover `55_ace_cover.py` via `run_ace.sh`.

## 4. Era-axis method (this work)
1. M-1 era encoder (frozen MERT layer-22 / MuQ layer-9 features of codec-resynthesized accompaniment): `python src/97_era_encoder.py mert leace` (nuisance concept removed by LEACE from the input), `... mert adv` (gradient-reversal heads, lambda 0 / .1 / .3), `CLS=1 python src/97_era_encoder.py mert leace` (six-way head); outputs `results/era_encoder_*.json` and `emb/A_eraz_<enc>_<mode><lambda>_s<seed>.npy`. Nuisance vector N: 10 PCs of Discogs-EffNet style tags plus 9 accompaniment signal statistics (`emb/A_dgtags.npy`, `emb/A_stats_acc.npy`).
2. M-2 z-conditioned adapter: `python src/98_era_axis_lora.py --z A_eraz_mert_leace0.0_s0 --name ax_ctrl --steps 3000 --bs 8 --lr 1e-4 --pdrop 0.1` (`--aug` for N-1 with `src/101_make_aug_codes.py`); generation, probe evaluation and content metrics in one driver: `bash src/run_ax.sh <gpu> ax_ctrl A_eraz_mert_leace0.0_s0 10` (era weight 10; `SUF=_dose` and eras `a0 a0.25 a0.5 a0.75 a1 neutral` for the dose-response). Variants: `ax_free` (z from `adv0.0`), `ax_ctrl_s1/s2` (seeds 1, 2), `ax_ctrlc` / `ax_freec` (six-way head), `ax_onehot`, `ax_ordlabel` (label-derived z), `ax_ctrl_aug` (bandwidth-randomized targets).
3. Evaluation: `10_eval_gen.py` (codec-matched MERT/MuQ probes), `13_content_metrics.py` (song identification, chords, tempo), `54_vinet_identity.py`, `36_discogs_tagger.py`, `18_quality.py`, `20_kad.py`, `run_avb.sh` + `83_fd_bootstrap.py` (FD-VGG), `run_avb_pair.sh` + `85_pair_metrics.py`, `94_nuisance_steering.py` (steering after LEACE erasure of style and bandwidth, random same-rank control), `78_six_decades.py` (six-decade tables; `OUT_PREFIX=` sets the output name).
4. Tables from stored results (no GPU): `bash src/99_make_all_tables.sh`.

## 5. Experiment registry (ID -> where)
IDs and their questions: `MASTER_PLAN.md` (sections 3-5). Per run: configuration and measured result in `WORKLOG.md` under the date and title given in the registry; result files under `results/`; launch commands in `src/run_*.sh` (each script header states its usage).

## 6. Seeds and statistics
Generation seeds 0, 1, 2 for all sets; adapter training seeds 0-2 where stated; 24 validation songs; per-song statistics with 95% song-bootstrap intervals (2000 resamples, `numpy` generator seed 0); exact McNemar tests with Holm correction for the stem probes; Wilcoxon signed-rank for paired per-song comparisons (no multiplicity correction unless stated).
