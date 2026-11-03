# Generated demos

Nine 10 s clips from the valence/arousal-conditioned Stable Audio Open 1.0 LoRA (run A8k_s2, 8,000 steps, CFG 5, first sample of each cell of the 3x3 target grid; peak-normalised mp3). File name = target valence and arousal on the 0-1 scale of the training labels (`gen_v<valence>_a<arousal>.mp3`).
Training audio: PMEmo chorus clips (not redistributed). Quantitative evaluation only (MERT-ridge evaluator; see `WORKLOG.md`); no listening test.
Reproduce: `scripts/sao_va.py train A8k_s2 <gpu> true 8000 2` then `scripts/sao_va.py gen A8k_s2 <gpu> 12 50 5.0`; the clips are the first sample of each target cell (generated with seed 1000 + batch offset).
