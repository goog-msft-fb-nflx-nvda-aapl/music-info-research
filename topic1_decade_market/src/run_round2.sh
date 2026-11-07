#!/bin/bash
# round 2026-11-07b: N-8b real-audio steering vectors (decade mean minus overall mean, neutral prompt), per-layer calibrated strength beta_l = c * mean hidden norm_l along the unit vector, layers 0-47, c in 0.02/0.05/0.1/0.2
cd ~/era_market; CUDA_VISIBLE_DEVICES=0 ./venv/bin/python src/147_steer_vectors_real.py --name real64 --n 64 > logs/steer_vec_real64.log 2>&1
i=0; for c in 0.02 0.05 0.1 0.2; do CALIB=1 BETA=$c LAYERS=0-47 bash src/run_steer.sh $i real64 rc$c & i=$((i+1)); done; wait; touch round2b.done
