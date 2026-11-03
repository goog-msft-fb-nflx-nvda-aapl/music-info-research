#!/bin/bash
# best-of-N language selection for the lyric route: generate seeds 2 and 3 (121_lyric_scale_seed.sh), dump per-clip market-probe probabilities for seeds 0-3 (plain and language-erased), then 135_bon_language.py. usage: bash 135_bon_language.sh <gpuA> <gpuB>   (from ~/era_market)
cd ~/era_market; A=$1; B=$2
(bash 121_lyric_scale_seed.sh $A 2 &) ; bash 121_lyric_scale_seed.sh $B 3; while [ ! -f lyric_scale_seed2.done ]; do sleep 20; done
export TEXT=lyrics_scale_text.json
for cfg in "0 ace_scale_swap_voc scale_market_probe" "1 ace_scale_swap_seed1_voc scale_market_probe_seed1"; do set -- $cfg; for le in 0 1; do SEED=$1 LEACE=$([ $le = 1 ] && echo 1 || echo 0) VOCDIR=$2 OUTNAME=$3 CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/120_swap_market_probe.py > logs/probe_clips_$1_$le.log 2>&1; done; done
GEN=ace_scale_swap CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/112_lyric_route_eval.py > logs/bonl_eval0.log 2>&1
CUDA_VISIBLE_DEVICES= ./venv/bin/python src/135_bon_language.py > logs/bon_language.log 2>&1
touch bon_language.done
