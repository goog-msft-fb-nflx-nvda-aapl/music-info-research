#!/bin/bash
# T-4k: scale-up of the lyric route to all 102 dataset-B validation clips (targets en, es, pt, de, it except the source language): text (Whisper + M2M-100), ACE-Step cover mode strength 0.95 on two GPUs, stem swap, vocal market probe, language/identity read-outs. usage: bash 121_lyric_scale.sh <gpuA> <gpuB>   (from ~/era_market)
cd ~/era_market; A=$1; B=$2; export TEXT=lyrics_scale_text.json
CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/121_lyric_scale_text.py > logs/lyric_scale_text.log 2>&1
(cd ext/ACE-Step-1.5; for k in 0 1; do g=$([ $k = 0 ] && echo $A || echo $B); CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --outdir ace_scale --shard $k/2 > ~/era_market/logs/ace_scale_$k.log 2>&1 & done; wait)
SRC=ace_scale DST=ace_scale_swap CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/119_stem_swap_route.py > logs/scale_swap.log 2>&1
VOCDIR=ace_scale_swap_voc OUTNAME=scale_market_probe CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/120_swap_market_probe.py > logs/scale_probe.log 2>&1
GEN=ace_scale_swap CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/112_lyric_route_eval.py > logs/scale_eval.log 2>&1
touch lyric_scale.done
