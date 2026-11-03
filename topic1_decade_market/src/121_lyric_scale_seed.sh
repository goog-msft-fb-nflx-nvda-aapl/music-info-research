#!/bin/bash
# T-4k replication with a second generation seed: ACE-Step cover mode seed 1 (same text file), stem swap, vocal market probe (plain and language-erased), read-outs. usage: bash 121_lyric_scale_seed.sh <gpu> <seed>   (from ~/era_market)
cd ~/era_market; g=$1; sd=$2; export TEXT=lyrics_scale_text.json SEED=$sd
(cd ext/ACE-Step-1.5; for k in 0 1; do CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --seeds $sd --outdir ace_scale_seed$sd --shard $k/2 > ~/era_market/logs/ace_scale_seed${sd}_$k.log 2>&1 & done; wait)
SRC=ace_scale_seed$sd DST=ace_scale_swap_seed$sd CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/119_stem_swap_route.py > logs/scale_swap_seed$sd.log 2>&1
VOCDIR=ace_scale_swap_seed${sd}_voc OUTNAME=scale_market_probe_seed$sd CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/120_swap_market_probe.py > logs/scale_probe_seed$sd.log 2>&1
LEACE=1 VOCDIR=ace_scale_swap_seed${sd}_voc OUTNAME=scale_market_probe_seed$sd CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/120_swap_market_probe.py > logs/scale_probe_leace_seed$sd.log 2>&1
GEN=ace_scale_swap_seed$sd CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/112_lyric_route_eval.py > logs/scale_eval_seed$sd.log 2>&1
touch lyric_scale_seed$sd.done
