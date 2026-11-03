#!/bin/bash
# T-4f: syllable-matched translations through the lyric route (cover mode 0.95 + stem swap + market probe + read-outs); same 77 clips as T-4k. usage: bash 133_syllable.sh <gpu>   (from ~/era_market)
cd ~/era_market; g=$1; export TEXT=lyrics_scale_text_syl.json
CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/133_syllable_translate.py > logs/syl_text.log 2>&1
(cd ext/ACE-Step-1.5; for k in 0 1; do CUDA_VISIBLE_DEVICES=$g ~/era_market/venv_ace/bin/python ~/era_market/src/111_ace_lyric_route.py --strength 0.95 --outdir ace_syl --shard $k/2 > ~/era_market/logs/ace_syl_$k.log 2>&1 & done; wait)
SRC=ace_syl DST=ace_syl_swap CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/119_stem_swap_route.py > logs/syl_swap.log 2>&1
VOCDIR=ace_syl_swap_voc OUTNAME=syl_market_probe CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/120_swap_market_probe.py > logs/syl_probe.log 2>&1
GEN=ace_syl_swap CUDA_VISIBLE_DEVICES=$g ./venv/bin/python src/112_lyric_route_eval.py > logs/syl_eval.log 2>&1
touch syllable.done
