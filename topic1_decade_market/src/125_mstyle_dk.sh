#!/bin/bash
# T-4o: after 113 recomputed the style tables with top_kl: generate the dk_ prompts (3 workers) and read out with 115. usage: bash 125_mstyle_dk.sh <gpuA> <gpuB> <gpuC>   (from ~/era_market)
cd ~/era_market; while [ ! -f market_styles2.done ]; do sleep 30; done
for k in "$1 0" "$2 8" "$3 16"; do set -- $k; GENDIR=gen_mstyle CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/114_market_style_gen.py --seeds 0 1 --rot $2 > logs/mstyle_dk_$2.log 2>&1 & done; wait
GENDIR=gen_mstyle ./venv_ess/bin/python src/115_market_style_eval.py > logs/mstyle_eval2.log 2>&1; touch mstyle_dk.done
