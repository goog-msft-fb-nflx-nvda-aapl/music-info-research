#!/bin/bash
# T-4p: descriptor selection on TRAIN clips, then dt_ generation on the 24 evaluation clips and the tag read-out. usage: bash 130_dsel.sh <gpuA> <gpuB> <gpuC>   (from ~/era_market)
cd ~/era_market; A=$1; B=$2; C=$3
for k in "$A 0" "$B 150" "$C 300"; do set -- $k; GENDIR=gen_dsel CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/128_dsel_gen.py --rot $2 > logs/dsel_gen_$2.log 2>&1 & done; wait
GENDIR=gen_dsel ./venv_ess/bin/python src/129_dsel_pick.py > logs/dsel_pick.log 2>&1
for k in "$A 0" "$B 8" "$C 16"; do set -- $k; GENDIR=gen_mstyle CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/114_market_style_gen.py --seeds 0 1 --rot $2 > logs/mstyle_dt_$2.log 2>&1 & done; wait
GENDIR=gen_mstyle ./venv_ess/bin/python src/115_market_style_eval.py > logs/mstyle_eval3.log 2>&1; touch dsel.done
