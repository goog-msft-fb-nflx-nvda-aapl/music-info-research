#!/bin/bash
# T-4q: word selection with the market probes on TRAIN clips, dp_ generation on the 24 evaluation clips, read-out with the tagger (115) and the probes (37). usage: bash 131_dsel_probe.sh <gpuA> <gpuB> <gpuC>   (from ~/era_market; needs gen_dsel from 130_dsel.sh)
cd ~/era_market; A=$1; B=$2; C=$3
GENDIR=gen_dsel CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/131_dsel_probe_pick.py > logs/dsel_probe_pick.log 2>&1
for k in "$A 0" "$B 8" "$C 16"; do set -- $k; GENDIR=gen_mstyle CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/114_market_style_gen.py --seeds 0 1 --rot $2 > logs/mstyle_dp_$2.log 2>&1 & done; wait
GENDIR=gen_mstyle ./venv_ess/bin/python src/115_market_style_eval.py > logs/mstyle_eval4.log 2>&1
COND_PREFIX=dp_ GENDIR=gen_mstyle CUDA_VISIBLE_DEVICES=$A ./venv/bin/python src/37_eval_market.py --out results/mstyle_probe_dp.json --seeds 0 1 > logs/mstyle_probe_dp.log 2>&1
touch dsel_probe.done
