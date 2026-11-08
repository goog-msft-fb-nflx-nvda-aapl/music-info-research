#!/bin/bash
# combined era x language dose grid. Accompaniment: one-hot+16 adapter (HPCP chroma) at era doses a0, a0.3, a0.5, a0.7, a1 (two generation halves on two GPUs), then 155_two_dose_grid.py with the language-dose vocals. usage: bash src/run_grid5.sh <gpuA> <gpuB> (from ~/era_market)
cd ~/era_market
HPCP_CHROMA=1 SUF=_hpcp_gridA bash src/run_ax.sh $1 ax_onehot_pfx16 A_eraz_onehot_s0 10 a0 a0.3 a0.5 &
HPCP_CHROMA=1 SUF=_hpcp_gridB bash src/run_ax.sh $2 ax_onehot_pfx16 A_eraz_onehot_s0 10 a0.7 a1 &
wait
CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/155_two_dose_grid.py > logs/two_dose_grid.log 2>&1
touch grid5.done
