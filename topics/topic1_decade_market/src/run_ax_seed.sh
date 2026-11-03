#!/bin/bash
# M-2 seed replicate: train the era-axis adapter with adapter seed S on z file with training seed S, then generate and evaluate. usage: run_ax_seed.sh <gpu> <name base, e.g. ax_ctrlc> <z prefix, e.g. A_eraz_mert_leacec0.0> <seed>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; b=$2; z=$3; s=$4
./venv/bin/python src/98_era_axis_lora.py --z ${z}_s$s --name ${b}_s$s --steps 3000 --bs 8 --lr 1e-4 --pdrop 0.1 --seed $s > logs/tr_${b}_s$s.log 2>&1
bash src/run_ax.sh $1 ${b}_s$s ${z}_s$s 10
