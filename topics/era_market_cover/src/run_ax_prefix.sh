#!/bin/bash
# N-3: era-axis adapter with P dedicated prefix tokens (z modifies only those P text positions). usage: run_ax_prefix.sh <gpu> <name> <z file> <P>   (then generation/eval as run_ax.sh, ge 10)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/98_era_axis_lora.py --z $3 --name $2 --prefix $4 --steps 3000 --bs 8 --lr 1e-4 --pdrop 0.1 > logs/tr_$2.log 2>&1
bash src/run_ax.sh $1 $2 $3 10
