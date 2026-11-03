#!/bin/bash
# N-5a best-of-N pool: extra generation seeds 3-7 for the ctrlc era-axis adapter (ge 10), same directory as the seed 0-2 clips. usage: run_bon_gen.sh <gpu> <eras...>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; g=$1; shift
./venv/bin/python src/99b_era_axis_gen.py --name ax_ctrlc --z A_eraz_mert_leacec0.0_s0 --gm 3 --ge 10 --outdir gen_ax_ctrlc_ge10 --seeds 3 4 5 6 7 --eras "$@" > logs/bon_gen_$g.log 2>&1
touch bon_gen_$g.done
