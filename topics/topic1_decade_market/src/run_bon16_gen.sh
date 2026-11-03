#!/bin/bash
# N-5a extension: best-of-N pool to 16 seeds. Extra generation seeds 8-15 for the ctrlc era-axis adapter (ge 10), same directory as seeds 0-7. usage: run_bon16_gen.sh <gpu> <eras...>   (e.g. GPU 0: 1960s 1970s 1980s neutral; GPU 1: 1990s 2000s 2010s)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; g=$1; shift
./venv/bin/python src/99b_era_axis_gen.py --name ax_ctrlc --z A_eraz_mert_leacec0.0_s0 --gm 3 --ge 10 --outdir gen_ax_ctrlc_ge10 --seeds 8 9 10 11 12 13 14 15 --eras "$@" > logs/bon16_gen_$g.log 2>&1
touch bon16_gen_$g.done
