#!/bin/bash
# N-5a extension: best-of-N pool to 32 seeds. Extra generation seeds 16-31 for the ctrlc era-axis adapter (ge 10), same directory as seeds 0-15. usage: run_bon16_gen.sh <gpu> <eras...>   (GPU 0: 1960s 1970s; 1: 1980s 1990s; 2: 2000s 2010s; 3: neutral)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; g=$1; shift
./venv/bin/python src/99b_era_axis_gen.py --name ax_ctrlc --z A_eraz_mert_leacec0.0_s0 --gm 3 --ge 10 --outdir gen_ax_ctrlc_ge10 --seeds $(seq 16 31) --eras "$@" > logs/bon32_gen_$g.log 2>&1
touch bon32_gen_$g.done
