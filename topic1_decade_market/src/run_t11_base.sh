#!/bin/bash
# T-11/N-2 held-out read-out of one-hot+16 prefix generations (CPU)
cd ~/era_market; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
for t in ax_ctrlc_s1_ge10 ax_ctrlc_s2_ge10; do
 ./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_$t --seeds 0 1 2 --eras neutral,1960s,1970s,1980s,1990s,2000s,2010s --out results/discogs_decade_$t.json > logs/t11_$t.log 2>&1; touch t11_$t.done
done
