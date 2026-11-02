#!/bin/bash
# N-5b: probe-guided SMC with the ctrlc era-axis adapter (ge 10), N particles, then probe evaluation and content metrics. usage: run_smc.sh <gpu> <beta> [N=8] [FIRST=0: score only stages at >= FIRST frames, tag gets _f<FIRST>]   (neutral clips are copied from gen_ax_ctrlc_ge10, seeds 0-2)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; b=$2; N=${3:-8}; F=${4:-0}; tag=smc_ctrlc_N${N}_b$b; [ "$F" != 0 ] && tag=${tag}_f$F; n=gen_$tag
./venv/bin/python src/104_smc_gen.py --name ax_ctrlc --z A_eraz_mert_leacec0.0_s0 --gm 3 --ge 10 --N $N --seg 100 --beta $b --first $F --outdir $n --seeds 0 1 2 > logs/$tag.log 2>&1
for f in gen_ax_ctrlc_ge10/*_neutral_s[012]_*.wav; do cp -n $f $n/; done
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${tag}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${tag}_eval.log 2>&1
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${tag}_content.json > logs/${tag}_content.log 2>&1
touch $tag.done
