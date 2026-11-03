#!/bin/bash
# N-5a extension: after run_bon32_gen.sh on GPU 0-3: probe evaluation and content metrics of the 32-seed pool, then the selection analysis (N = 1 ... 32). usage: run_bon32_eval.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; n=gen_ax_ctrlc_ge10; t=ax_ctrlc_bon32
until [ -f bon32_gen_0.done ] && [ -f bon32_gen_1.done ] && [ -f bon32_gen_2.done ] && [ -f bon32_gen_3.done ]; do sleep 60; done
S32=$(seq -s, 0 31)
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${t}_eval.json --train_cond acccodec_norm --seeds ${S32//,/ } --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${t}_eval.log 2>&1
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s SEEDS=$S32 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${t}_content.json > logs/${t}_content.log 2>&1
CUDA_VISIBLE_DEVICES= ./venv/bin/python src/103_best_of_n.py $t 32 > logs/${t}_bon.log 2>&1
touch ${t}.done
