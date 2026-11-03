#!/bin/bash
# N-7 pilot: generation with (or without) the era LoRA and the era read-out. usage: bash 137_ace_lora_eval.sh <gpu> <outdir> <lora dir or none> [strength=0.5]   (from ~/era_market)
cd ~/era_market; g=$1; o=$2; l=$3; st=${4:-0.5}; export CUDA_VISIBLE_DEVICES=$g
LA=""; [ "$l" != none ] && LA="--lora $l"
(cd ext/ACE-Step-1.5; ~/era_market/venv_ace/bin/python ~/era_market/src/137_ace_lora_gen.py $LA --strength $st --outdir $o --seeds 0 1 > ~/era_market/logs/$o.log 2>&1)
ERAS=neutral,1960s,1970s,1980s,1990s,2000s,2010s GENDIR=$o ./venv/bin/python src/10_eval_gen.py --out results/${o}_eval.json --train_cond acccodec_norm --seeds 0 1 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${o}_eval.log 2>&1
OUT_PREFIX=results/six_decades_$o python3 src/78_six_decades.py $o=results/${o}_eval.json > logs/${o}_six.log 2>&1
touch $o.done
