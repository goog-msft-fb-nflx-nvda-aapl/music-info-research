#!/bin/bash
# M-2/M-3: generation with the era-axis adapter, then probe evaluation and content metrics. usage: [SUF=_dose] run_ax.sh <gpu> <adapter name> <z file> <ge> [eras...]  (eras: neutral, decades, or aX = interpolation between the 1960s and 2010s prototypes)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; name=$2; zf=$3; ge=$4; shift 4; E=${@:-neutral 1960s 1970s 1980s 1990s 2000s 2010s}
n=gen_${name}_ge${ge}${SUF}; tag=${name}_ge${ge}${SUF}; ES=$(echo $E | tr ' ' ',')
./venv/bin/python src/99b_era_axis_gen.py --name $name --z $zf --gm 3 --ge $ge --outdir $n --seeds 0 1 2 --eras $E > logs/$tag.log 2>&1
ERAS=$ES GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${tag}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${tag}_eval.log 2>&1
ERAS=$ES SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${tag}_content.json > logs/${tag}_content.log 2>&1
touch $tag.done
