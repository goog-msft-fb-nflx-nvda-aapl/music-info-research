#!/bin/bash
# N-7b (3): ACE-Step era LoRA trained with year-augmented captions (era prompt + ", recorded in <year>", year drawn uniformly inside the clip's decade, fixed seed), then generation with the decade's x4 year appended and the era read-out. usage: bash 144_ace_lora_year.sh <gpu> [epochs=3]   (from ~/era_market)
cd ~/era_market; g=$1; ep=${2:-3}; export CUDA_VISIBLE_DEVICES=$g
if [ ! -d ace_lora/data_yr ]; then
  mkdir -p ace_lora/data_yr; cp ace_lora/data/*.wav ace_lora/data/*.lyrics.txt ace_lora/data_yr/
  ./venv/bin/python - <<'P'
import json,glob,os,random,re
random.seed(0)
for f in glob.glob("ace_lora/data/*.json"):
    d=json.load(open(f)); m=re.search(r"a (\d{3})0s", d["caption"]); d["caption"]+=", recorded in %s%d"%(m.group(1),random.randint(0,9)); json.dump(d,open("ace_lora/data_yr/"+os.path.basename(f),"w"))
P
fi
cd ext/ACE-Step-1.5; R=$PWD; PY=~/era_market/venv_ace/bin/python; W=~/era_market/src/136_run_trainer.py
[ "$(ls $R/ace_lora_tensors_yr 2>/dev/null | wc -l)" -gt 100 ] || $PY $W --preprocess --audio-dir ~/era_market/ace_lora/data_yr --tensor-output $R/ace_lora_tensors_yr --checkpoint-dir ./checkpoints --model-variant turbo --max-duration 30 --device cuda:0 > ~/era_market/logs/ace_lora_pre_yr.log 2>&1
$PY $W --dataset-dir $R/ace_lora_tensors_yr --output-dir $R/ace_lora_out_yr_e$ep --checkpoint-dir ./checkpoints --model-variant turbo --adapter-type lora --rank 64 --alpha 128 --batch-size 4 --gradient-accumulation 1 --epochs $ep --save-every 1 --lr 1e-4 --device cuda:0 --yes --plain > ~/era_market/logs/ace_lora_train_yr_e$ep.log 2>&1
cd ~/era_market; YEAR_CAPTION=1 bash src/137_ace_lora_eval.sh $g ace_lora_gen_yr_e$ep $R/ace_lora_out_yr_e$ep/final 0.5
