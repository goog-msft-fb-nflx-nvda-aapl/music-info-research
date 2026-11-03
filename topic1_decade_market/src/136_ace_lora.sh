#!/bin/bash
# N-7 pilot: ACE-Step 1.5 (turbo) LoRA on dataset-A TRAIN accompaniment stems with era captions: dataset build, preprocessing to tensors, LoRA training (rank 64, alpha 128, batch 4). usage: bash 136_ace_lora.sh <gpu> [epochs] [training seed]   (from ~/era_market)
cd ~/era_market; g=$1; ep=${2:-3}; sd=${3:-}; SEEDARG=""; SUF=""; [ -n "$sd" ] && SEEDARG="--seed $sd" && SUF="_s$sd"; export CUDA_VISIBLE_DEVICES=$g   # the trainer addresses cuda:0 internally
[ -d ace_lora/data ] || ./venv/bin/python src/136_ace_lora_data.py
cd ext/ACE-Step-1.5; R=$PWD   # the trainer only accepts dataset and output paths inside the ACE-Step checkout (path-safety check)
PY=~/era_market/venv_ace/bin/python; W=~/era_market/src/136_run_trainer.py
[ "$(ls $R/ace_lora_tensors 2>/dev/null | wc -l)" -gt 100 ] || $PY $W --preprocess --audio-dir ~/era_market/ace_lora/data --tensor-output $R/ace_lora_tensors --checkpoint-dir ./checkpoints --model-variant turbo --max-duration 30 --device cuda:0 > ~/era_market/logs/ace_lora_pre.log 2>&1
$PY $W --dataset-dir $R/ace_lora_tensors --output-dir $R/ace_lora_out_e$ep$SUF --checkpoint-dir ./checkpoints --model-variant turbo --adapter-type lora --rank 64 --alpha 128 --batch-size 4 --gradient-accumulation 1 --epochs $ep --save-every 1 --lr 1e-4 --device cuda:0 $SEEDARG --yes --plain > ~/era_market/logs/ace_lora_train_e$ep$SUF.log 2>&1
touch ~/era_market/ace_lora_train_e$ep$SUF.done
