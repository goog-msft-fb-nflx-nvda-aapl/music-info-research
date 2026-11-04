#!/bin/bash
# T-4h-b step 3: SoulX-Singer SVS, target = own annotation (original melody + ASR lyrics), prompt = own vocal (self timbre). usage: 138_sx_synth.sh <gpu>   control=melody (F0) by default, CONTROL=score for MIDI
H=~/era_market; SX=$H/ext/SoulX-Singer; PY=$H/venv_sx/bin/python; cd $SX; export PYTHONPATH=$SX CUDA_VISIBLE_DEVICES=$1; C=${CONTROL:-melody}; OUT=$H/sx/gen_$C; mkdir -p $OUT
for d in $H/sx/ann/*/; do sid=$(basename $d); [ $sid = test ] && continue; [ -f $OUT/$sid/generated.wav ] && continue
  w=$(ls $d/long_cut_wavs/*.wav | head -1); m=$d/metadata.json
  $PY -m cli.inference --device cuda --model_path pretrained_models/SoulX-Singer/model.pt --config soulxsinger/config/soulxsinger.yaml --prompt_wav_path $w --prompt_metadata_path $m --target_metadata_path $m --phoneset_path soulxsinger/utils/phoneme/phone_set.json --save_dir $OUT/$sid --auto_shift --pitch_shift 0 --fp16 --control $C > $H/logs/sx_syn_$sid.log 2>&1 || echo FAIL $sid
done; echo SYN_DONE
