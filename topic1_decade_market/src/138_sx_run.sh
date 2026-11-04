#!/bin/bash
# T-4h-b: SoulX-Singer melody-driven resynthesis (own transcription, own timbre as prompt) of the vocal stem. usage: 138_sx_run.sh <gpu>  (after 138_sx_prep.py)
H=~/era_market; SX=$H/ext/SoulX-Singer; PY=$H/venv_sx/bin/python; cd $SX; export PYTHONPATH=$SX CUDA_VISIBLE_DEVICES=$1
for f in $H/sx/in/*.wav; do
  sid=$(basename $f .wav); d=$H/sx/ann/$sid; [ -f $d/metadata.json ] && continue
  $PY $H/src/138_sx_ann.py --audio_path $f --save_dir $d --language English --device cuda --vocal_sep False --max_merge_duration 30000 --midi_transcribe True > $H/logs/sx_ann_$sid.log 2>&1
done
echo ANN_DONE
