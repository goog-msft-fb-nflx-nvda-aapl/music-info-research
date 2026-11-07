#!/bin/bash
# archive the scored generation directories of 2026-11-07 (CLAP embeddings + demo set per configuration); wavs are deleted only after this finished and was checked. usage: bash src/run_archive.sh <gpu> (from ~/era_market)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1
while read d c; do ./venv_mcl/bin/python src/150_archive_round.py --gendir $d --config $c > logs/archive_$c.log 2>&1; done <<'LIST'
gen_ax_ctrlc_ge10_hpcp musicgen_adapter_ctrlc_hpcp
gen_ax_onehot_pfx16_ge10_hpcp adapter_onehot16_hpcp
gen_ax_onehot_pfx16_ge10_hpcp_dose adapter_onehot16_dose
gen_ax_onehot_pfx16_ge10_hpcp_dose2 adapter_onehot16_dose2
gen_steer_b4 steer_prompt_b4
gen_steer_b16 steer_prompt_b16
gen_steer_b64 steer_prompt_b64
gen_steer_rc0.02 steer_real_c0.02
gen_steer_rc0.05 steer_real_c0.05
gen_steer_rc0.1 steer_real_c0.1
gen_steer_rc0.2 steer_real_c0.2
gen_ctrl_b0 control_unsteered_1990s
gen_ctrl_rc05k4 control_real_c0.05_4rows
gen_ctrl_rc2k4 control_real_c0.2_4rows
LIST
touch archive.done
