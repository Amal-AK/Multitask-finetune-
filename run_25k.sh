#!/bin/bash
set -euo pipefail

python run.py \
    --model_name_or_path microsoft/unixcoder-base \
    --peft_module lora \
    --loss_weighting normalized \
    --tasks vul_detection,clone_detection,code_search,flakiness_detect \
    --max_train_samples 25000 \
    --max_eval_samples 2000 \
    --num_train_epochs 10 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --code_length 512 \
    --nl_length 128 \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --output_model_name unixcoder_lora_25k_qkv \
    2>&1 | tee logs/run_25k.log
