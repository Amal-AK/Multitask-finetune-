#!/bin/bash
set -euo pipefail

conda run --no-capture-output -n multitask env CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --model_name_or_path microsoft/unixcoder-base \
    --peft_module lora \
    --loss_weighting uncertainty \
    --tasks vul_detection,clone_detection,code_search,flakiness_detect \
    --max_train_samples 100000 \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --code_length 512 \
    --nl_length 128 \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --sampling_temperature 0.5 \
    --output_model_name unixcoder_lora_100k \
    2>&1 | tee logs/run_100k.log

conda run --no-capture-output -n multitask env CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --model_name_or_path microsoft/unixcoder-base \
    --peft_module lora \
    --loss_weighting normalized \
    --tasks vul_detection,clone_detection,code_search,flakiness_detect \
    --max_train_samples 100000 \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --code_length 512 \
    --nl_length 128 \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --sampling_temperature 0.5 \
    --output_model_name unixcoder_lora_100k_normalized \
    2>&1 | tee logs/run_100k_normalized.log
