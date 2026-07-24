#!/bin/bash
# ModernBERT prefix tuning rerun — fixed attn_implementation=eager for SDPA/prefix compat
# GPU 0 (single run)
set -euo pipefail

mkdir -p logs

echo "=== ModernBERT prefix rerun — $(date) ==="
echo "    Fix: attn_implementation=eager (SDPA rejects extended key length from prefix tokens)"
echo ""

conda run --no-capture-output -n multitask \
    env CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --model_name_or_path answerdotai/ModernBERT-base \
        --peft_module prefix \
        --bottleneck_dim 64 --lora_r 32 \
        --loss_weighting normalized \
        --tasks vul_detection,clone_detection,code_search,flakiness_detect \
        --max_train_samples 300000 --num_train_epochs 10 \
        --train_batch_size 32 --eval_batch_size 32 \
        --code_length 512 --nl_length 128 \
        --learning_rate 1e-4 --max_grad_norm 1.0 \
        --sampling_temperature 0.3 \
        --output_model_name modernbert_prefix_norm_v2 \
        2>&1 | tee logs/modernbert_prefix_norm_v2_300k.log

echo ""
echo "=== ModernBERT prefix done — $(date) ==="
