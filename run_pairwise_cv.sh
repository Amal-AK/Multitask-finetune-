#!/bin/bash
# Pairwise: clone_detection + vul_detection — serial adapter (d=64), norm_v2
# GPU 3 (sequential): UniXcoder → ModernBERT → CodeT5+ → DeepSeek → Qwen
set -euo pipefail

mkdir -p logs/pairwise

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
TASKS="clone_detection,vul_detection"
LOSS="normalized"
GPU=3

run_model() {
    local MODEL_PATH="$1"
    local LABEL="$2"
    local BATCH="$3"

    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path "$MODEL_PATH" \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size "$BATCH" --eval_batch_size "$BATCH" \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/pairwise/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== Pairwise clone+vul — $(date) ==="
echo "    tasks=$TASKS  peft=adapter(d=$BDIM)  loss=$LOSS  epochs=$EPOCHS"
echo "    GPU${GPU} (sequential): UniXcoder → ModernBERT → CodeT5+ → DeepSeek → Qwen"
echo ""

run_model microsoft/unixcoder-base             unixcoder_clone_vul_adapter_norm_v2   128
run_model answerdotai/ModernBERT-base           modernbert_clone_vul_adapter_norm_v2  128
run_model Salesforce/codet5p-770m               codet5p_clone_vul_adapter_norm_v2      64
run_model deepseek-ai/deepseek-coder-1.3b-base  deepseek_clone_vul_adapter_norm_v2     32
run_model Qwen/Qwen2.5-Coder-1.5B              qwen25_clone_vul_adapter_norm_v2        32

echo ""
echo "=== Pairwise clone+vul complete — $(date) ==="
