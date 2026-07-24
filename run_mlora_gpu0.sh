#!/bin/bash
# MTL-LoRA 4-task runs — all 5 models sequentially on GPU 0, 300K samples, normalized loss
set -euo pipefail

mkdir -p logs

EPOCHS=10
MAXN=300000
LR=1e-4
LORA_R=32
B_NUM=4
B_SCALE=1.0
TASKS="vul_detection,clone_detection,code_search,flakiness_detect"
LOSS="normalized"
GPU=0

run_model() {
    local MODEL_PATH="$1"
    local LABEL="$2"
    local BATCH="$3"

    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run_mlora.py \
            --model_name_or_path "$MODEL_PATH" \
            --lora_r $LORA_R \
            --mlora_B_num $B_NUM \
            --mlora_B_scale $B_SCALE \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size "$BATCH" --eval_batch_size "$BATCH" \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== MTL-LoRA 4-task runs (GPU 0) — $(date) ==="
echo "    r=${LORA_R}  B_num=${B_NUM}  B_scale=${B_SCALE}  loss=${LOSS}  epochs=${EPOCHS}"
echo ""

run_model microsoft/unixcoder-base             unixcoder_mlora_norm_v2_300k   32
run_model answerdotai/ModernBERT-base           modernbert_mlora_norm_v2_300k  32
run_model Salesforce/codet5p-770m               codet5p_mlora_norm_v2_300k     16
run_model deepseek-ai/deepseek-coder-1.3b-base  deepseek_mlora_norm_v2_300k     8
run_model Qwen/Qwen2.5-Coder-1.5B              qwen25_mlora_norm_v2_300k        8

echo ""
echo "=== MTL-LoRA runs complete (GPU 0) — $(date) ==="
