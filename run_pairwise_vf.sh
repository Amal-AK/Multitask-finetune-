#!/bin/bash
# Pairwise: vul_detection + flakiness_detect — serial adapter (d=64), norm_v2
# GPU 2 (serial): UniXcoder-base → ModernBERT-base → CodeT5+ 770M
# GPU 3 (serial): DeepSeek-Coder 1.3B → Qwen2.5-Coder-1.5B
#
# Usage:
#   bash run_pairwise_vf.sh           # launch both GPUs in parallel
#   bash run_pairwise_vf.sh gpu2      # GPU 2 only (3 small models)
#   bash run_pairwise_vf.sh gpu3      # GPU 3 only (2 large models)
set -euo pipefail

mkdir -p logs/pairwise

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
TASKS="vul_detection,flakiness_detect"
LOSS="normalized"

# ============================================================
# GPU 2 — UniXcoder → ModernBERT → CodeT5+ (sequential)
# ============================================================
run_gpu2() {
    local l1="unixcoder_vul_flak_adapter_norm_v2"
    local l2="modernbert_vul_flak_adapter_norm_v2"
    local l3="codet5p_vul_flak_adapter_norm_v2"

    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=2 python -u run.py \
            --model_name_or_path microsoft/unixcoder-base \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size 32 --eval_batch_size 32 \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$l1" \
            2>&1 | tee "logs/pairwise/${l1}.log" &

    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=2 python -u run.py \
            --model_name_or_path answerdotai/ModernBERT-base \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size 32 --eval_batch_size 32 \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$l2" \
            2>&1 | tee "logs/pairwise/${l2}.log" &

    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=2 python -u run.py \
            --model_name_or_path Salesforce/codet5p-770m \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size 16 --eval_batch_size 16 \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$l3" \
            2>&1 | tee "logs/pairwise/${l3}.log" &

    wait
    echo "[GPU2] All 3 models done  $(date)"
}

# ============================================================
# GPU 3 — DeepSeek → Qwen (sequential)
# ============================================================
run_gpu3() {
    local l1="deepseek_vul_flak_adapter_norm_v2"
    local l2="qwen25_vul_flak_adapter_norm_v2"

    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=3 python -u run.py \
            --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size 8 --eval_batch_size 8 \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$l1" \
            2>&1 | tee "logs/pairwise/${l1}.log" &

    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=3 python -u run.py \
            --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
            --peft_module adapter --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks $TASKS \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size 8 --eval_batch_size 8 \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$l2" \
            2>&1 | tee "logs/pairwise/${l2}.log" &

    wait
    echo "[GPU3] Both models done  $(date)"
}

echo "=== Pairwise vul+flak — $(date) ==="
echo "    tasks=$TASKS  peft=adapter(d=$BDIM)  loss=$LOSS  epochs=$EPOCHS"
echo "    GPU2 (parallel): UniXcoder + ModernBERT + CodeT5+"
echo "    GPU3 (parallel): DeepSeek + Qwen"
echo ""

run_gpu2 &
run_gpu3 &
wait

echo ""
echo "=== Pairwise vul+flak complete — $(date) ==="
