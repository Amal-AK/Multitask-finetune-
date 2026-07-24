#!/bin/bash
# Pairwise: clone_detection + code_search — serial adapter (d=64), norm_v2
# All 5 models across 4 GPUs in parallel:
#   GPU 0: UniXcoder → ModernBERT (sequential, both small)
#   GPU 1: CodeT5+
#   GPU 2: DeepSeek
#   GPU 3: Qwen
#
# Batches halved vs non-search pairs: code_search doubles the forward pass
# (code + NL both run through the encoder each step).
set -euo pipefail

mkdir -p logs/pairwise

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64
TASKS="clone_detection,code_search"
LOSS="normalized"

run_model() {
    local GPU="$1" MODEL="$2" LABEL="$3" BATCH="$4"
    echo "[GPU${GPU}] START  ${LABEL}  $(date)"
    conda run --no-capture-output -n multitask \
        env CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
            --model_name_or_path  "$MODEL" \
            --peft_module adapter  --bottleneck_dim $BDIM \
            --loss_weighting $LOSS --tasks "$TASKS" \
            --max_train_samples $MAXN --num_train_epochs $EPOCHS \
            --train_batch_size "$BATCH" --eval_batch_size "$BATCH" \
            --code_length 512 --nl_length 128 \
            --learning_rate $LR --max_grad_norm 1.0 \
            --sampling_temperature 0.3 \
            --output_model_name "$LABEL" \
            2>&1 | tee "logs/pairwise/${LABEL}.log"
    echo "[GPU${GPU}] DONE   ${LABEL}  $(date)"
}

echo "=== Pairwise clone+search — $(date) ==="
echo "    tasks=$TASKS  peft=adapter(d=$BDIM)  loss=$LOSS  epochs=$EPOCHS"
echo "    GPU0: UniXcoder→ModernBERT   GPU1: CodeT5+   GPU2: DeepSeek   GPU3: Qwen"
echo ""

( run_model 0 microsoft/unixcoder-base            unixcoder_clone_search_adapter_norm_v2   32
  run_model 0 answerdotai/ModernBERT-base          modernbert_clone_search_adapter_norm_v2  32 ) &

run_model 1 Salesforce/codet5p-770m               codet5p_clone_search_adapter_norm_v2     16 &
run_model 2 deepseek-ai/deepseek-coder-1.3b-base  deepseek_clone_search_adapter_norm_v2     8 &
run_model 3 Qwen/Qwen2.5-Coder-1.5B              qwen25_clone_search_adapter_norm_v2       8 &

wait

echo ""
echo "=== Pairwise clone+search complete — $(date) ==="
