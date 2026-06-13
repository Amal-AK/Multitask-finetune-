#!/bin/bash
# 300K full runs: {unixcoder, codet5p-770m} × {adapter, parallel_adapter, prefix} × {norm_v2, unc_v2}
# GPU 0: UniXcoder — 6 runs in sequence (adapter→parallel_adapter→prefix, each × 2 loss modes)
# GPU 1: CodeT5+ 770M — 6 runs in sequence (same order)
# Both GPUs run in parallel; each GPU's 6 runs are strictly sequential (model×peft×loss order).
#
# Usage:
#   bash run_300k.sh            # launch both GPUs
#   bash run_300k.sh unix       # GPU 0 only (UniXcoder)
#   bash run_300k.sh codet5p    # GPU 1 only (CodeT5+ 770M)
set -euo pipefail

mkdir -p logs

FILTER="${1:-all}"
LOSS_FILTER="${2:-all}"   # "normalized" | "uncertainty" | "all"

EPOCHS=10
MAXN=300000
LR=1e-4
BDIM=64

# ============================================================
# GPU 0 — UniXcoder (microsoft/unixcoder-base, roberta)
# batch=32 (encoder-only, ~125M params)
# ============================================================
run_unix() {
    for peft in adapter parallel_adapter prefix; do
        for loss in normalized uncertainty; do
            [[ "$LOSS_FILTER" != "all" && "$loss" != "$LOSS_FILTER" ]] && continue
            local tag
            tag="$([ "$loss" = "normalized" ] && echo "norm_v2" || echo "unc_v2")"
            local label="unix_${peft}_${tag}"
            echo "[GPU0] START  $label  $(date)"
            conda run --no-capture-output -n multitask \
                env CUDA_VISIBLE_DEVICES=0 python -u run.py \
                    --model_name_or_path microsoft/unixcoder-base \
                    --peft_module         "$peft" \
                    --bottleneck_dim      $BDIM \
                    --loss_weighting      "$loss" \
                    --tasks               vul_detection,clone_detection,code_search,flakiness_detect \
                    --max_train_samples   $MAXN \
                    --num_train_epochs    $EPOCHS \
                    --train_batch_size    32 \
                    --eval_batch_size     32 \
                    --code_length         512 \
                    --nl_length           128 \
                    --learning_rate       $LR \
                    --max_grad_norm       1.0 \
                    --sampling_temperature 0.3 \
                    --output_model_name   "$label" \
                    2>&1 | tee "logs/${label}_300k.log"
            echo "[GPU0] DONE   $label  $(date)"
        done
    done
}

# ============================================================
# GPU 1 — CodeT5+ 770M (Salesforce/codet5p-770m, t5 encoder)
# batch=16 (770M params — half of unixcoder batch to match VRAM)
# ============================================================
run_codet5p() {
    for peft in adapter parallel_adapter prefix; do
        for loss in normalized uncertainty; do
            [[ "$LOSS_FILTER" != "all" && "$loss" != "$LOSS_FILTER" ]] && continue
            local tag
            tag="$([ "$loss" = "normalized" ] && echo "norm_v2" || echo "unc_v2")"
            local label="codet5p_${peft}_${tag}"
            echo "[GPU1] START  $label  $(date)"
            conda run --no-capture-output -n multitask \
                env CUDA_VISIBLE_DEVICES=1 python -u run.py \
                    --model_name_or_path Salesforce/codet5p-770m \
                    --peft_module         "$peft" \
                    --bottleneck_dim      $BDIM \
                    --loss_weighting      "$loss" \
                    --tasks               vul_detection,clone_detection,code_search,flakiness_detect \
                    --max_train_samples   $MAXN \
                    --num_train_epochs    $EPOCHS \
                    --train_batch_size    16 \
                    --eval_batch_size     16 \
                    --code_length         512 \
                    --nl_length           128 \
                    --learning_rate       $LR \
                    --max_grad_norm       1.0 \
                    --sampling_temperature 0.3 \
                    --output_model_name   "$label" \
                    2>&1 | tee "logs/${label}_300k.log"
            echo "[GPU1] DONE   $label  $(date)"
        done
    done
}

echo "=== 300K run — $(date) ==="
echo "    epochs=$EPOCHS  max_samples=$MAXN  adapter_dim=$BDIM  lr=$LR"
echo "    sequence order: model × peft(adapter→parallel_adapter→prefix) × loss(norm_v2→unc_v2)"
echo ""

case "$FILTER" in
    unix)    run_unix ;;
    codet5p) run_codet5p ;;
    *)
        run_unix    &
        run_codet5p &
        wait
        ;;
esac

echo ""
echo "=== All 300K runs complete — $(date) ==="
