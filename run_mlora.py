#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTL-LoRA training entry point.

Uses mLoRALinear (multi-task LoRA with shared A, per-task lambda gating, and a
learnable pool of B matrices) as the PEFT method, injected via forward pre-hooks
so that HuggingFace transformer internals need no changes.

Architecture (per linear layer):
  x → A (shared) → λ_t (per-task diagonal) → Σ_b w_t[b] B_b (task-weighted pool) → Δy
  final: y = W x + scaling * Δy

Reference: "MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning"
           https://github.com/pUmpKin-Co/MTL-LoRA
"""
import argparse
import logging
import os
import pprint
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataset import ConcatDataset
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from model import MultiTaskModel_MTL
from mlora import mLoRALinear

# Import shared training/eval infrastructure from run.py
from run import (
    _build_split_dataloaders,
    _compute_gradient_conflicts,
    _concat_train_loader,
    _encode_binary,
    _encode_codebase,
    _encode_retrieval,
    _estimate_initial_losses,
    _log_metrics_table,
    _resolve_active_tasks,
    evaluate,
    test_model,
    train,
)
from utilities import (
    BatchSchedulerSampler,
    MultiTaskEarlyStopper,
    TemperatureSampler,
    TextDataset_clone_detect,
    TextDataset_code_search,
    TextDataset_flakyTest,
    TextDataset_vul_detect,
    save_trainable_params,
    set_seed,
    update_validation_results,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- Target linear-layer name patterns per model_type (same modules as LoRA) ----
_MLORA_PATTERNS = {
    "roberta":     ["query", "key", "value", "attention.output.dense"],
    "t5":          ["q", "k", "v", "o"],
    "codet5p":     ["q", "k", "v", "o"],
    "deepseek_v2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2":       ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3":       ["q_proj", "k_proj", "v_proj", "o_proj"],
    "modernbert":  ["Wqkv", "Wo"],
    "llama":       ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral":     ["q_proj", "k_proj", "v_proj", "o_proj"],
}


# ================== MTL-LoRA helpers ==================

class _MLoRATaskStore:
    """Mutable container that holds the current batch's task IDs.
    Updated by MTLModelMLoRA.forward() before each encoder call;
    read by the forward pre-hook registered on every mLoRALinear layer.
    """
    def __init__(self):
        self.lambda_index: torch.Tensor | None = None


def _make_mlora_hook(store: _MLoRATaskStore):
    """Return a forward pre-hook that injects lambda_index as 2nd positional arg.

    HuggingFace calls a linear layer as  layer(x), giving args=(x,).
    The hook intercepts this and returns (x, lambda_index) so that
    mLoRALinear.forward(x, lambda_index) receives both arguments.
    """
    def _hook(module, args):
        if store.lambda_index is not None and len(args) == 1:
            return (args[0], store.lambda_index)
        return args
    return _hook


def _inject_mlora(base_model, target_names, store, r, lora_alpha, lora_dropout,
                  B_num, lambda_num, B_scale, diagonal_format=True):
    """Replace matching nn.Linear layers with mLoRALinear, register hooks, freeze base.

    Returns list of replaced module full-paths.
    Matching rule: full dotted name ends with any string in target_names,
                   OR the immediate attribute name equals a target.
    """
    # Collect first to avoid mutating the module tree during iteration
    to_replace = []
    for full_name, module in base_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        attr = full_name.split(".")[-1]
        if any(full_name == p or full_name.endswith(f".{p}") or attr == p
               for p in target_names):
            to_replace.append(full_name)

    replaced = []
    for full_name in to_replace:
        parts = full_name.split(".")
        parent = base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        module = getattr(parent, parts[-1])

        new_layer = mLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            B_num=B_num,
            lambda_num=lambda_num,
            diagonal_format=diagonal_format,
            B_scale=B_scale,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=module.bias is not None,
        )
        new_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_layer.bias.data.copy_(module.bias.data)
        new_layer.register_forward_pre_hook(_make_mlora_hook(store))
        setattr(parent, parts[-1], new_layer)
        replaced.append(full_name)

    # Freeze entire backbone; mLoRALinear already marks .weight frozen.
    for p in base_model.parameters():
        p.requires_grad = False

    # Unfreeze the four mLoRA parameter tensors in each injected layer.
    for m in base_model.modules():
        if isinstance(m, mLoRALinear):
            for attr in ("lora_A", "lora_lambdas", "lora_B", "lora_B_w"):
                p = getattr(m, attr, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad = True

    return replaced


class MTLModelMLoRA(MultiTaskModel_MTL):
    """MultiTaskModel_MTL subclass that sets the mLoRA task context before encoding.

    Before calling _encode(), it writes task_ids into self._mlora_store so that
    every mLoRALinear forward pre-hook can pick up the correct lambda_index for
    the current batch without requiring any changes to HuggingFace internals.
    """
    def __init__(self, encoder, config, mlora_store: _MLoRATaskStore):
        super().__init__(encoder, config)
        self._mlora_store = mlora_store

    def forward(self, input_ids, task_ids):
        self._mlora_store.lambda_index = task_ids
        return super().forward(input_ids, task_ids)


# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser()

    # --- data files ---
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--train_data_file_vul",   default="./datasets/dataset_vulnerabilty/train.jsonl")
    parser.add_argument("--eval_data_file_vul",    default="./datasets/dataset_vulnerabilty/valid.jsonl")
    parser.add_argument("--test_data_file_vul",    default="./datasets/dataset_vulnerabilty/test.jsonl")
    parser.add_argument("--train_data_file_clone", default="./datasets/dataset_clone/train.txt")
    parser.add_argument("--eval_data_file_clone",  default="./datasets/dataset_clone/valid.txt")
    parser.add_argument("--test_data_file_clone",  default="./datasets/dataset_clone/test.txt")
    parser.add_argument("--train_data_file_flaky", default="./datasets/dataset_flakytest/train.json")
    parser.add_argument("--eval_data_file_flaky",  default="./datasets/dataset_flakytest/valid.json")
    parser.add_argument("--test_data_file_flaky",  default="./datasets/dataset_flakytest/test.json")
    parser.add_argument("--train_data_file_CodeSearch", default="./datasets/code_search/train.jsonl")
    parser.add_argument("--eval_data_file_CodeSearch",  default="./datasets/code_search/valid.jsonl")
    parser.add_argument("--test_data_file_CodeSearch",  default="./datasets/code_search/test.jsonl")
    parser.add_argument("--codebase_file",   default=None)
    parser.add_argument("--data_file_clone", default="./datasets/dataset_clone/data.jsonl")

    # --- model & training ---
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base")
    parser.add_argument("--output_model_name",  default="unixcoder_mlora_norm_v2")
    parser.add_argument("--nl_length",  default=128, type=int)
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval",  action="store_true")
    parser.add_argument("--do_test",  action="store_true")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size",  default=32, type=int)
    parser.add_argument("--learning_rate",    default=1e-4, type=float)
    parser.add_argument("--max_grad_norm",    default=1.0,  type=float)
    parser.add_argument("--num_train_epochs", default=10,   type=int)
    parser.add_argument("--seed",             default=42,   type=int)
    parser.add_argument("--cuda_device",      default="0",  type=str)
    parser.add_argument("--force_fp32", action="store_true")
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_eval_samples",  default=None, type=int)

    # --- data rates & groups ---
    parser.add_argument("--n_groups",                    default=3,   type=int)
    parser.add_argument("--train_data_rate_vul",         default=1.0, type=float)
    parser.add_argument("--train_data_rate_clone",       default=0.2, type=float)
    parser.add_argument("--train_data_rate_code_search", default=1.0, type=float)
    parser.add_argument("--train_data_rate_flaky",       default=1.0, type=float)

    # --- task selection ---
    parser.add_argument("--tasks", dest="tasks_csv", type=str, default="")
    parser.add_argument("--train_vul",         action="store_true")
    parser.add_argument("--train_clone",       action="store_true")
    parser.add_argument("--train_code_search", action="store_true")
    parser.add_argument("--train_flaky",       action="store_true")

    # --- loss weighting ---
    parser.add_argument("--loss_weighting", default="normalized", type=str,
                        choices=["uncertainty", "gradnorm", "uniform", "normalized", "famo"])
    parser.add_argument("--gradnorm_alpha",     default=1.5,  type=float)
    parser.add_argument("--famo_gamma",         default=0.02, type=float)
    parser.add_argument("--sampling_temperature", default=0.3, type=float)

    # --- MTL-LoRA specific ---
    parser.add_argument("--lora_r",       default=32, type=int,   help="LoRA rank r")
    parser.add_argument("--mlora_B_num",  default=4,  type=int,   help="Number of B matrices in the pool")
    parser.add_argument("--mlora_B_scale",default=1.0,type=float, help="Temperature for softmax over B matrices")
    parser.add_argument("--mlora_diagonal", action="store_true", default=True,
                        help="Use diagonal lambda (default True; False uses full r×r matrix)")

    torch.cuda.empty_cache()
    args = parser.parse_args()
    # run.py's train() checks args.peft_module for gradient-conflict logic
    args.peft_module = "mlora"
    set_seed(args.seed)

    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.n_gpu > 1:
        device = torch.device("cuda:0")
    elif args.n_gpu == 1:
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    autocast_dtype = torch.float32 if getattr(args, "force_fp32", False) else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    args.autocast_dtype = autocast_dtype

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # ModernBERT auto-enables torch.compile on its MLP when Triton is present, which hits a
    # PyTorch Inductor bug (AttributeError: 'float' object has no attribute 'meta').
    config.reference_compile = False
    args.tasks = _resolve_active_tasks(args)
    config.tasks = args.tasks
    config.loss_weighting = args.loss_weighting

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    try:
        base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                               trust_remote_code=True, weights_only=False)
    except (ValueError, OSError, RuntimeError) as e:
        logger.warning("AutoModel failed (%s); retrying with AutoModelForSeq2SeqLM.", e)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                           trust_remote_code=True,
                                                           weights_only=False)

    # ---- Extract encoder for seq2seq (same logic as run.py) ----
    model_type = config.model_type.lower()
    if hasattr(base_model, "encoder") and not model_type.startswith("roberta"):
        encoder_part = base_model.encoder
    else:
        encoder_part = base_model

    # ---- Inject mLoRALinear ----
    target_names = _MLORA_PATTERNS.get(model_type)
    if target_names is None:
        raise ValueError(
            f"No mLoRA patterns for model_type='{model_type}'. "
            f"Add an entry to _MLORA_PATTERNS in run_mlora.py."
        )

    lambda_num = 4   # global task IDs 0-3 are always valid indices
    task_store = _MLoRATaskStore()

    replaced = _inject_mlora(
        base_model=encoder_part,
        target_names=target_names,
        store=task_store,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        B_num=args.mlora_B_num,
        lambda_num=lambda_num,
        B_scale=args.mlora_B_scale,
        diagonal_format=args.mlora_diagonal,
    )
    logger.info("mLoRA: replaced %d Linear layers with mLoRALinear  (r=%d, B_num=%d, λ_num=%d)",
                len(replaced), args.lora_r, args.mlora_B_num, lambda_num)
    for name in replaced:
        logger.debug("  injected: %s", name)

    # Trainable-param report (encoder_part only — for seq2seq models like T5/CodeT5+,
    # base_model also contains the decoder, which is never frozen here since it's not
    # part of the wrapped MTL model and would otherwise inflate this count).
    total_params     = sum(p.numel() for p in encoder_part.parameters())
    trainable_params = sum(p.numel() for p in encoder_part.parameters() if p.requires_grad)
    logger.info("mLoRA encoder params — trainable: %d / %d  (%.2f%%)",
                trainable_params, total_params, 100.0 * trainable_params / max(total_params, 1))

    # ---- Build MTL model ----
    MTLmodel = MTLModelMLoRA(encoder_part, config, task_store).to(args.device)

    # Cast any fp16 trainable params to fp32 to prevent Adam second-moment underflow
    for p in MTLmodel.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    if args.n_gpu > 1:
        MTLmodel = torch.nn.DataParallel(MTLmodel, dim=0)

    logger.info("Model:\n%s", MTLmodel)

    if args.do_train:
        train_results, valid_results = train(args, MTLmodel, tokenizer)
        logger.info("Train results:\n%s",      pprint.pformat(train_results))
        logger.info("Validation results:\n%s", pprint.pformat(valid_results))

    if args.do_eval:
        eval_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="eval"))
        eval_results = evaluate(args, MTLmodel, tokenizer, eval_loaders)
        logger.info("\n***** Eval results *****")
        for k, res in eval_results.items():
            logger.info("  [%s] %s", k, str(res))

    if args.do_test:
        test_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="test"))
        test_res = test_model(args, MTLmodel, tokenizer, test_loaders)
        logger.info("\n***** Test results *****")
        for k, res in test_res.items():
            logger.info("  [%s] %s", k, str(res))


if __name__ == "__main__":
    main()
