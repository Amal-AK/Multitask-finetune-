#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer-rebuttal experiment: add ONE generative analysis task (code_repair, CodeXGLUE
Bugs2Fix) to CodeT5+ 770M's existing 4-task setup, to test whether the paper's findings
(PEFT vs full FT, single- vs multi-task) generalize beyond classification/retrieval.

Self-contained: imports the main codebase's *generic, read-only* helper functions
(_encode_binary, _binary_metrics, _encode_retrieval, _retrieval_mrr, _log_metrics_table,
_pad_seqs, _worker_init_fn) from run.py at the project root, but does not modify them,
and defines its own model (model.py), dataset (dataset.py), and training loop here so the
main 4-task codebase is untouched.

Architecture:
  - CodeT5+ loaded as the FULL T5ForConditionalGeneration (encoder+decoder+lm_head).
  - OpenDelta AdapterModel injected into the WHOLE model (encoder AND decoder attention/
    mlp blocks), since code_repair needs the decoder to adapt too, not just the encoder.
  - The 4 existing tasks use model.encoder (== full_model.encoder, same object) for
    pooling, exactly like the main codebase.
  - code_repair uses the full T5ForConditionalGeneration's own forward(labels=...) for a
    teacher-forced seq2seq loss, and .generate() for exact-match eval.
"""
import argparse
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataset import ConcatDataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from opendelta import AdapterModel

# Append (not insert at 0!) so this script's own directory — already sys.path[0] when run
# directly — takes priority for same-named modules. Local files are named repair_model.py /
# repair_dataset.py (not model.py/dataset.py) specifically to avoid colliding with the
# project root's own model.py/run.py, which root's run.py needs to import internally
# regardless of which directory's sys.path entry is checked first.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from run import (  # noqa: E402  (read-only reuse of generic helpers; nothing imported here is modified)
    _encode_binary,
    _binary_metrics,
    _encode_retrieval,
    _encode_codebase,
    _retrieval_mrr,
    _log_metrics_table,
    _pad_seqs,
    _worker_init_fn,
)
from utilities import (  # noqa: E402
    TextDataset_vul_detect,
    TextDataset_clone_detect,
    TextDataset_code_search,
    TextDataset_flakyTest,
    TemperatureSampler,
    MultiTaskEarlyStopper,
    set_seed,
    update_validation_results,
    save_trainable_params,
)

from repair_dataset import TextDataset_code_repair
from repair_model import MultiTaskModelWithRepair, TASK_ID

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CONFIG_TASKS = ["vul_detection", "clone_detection", "code_search", "flakiness_detect", "code_repair"]

TASK_REGISTRY = {
    "vul_detection":    {"type": "binary",    "dataset": TextDataset_vul_detect},
    "clone_detection":  {"type": "binary",    "dataset": TextDataset_clone_detect},
    "code_search":      {"type": "retrieval", "dataset": TextDataset_code_search},
    "flakiness_detect": {"type": "binary",    "dataset": TextDataset_flakyTest},
    "code_repair":      {"type": "generation","dataset": TextDataset_code_repair},
}


# ================== code_repair eval helper ==================

def _encode_repair_exact_match(args, model, dl, tokenizer, max_eval_batches=None):
    """Generate fixes and compute exact-match accuracy + mean generation loss.

    max_eval_batches caps cost for per-epoch validation; pass None for a full pass
    (used for the best-checkpoint test and the final test, matching how the other
    4 tasks are always evaluated in full).
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()
    correct, total, loss_sum, nb = 0, 0, 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            buggy = batch[0].to(args.device, non_blocking=True)
            fixed = batch[1].to(args.device, non_blocking=True)
            with autocast("cuda", dtype=args.autocast_dtype):
                loss = m.forward_repair(buggy, fixed)
                gen  = m.generate_repair(buggy, max_length=args.code_length)
            loss_sum += float(loss.item())
            nb += 1
            for g, t in zip(gen, fixed):
                g_text = tokenizer.decode(g, skip_special_tokens=True).strip()
                t_text = tokenizer.decode(t, skip_special_tokens=True).strip()
                correct += int(g_text == t_text)
                total += 1
            if max_eval_batches and (i + 1) >= max_eval_batches:
                break
    em = correct / max(total, 1)
    return {"task": "code_repair", "acc": round(em, 4), "eval_loss": round(loss_sum / max(nb, 1), 4)}


# ================== dataloaders ==================

def _build_split_dataloaders(tokenizer, args, active_keys, split="train"):
    result = []
    file_arg_map = {
        "vul_detection":    {"train": args.train_data_file_vul, "eval": args.eval_data_file_vul, "test": args.test_data_file_vul},
        "clone_detection":  {"train": args.train_data_file_clone, "eval": args.eval_data_file_clone, "test": args.test_data_file_clone},
        "code_search":      {"train": args.train_data_file_CodeSearch, "eval": args.eval_data_file_CodeSearch, "test": args.test_data_file_CodeSearch},
        "flakiness_detect": {"train": args.train_data_file_flaky, "eval": args.eval_data_file_flaky, "test": args.test_data_file_flaky},
        "code_repair":      {"train": args.train_data_file_repair, "eval": args.eval_data_file_repair, "test": args.test_data_file_repair},
    }
    for key in active_keys:
        meta = TASK_REGISTRY[key]
        ds = meta["dataset"](tokenizer, args, file_arg_map[key][split])
        if split == "train":
            result.append((key, ds))
        else:
            result.append((key, DataLoader(ds, sampler=SequentialSampler(ds),
                                           batch_size=args.eval_batch_size,
                                           num_workers=4, pin_memory=False,
                                           worker_init_fn=_worker_init_fn)))
    return result


def _concat_train_loader(train_sets, args):
    concat = ConcatDataset([ds for _, ds in train_sets])
    sampler = TemperatureSampler(dataset=concat, batch_size=args.train_batch_size,
                                 temperature=args.sampling_temperature)
    logger.info("Using TemperatureSampler  T=%.2f  probs=%s",
                args.sampling_temperature, [f"{p:.3f}" for p in sampler.probs])
    return DataLoader(
        dataset=concat, sampler=sampler, batch_size=args.train_batch_size,
        shuffle=False, num_workers=4, pin_memory=False, worker_init_fn=_worker_init_fn,
    )


# ================== evaluate / test ==================

def evaluate(args, model, tokenizer, loaders_dict, repair_eval_cap_batches=20):
    model.eval()
    results = {}
    for task, dl in loaders_dict.items():
        if TASK_REGISTRY[task]["type"] == "binary":
            logits, labels, loss = _encode_binary(args, model, dl, task)
            results[task] = _binary_metrics(task, logits, labels, eval_loss=loss)
        elif TASK_REGISTRY[task]["type"] == "retrieval":
            code_vecs, nl_vecs = _encode_retrieval(args, model, dl)
            nl_urls = [ex.url for ex in dl.dataset.examples]
            if getattr(args, "codebase_file", None):
                cb_ds = TextDataset_code_search(tokenizer, args, args.codebase_file)
                cb_dl = DataLoader(cb_ds, sampler=SequentialSampler(cb_ds),
                                   batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
                cb_code_vecs = _encode_codebase(args, model, cb_dl)
                code_urls = [ex.url for ex in cb_ds.examples]
            else:
                cb_code_vecs, code_urls = code_vecs, nl_urls
            results[task] = {"task": task, "mrr": _retrieval_mrr(cb_code_vecs, nl_vecs, code_urls, nl_urls)}
        else:  # generation
            results[task] = _encode_repair_exact_match(args, model, dl, tokenizer,
                                                        max_eval_batches=repair_eval_cap_batches)
    return results


def test_model(args, model, tokenizer, loaders_dict):
    model.eval()
    results = {}
    for task, dl in loaders_dict.items():
        logger.info("  Testing %s (%d examples)", task, len(dl.dataset))
        if TASK_REGISTRY[task]["type"] == "binary":
            logits, labels, _ = _encode_binary(args, model, dl, task)
            results[task] = _binary_metrics(task, logits, labels)
        elif TASK_REGISTRY[task]["type"] == "retrieval":
            code_vecs, nl_vecs = _encode_retrieval(args, model, dl)
            nl_urls = [ex.url for ex in dl.dataset.examples]
            if getattr(args, "codebase_file", None):
                cb_ds = TextDataset_code_search(tokenizer, args, args.codebase_file)
                cb_dl = DataLoader(cb_ds, sampler=SequentialSampler(cb_ds),
                                   batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
                cb_code_vecs = _encode_codebase(args, model, cb_dl)
                code_urls = [ex.url for ex in cb_ds.examples]
            else:
                cb_code_vecs, code_urls = code_vecs, nl_urls
            results[task] = {"task": task, "mrr": _retrieval_mrr(cb_code_vecs, nl_vecs, code_urls, nl_urls)}
        else:  # generation — full pass, no cap
            results[task] = _encode_repair_exact_match(args, model, dl, tokenizer, max_eval_batches=None)
    return results


# ================== train ==================

def train(args, model, tokenizer):
    active_keys = CONFIG_TASKS[:]

    train_sets   = _build_split_dataloaders(tokenizer, args, active_keys, split="train")
    dataloader   = _concat_train_loader(train_sets, args)
    eval_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="eval"))
    test_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="test"))

    steps_per_epoch = len(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    max_steps = steps_per_epoch * args.num_train_epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(max_steps * 0.1), num_training_steps=max_steps
    )
    scaler = GradScaler("cuda", enabled=(args.autocast_dtype == torch.float16))

    static_weights = {k: 1.0 for k in active_keys}
    logger.info("  Normalized weights: uniform for epoch 1, updated each epoch (w ~ L_i).")

    logger.info("=" * 70)
    logger.info("***** Running Training (code_repair 5-task experiment) *****")
    logger.info("  Model      : %s", args.model_name_or_path)
    logger.info("  Epochs     : %d", args.num_train_epochs)
    logger.info("  Batch size : %d  |  LR: %.2e", args.train_batch_size, args.learning_rate)
    logger.info("  Steps/epoch: ~%d  |  Total steps: ~%d", steps_per_epoch, max_steps)
    for k, ds in train_sets:
        logger.info("    %-22s  %d examples", k, len(ds))
    logger.info("=" * 70)

    _task_bce = {}
    for key, ds in train_sets:
        if TASK_REGISTRY[key]["type"] != "binary":
            continue
        labels = [ex.label for ex in ds.examples]
        pos = sum(labels)
        neg = len(labels) - pos
        if pos > 0 and (neg / pos) > 2.0:
            pw = neg / pos
            _task_bce[key] = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pw], device=args.device, dtype=torch.float32))
            logger.info("  pos_weight %-22s = %.1f  (pos=%d  neg=%d)", key, pw, int(pos), int(neg))
        else:
            _task_bce[key] = nn.BCEWithLogitsLoss()
    bce = nn.BCEWithLogitsLoss()
    ce = CrossEntropyLoss()

    def _bce(task_name, logits, labels):
        return _task_bce.get(task_name, bce)(logits, labels)

    best_score = -np.inf
    train_results, validation_results = {}, {}
    early_stopper = MultiTaskEarlyStopper(active_keys, patience=4)
    pad_id = (model.module if hasattr(model, "module") else model).pad_token_id

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(args.device)
        losses_per_task = {k: [] for k in active_keys}
        total_losses = []
        step = 0
        model.train()

        for batch in dataloader:
            task_name = CONFIG_TASKS[int(batch[2][0].item())]
            m = model.module if hasattr(model, "module") else model

            with autocast("cuda", dtype=args.autocast_dtype):
                if task_name == "code_search":
                    code_seq = batch[0].to(args.device, non_blocking=True)
                    nl_seq   = batch[1].to(args.device, non_blocking=True)
                    task_ids = batch[2].to(args.device)
                    bs = code_seq.size(0)
                    code_p, nl_p = _pad_seqs([code_seq, nl_seq], pad_id)
                    vecs = model(torch.cat([code_p, nl_p], 0), task_ids.repeat(2))["code_search"]
                    scores = torch.einsum("ab,cb->ac", vecs[bs:], vecs[:bs])
                    loss = ce(scores * 20, torch.arange(bs, device=scores.device))
                elif task_name == "code_repair":
                    buggy    = batch[0].to(args.device, non_blocking=True)
                    fixed    = batch[1].to(args.device, non_blocking=True)
                    task_ids = batch[2].to(args.device)
                    # Routed through forward() (not m.forward_repair directly) so
                    # nn.DataParallel's scatter/gather applies when n_gpu > 1; each
                    # replica returns a 1-element loss tensor, .mean() combines them.
                    loss = model(buggy, task_ids, fixed_ids=fixed)["code_repair_loss"].mean()
                else:
                    code_seq = batch[0].to(args.device, non_blocking=True)
                    labels   = batch[1].to(args.device, non_blocking=True).float()
                    task_ids = batch[2].to(args.device)
                    logit = model(code_seq, task_ids)[task_name]
                    loss = _bce(task_name, logit.squeeze(), labels.squeeze())

            losses_per_task[task_name].append(loss.item())
            total_loss = static_weights[task_name] * loss
            total_losses.append(total_loss.item())

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            if step % 32 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                task_str = "  ".join(f"{k}={np.mean(v):.4f}" for k, v in losses_per_task.items() if v)
                logger.info("E%d/%d  step %d/%d  total=%.4f  lr=%.2e  |  %s",
                            epoch + 1, args.num_train_epochs, step, steps_per_epoch,
                            float(np.mean(total_losses)), lr_now, task_str)

        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
        train_results.setdefault("total_train_loss", []).append(round(np.mean(total_losses), 4))
        train_results.setdefault("epoch_time_s", []).append(round(epoch_time, 1))
        train_results.setdefault("peak_gpu_memory_gb", []).append(round(peak_mem, 3))
        for k in active_keys:
            if losses_per_task[k]:
                train_results.setdefault(f"{k}_train_loss", []).append(round(np.mean(losses_per_task[k]), 4))

        logger.info("-" * 70)
        logger.info("Epoch %d/%d — Train Summary", epoch + 1, args.num_train_epochs)
        for k in active_keys:
            loss_val = np.mean(losses_per_task[k]) if losses_per_task[k] else float("nan")
            logger.info("  %-22s  loss=%8.4f  weight=%8.4f", k, loss_val, static_weights[k])
        logger.info("  %-22s  TOTAL=%8.4f  epoch_time=%.1fs  peak_gpu_mem=%.3fGB",
                    "", np.mean(total_losses), epoch_time, peak_mem)
        logger.info("-" * 70)

        # Normalized weight update: w_i ~ L_i for next epoch
        epoch_losses = {k: np.mean(v) for k, v in losses_per_task.items() if v}
        raw_w = {k: max(v, 1e-8) for k, v in epoch_losses.items()}
        w_sum = sum(raw_w.values())
        static_weights = {k: v * len(active_keys) / w_sum for k, v in raw_w.items()}
        logger.info("  Normalized weights -> next epoch: %s", {k: f"{v:.4f}" for k, v in static_weights.items()})

        eval_results = evaluate(args, model, tokenizer, eval_loaders,
                                repair_eval_cap_batches=args.repair_eval_cap_batches)
        for k, res in eval_results.items():
            update_validation_results(res, validation_results)

        if early_stopper.early_stop(validation_results):
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

        scores = []
        for k in active_keys:
            t = TASK_REGISTRY[k]["type"]
            if t == "binary":
                scores.append(eval_results[k]["f1"])
            elif t == "retrieval":
                scores.append(eval_results[k]["mrr"])
            else:
                scores.append(eval_results[k]["acc"])  # exact-match for code_repair
        combined = float(np.mean(scores))
        _log_metrics_table("Eval", epoch + 1, eval_results, combined)

        if combined >= best_score:
            best_score = combined
            save_trainable_params(model, f"./adapters/{args.output_model_name}/model.bin")
            logger.info("  >> New best (%.4f) — checkpoint saved.", best_score)
            test_res = test_model(args, model, tokenizer, test_loaders)
            _log_metrics_table("Test", epoch + 1, test_res)

    test_res = test_model(args, model, tokenizer, test_loaders)
    _log_metrics_table("Final Test", args.num_train_epochs, test_res)
    return train_results, validation_results


# ================== main ==================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dpath(*parts):
    """Absolute path under the project root's datasets/ dir — robust regardless of CWD."""
    return os.path.join(_PROJECT_ROOT, "datasets", *parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file_vul",   default=_dpath("dataset_vulnerabilty", "train.jsonl"))
    parser.add_argument("--eval_data_file_vul",    default=_dpath("dataset_vulnerabilty", "valid.jsonl"))
    parser.add_argument("--test_data_file_vul",    default=_dpath("dataset_vulnerabilty", "test.jsonl"))
    parser.add_argument("--train_data_file_clone", default=_dpath("dataset_clone", "train.txt"))
    parser.add_argument("--eval_data_file_clone",  default=_dpath("dataset_clone", "valid.txt"))
    parser.add_argument("--test_data_file_clone",  default=_dpath("dataset_clone", "test.txt"))
    parser.add_argument("--train_data_file_flaky", default=_dpath("dataset_flakytest", "train.json"))
    parser.add_argument("--eval_data_file_flaky",  default=_dpath("dataset_flakytest", "valid.json"))
    parser.add_argument("--test_data_file_flaky",  default=_dpath("dataset_flakytest", "test.json"))
    parser.add_argument("--train_data_file_CodeSearch", default=_dpath("code_search", "train.jsonl"))
    parser.add_argument("--eval_data_file_CodeSearch",  default=_dpath("code_search", "valid.jsonl"))
    parser.add_argument("--test_data_file_CodeSearch",  default=_dpath("code_search", "test.jsonl"))
    parser.add_argument("--train_data_file_repair", default=_dpath("dataset_repair", "train.jsonl"))
    parser.add_argument("--eval_data_file_repair",  default=_dpath("dataset_repair", "valid.jsonl"))
    parser.add_argument("--test_data_file_repair",  default=_dpath("dataset_repair", "test.jsonl"))
    parser.add_argument("--codebase_file", default=None)
    parser.add_argument("--data_file_clone", default=_dpath("dataset_clone", "data.jsonl"))

    parser.add_argument("--model_name_or_path", default="Salesforce/codet5p-770m")
    parser.add_argument("--bottleneck_dim", default=64, type=int)
    parser.add_argument("--output_model_name", default="codet5p_repair_5task_adapter_norm_v2")

    parser.add_argument("--nl_length",  default=128, type=int)
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size",  default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda_device", default="0", type=str)
    parser.add_argument("--sampling_temperature", default=0.3, type=float)
    parser.add_argument("--repair_eval_cap_batches", default=20, type=int,
                        help="Cap generation batches during per-epoch validation (speed). "
                             "Final test always runs the full set.")
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_eval_samples",  default=None, type=int)

    torch.cuda.empty_cache()
    args = parser.parse_args()
    args.peft_module = "adapter"  # for parity with imported helpers' expectations
    set_seed(args.seed)

    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.n_gpu > 1:
        device = torch.device("cuda:0")   # first of the CUDA_VISIBLE_DEVICES-selected GPUs
    elif args.n_gpu == 1:
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    args.device = device
    args.autocast_dtype = torch.bfloat16 if (args.n_gpu and torch.cuda.is_bf16_supported()) else torch.float16
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.tasks = CONFIG_TASKS[:4]  # only the pooling tasks need a task_heads entry shape
    config.pad_token_id = config.pad_token_id or 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                        trust_remote_code=True, weights_only=False)

    # Inject adapters into BOTH encoder and decoder (code_repair needs the decoder to
    # adapt too — unlike the 4 pooling tasks, which only ever touch the encoder).
    adapter_patterns = ["SelfAttention", "DenseReluDense", "EncDecAttention"]
    delta_model = AdapterModel(backbone_model=full_model, modified_modules=adapter_patterns,
                               bottleneck_dim=args.bottleneck_dim)
    # Deliberately NOT excluding lm_head from freezing (unlike the main run.py's pattern,
    # which excludes it defensively for the encoder-only extraction where lm_head doesn't
    # exist) — keeping lm_head frozen here keeps this experiment "PEFT-style" (only deltas
    # + task heads trainable), consistent with the main codebase's adapter experiments.
    delta_model.freeze_module(exclude=["deltas"])
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

    encoder_part = full_model.encoder

    # Eager-instantiate lazily-built AdapterLayer weights before the optimizer is built
    # (same fix as the main run.py — otherwise the optimizer never sees these params).
    _hs = config.d_model
    _dummy = torch.zeros(1, 2, _hs, dtype=torch.float32)
    for al in delta_model.delta_modules:
        if hasattr(al, "init_device"):
            al.init_device = args.device
        if hasattr(al, "device"):
            al.device = args.device
        if not al.instantiated:
            al.instantiate(_dummy)

    MTLmodel = MultiTaskModelWithRepair(encoder_part, full_model, config).to(args.device)

    # fp16 Adam second-moment underflow fix (CodeT5+'s config loads in fp16, OpenDelta
    # then creates adapter weights in fp16 too) — cast all trainable params to fp32.
    for p in MTLmodel.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    if args.n_gpu > 1:
        MTLmodel = torch.nn.DataParallel(MTLmodel, dim=0)
        logger.info("Wrapped in DataParallel across %d GPUs.", args.n_gpu)

    logger.info("Model:\n%s", MTLmodel)

    train_results, valid_results = train(args, MTLmodel, tokenizer)
    logger.info("Train results:\n%s", pprint.pformat(train_results))
    logger.info("Validation results:\n%s", pprint.pformat(valid_results))


if __name__ == "__main__":
    main()
