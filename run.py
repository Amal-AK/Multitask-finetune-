#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pprint
import random
import torch
import numpy as np
from model import MultiTaskModel_MTL
from torch.utils.data.dataset import ConcatDataset
import torch.nn as nn
import transformers
from torch.amp import autocast
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    get_linear_schedule_with_warmup,
    AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
)
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType
from opendelta import AdapterModel, ParallelAdapterModel
from utilities import (
    TextDataset_vul_detect, TextDataset_clone_detect,
    TextDataset_code_search, TextDataset_flakyTest,
    BatchSchedulerSampler, TemperatureSampler, MultiTaskEarlyStopper,
    set_seed, update_validation_results,
    save_trainable_params,
)
from sklearn.metrics import recall_score, precision_score, f1_score

# ================== ENV & LOGGING ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
CONFIG_TASKS = ["vul_detection", "clone_detection", "code_search", "flakiness_detect"]


# ================== FLEXIBLE TASK REGISTRY ==================
TASK_REGISTRY = {
    "vul_detection": {
        "type": "binary",
        "dataset": TextDataset_vul_detect,
        "train_arg": "train_data_file_vul",
        "eval_arg":  "eval_data_file_vul",
        "test_arg":  "test_data_file_vul",
    },
    "clone_detection": {
        "type": "binary",
        "dataset": TextDataset_clone_detect,
        "train_arg": "train_data_file_clone",
        "eval_arg":  "eval_data_file_clone",
        "test_arg":  "test_data_file_clone",
    },
    "code_search": {
        "type": "retrieval",
        "dataset": TextDataset_code_search,
        "train_arg": "train_data_file_CodeSearch",
        "eval_arg":  "eval_data_file_CodeSearch",
        "test_arg":  "test_data_file_CodeSearch",
    },
    "flakiness_detect": {
        "type": "binary",
        "dataset": TextDataset_flakyTest,
        "train_arg": "train_data_file_flaky",
        "eval_arg":  "eval_data_file_flaky",
        "test_arg":  "test_data_file_flaky",
    },
}



def _resolve_active_tasks(args):
    """
    Determine which tasks to run (1–4) using pretty names only.
    Priority: --tasks CSV > boolean flags > default (all 4).
    """
    if getattr(args, "tasks_csv", None):
        keys = [k.strip() for k in args.tasks_csv.split(",") if k.strip()]
    else:
        keys = []
        if getattr(args, "train_vul", False):         keys.append("vul_detection")
        if getattr(args, "train_clone", False):       keys.append("clone_detection")
        if getattr(args, "train_code_search", False): keys.append("code_search")
        if getattr(args, "train_flaky", False):       keys.append("flakiness_detect")
    if not keys:
        keys = CONFIG_TASKS[:]  # all pretty names
    if len(keys) < 1 or len(keys) > 4:
        raise ValueError(f"You must choose between 1 and 4 tasks, got {len(keys)}: {keys}")
    for k in keys:
        if k not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{k}'. Valid: {list(TASK_REGISTRY.keys())}")
    return keys




def _build_split_dataloaders(tokenizer, args, active_keys, split="train"):
    """
    Return list[(key, dataset_or_dataloader)] for the requested split.
    'train' returns raw datasets; 'eval'/'test' returns DataLoaders.
    """
    result = []
    for key in active_keys:
        meta = TASK_REGISTRY[key]
        ds = meta["dataset"](tokenizer, args, getattr(args, meta[f"{split}_arg"]))
        if split == "train":
            result.append((key, ds))
        else:
            result.append((key, DataLoader(ds, sampler=SequentialSampler(ds),
                                           batch_size=args.eval_batch_size,
                                           num_workers=4, pin_memory=False,
                                           worker_init_fn=_worker_init_fn)))
    return result

def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def _concat_train_loader(train_sets, args):
    concat = ConcatDataset([ds for _, ds in train_sets])
    temp   = getattr(args, 'sampling_temperature', 0.0)
    if temp > 0:
        sampler = TemperatureSampler(dataset=concat, batch_size=args.train_batch_size,
                                     temperature=temp)
        logger.info("Using TemperatureSampler  T=%.2f  probs=%s",
                    temp, [f"{p:.3f}" for p in sampler.probs])
    else:
        sampler = BatchSchedulerSampler(dataset=concat, batch_size=args.train_batch_size)
    return DataLoader(
        dataset=concat,
        sampler=sampler,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        worker_init_fn=_worker_init_fn,
    )

def _pad_seqs(tensors, pad_id):
    """Pad a list of (B, T_i) int tensors to the same T = max(T_i).

    Needed because code sequences (code_length) and NL sequences (nl_length)
    may differ; the single encoder forward pass requires a uniform seq_len.
    """
    max_len = max(t.size(1) for t in tensors)
    return [
        F.pad(t, (0, max_len - t.size(1)), value=pad_id) if t.size(1) < max_len else t
        for t in tensors
    ]


# ================== NORMALIZED STATIC WEIGHTS ==================

def _estimate_initial_losses(args, model, train_sets, active_keys, n_batches=50):
    """Run n_batches without grad updates to estimate the mean loss per task.
    Used once before training to set frozen inverse-loss weights.
    Returns {task_name: mean_loss}.
    """
    # Always use round-robin here so every task is represented equally,
    # regardless of the sampling strategy used during training.
    concat     = ConcatDataset([ds for _, ds in train_sets])
    tmp_loader = DataLoader(
        dataset=concat,
        sampler=BatchSchedulerSampler(dataset=concat, batch_size=args.train_batch_size),
        batch_size=args.train_batch_size,
        shuffle=False, num_workers=4, pin_memory=False,
        worker_init_fn=_worker_init_fn,
    )
    bce = nn.BCEWithLogitsLoss()
    ce  = CrossEntropyLoss()
    pad_id = (model.module if args.n_gpu > 1 else model).pad_token_id
    n_tasks = len(active_keys)
    losses_per_task = {k: [] for k in active_keys}

    model.eval()
    with torch.no_grad():
        data_iter = iter(tmp_loader)
        for _ in range(n_batches):
            try:
                batches = [next(data_iter) for _ in range(n_tasks)]
            except StopIteration:
                break
            all_seqs, all_task_ids, binary_data, cs_batch_size = [], [], [], 0
            for b in batches:
                task_name = CONFIG_TASKS[int(b[2][0].item())]
                code_seq  = b[0].to(args.device, non_blocking=True)
                all_seqs.append(code_seq)
                all_task_ids.append(b[2].to(args.device))
                if task_name == "code_search":
                    nl_seq = b[1].to(args.device, non_blocking=True)
                    cs_batch_size = code_seq.size(0)
                    all_seqs.append(nl_seq)
                    all_task_ids.append(b[2].to(args.device))
                else:
                    binary_data.append((task_name, b[1].to(args.device).float()))
            all_seqs = _pad_seqs(all_seqs, pad_id)
            with autocast("cuda", dtype=args.autocast_dtype):
                outputs = model(torch.cat(all_seqs, 0), torch.cat(all_task_ids, 0))
                for task_name, labels in binary_data:
                    losses_per_task[task_name].append(
                        bce(outputs[task_name].squeeze(), labels.squeeze()).item()
                    )
                if cs_batch_size > 0:
                    cs_vecs = outputs["code_search"]
                    scores  = torch.einsum("ab,cb->ac", cs_vecs[cs_batch_size:], cs_vecs[:cs_batch_size])
                    losses_per_task["code_search"].append(
                        ce(scores * 20, torch.arange(cs_batch_size, device=scores.device)).item()
                    )
    model.train()
    return {k: float(np.mean(v)) if v else 1.0 for k, v in losses_per_task.items()}


# ================== GRADNORM ==================

def _gradnorm_step(m, per_task_losses, gn_weights, gn_initial_losses, active_tasks, alpha):
    """One GradNorm weight update (Chen et al., NeurIPS 2018), DataParallel-compatible.

    Uses detached gradient norms (no create_graph) so the forward graph is freed
    after this call and DataParallel replicas on all GPUs can be properly cleaned up.
    gn_weights are updated via a direct proportional rule instead of optimizer.step().
    """
    ref_param = None
    for p in m.encoder.parameters():
        if p.requires_grad:
            ref_param = p
    if ref_param is None:
        return

    for task_name, loss in per_task_losses:
        gn_initial_losses.setdefault(task_name, loss.item())

    # Compute per-task gradient norms (detached — no second-order graph)
    norms = []
    for task_name, task_loss in per_task_losses:
        i = active_tasks.index(task_name)
        w_i = gn_weights[i].abs().item()
        g = torch.autograd.grad(
            w_i * task_loss, ref_param,
            retain_graph=True, create_graph=False, allow_unused=True,
        )[0]
        norms.append(g.detach().norm(2) if g is not None else ref_param.new_zeros([]))
    norms = torch.stack(norms)          # all detached — graph can be freed after this
    mean_norm = norms.mean()

    curr = torch.tensor([l.item() for _, l in per_task_losses], device=gn_weights.device)
    init = torch.tensor(
        [gn_initial_losses[t] for t, _ in per_task_losses], device=gn_weights.device
    )
    r = curr / init.clamp(min=1e-8)
    r_hat = (r / r.mean().clamp(min=1e-8)).pow(alpha)
    targets = (mean_norm * r_hat).detach()

    # Direct proportional update: push weights toward equalising gradient norms
    with torch.no_grad():
        errors = norms - targets           # positive → norm too large → reduce weight
        gn_weights.data -= 0.01 * errors * gn_weights.data.abs()
        gn_weights.clamp_(min=1e-4)
        gn_weights.data = gn_weights * len(active_tasks) / gn_weights.sum()


# ================== GRADIENT CONFLICT METRICS ==================

def _compute_gradient_conflicts(args, model, train_sets, active_keys):
    """Compute pairwise cosine similarity and L2 norm of per-task gradients on shared encoder params.

    One batch per task is run independently; gradients are collected for all
    encoder parameters that require grad. Positive similarity = tasks aligned;
    negative = tasks conflicting (pulling encoder in opposite directions).
    Returns:
        conflicts  — {(task_i, task_j): cosine_sim} for all i < j pairs
        grad_norms — {task_name: l2_norm}
    """
    m = model.module if args.n_gpu > 1 else model
    pad_id = m.pad_token_id
    shared_params = [p for p in m.encoder.parameters() if p.requires_grad]
    if not shared_params:
        return {}

    concat = ConcatDataset([ds for _, ds in train_sets])
    tmp_loader = DataLoader(
        dataset=concat,
        sampler=BatchSchedulerSampler(dataset=concat, batch_size=args.train_batch_size),
        batch_size=args.train_batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn  = CrossEntropyLoss()
    n_tasks = len(active_keys)

    data_iter = iter(tmp_loader)
    try:
        batches = [next(data_iter) for _ in range(n_tasks)]
    except StopIteration:
        return {}

    model.eval()
    task_grads = {}

    for b in batches:
        task_name = CONFIG_TASKS[int(b[2][0].item())]
        model.zero_grad()
        code_seq = b[0].to(args.device, non_blocking=True)
        task_ids = b[2].to(args.device)

        with autocast("cuda", dtype=args.autocast_dtype):
            if task_name == "code_search":
                nl_seq = b[1].to(args.device, non_blocking=True)
                bs = code_seq.size(0)
                code_p, nl_p = _pad_seqs([code_seq, nl_seq], pad_id)
                vecs   = model(torch.cat([code_p, nl_p], 0), task_ids.repeat(2))["code_search"]
                scores = torch.einsum("ab,cb->ac", vecs[bs:], vecs[:bs])
                loss   = ce_fn(scores * 20, torch.arange(bs, device=scores.device))
            else:
                labels = b[1].to(args.device, non_blocking=True).float()
                logit  = model(code_seq, task_ids)[task_name]
                loss   = bce_fn(logit.squeeze(), labels.squeeze())

        loss.backward()

        grads = []
        for p in shared_params:
            g = p.grad.detach().float().view(-1) if p.grad is not None else p.new_zeros(p.numel()).float()
            grads.append(g)
        task_grads[task_name] = torch.cat(grads)

    model.zero_grad()
    model.train()

    grad_norms = {t: round(g.norm(2).item(), 6) for t, g in task_grads.items()}

    conflicts = {}
    for i, ti in enumerate(active_keys):
        for j, tj in enumerate(active_keys):
            if j <= i:
                continue
            gi, gj = task_grads.get(ti), task_grads.get(tj)
            if gi is None or gj is None:
                continue
            cos_sim = F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0)).item()
            conflicts[(ti, tj)] = round(cos_sim, 4)

    return conflicts, grad_norms


# ================== TRAIN ==================
def train(args, model, tokenizer):
    """Train on any subset of tasks (1–4) selected via flags/CSV."""
    active_keys = _resolve_active_tasks(args)

    # Build train loaders (concat into one with scheduler)
    train_sets = _build_split_dataloaders(tokenizer, args, active_keys, split="train")
    dataloader = _concat_train_loader(train_sets, args)

    # Build eval & test loaders per task
    eval_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="eval"))
    test_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="test"))

    # Optimizer/scheduler
    # log_sigma2 needs a separate, higher lr because Adam normalizes gradient magnitudes:
    # all sigma entries would otherwise move at identical rates regardless of task loss
    # magnitude, defeating the purpose of uncertainty weighting. A 100× lr here lets
    # sigma converge within the same number of epochs as the task heads.
    use_temp_sampling = getattr(args, 'sampling_temperature', 0.0) > 0
    # Temperature mode: one batch = one optimizer step → steps_per_epoch = len(dataloader).
    # Round-robin mode: n_tasks batches are consumed per step → divide by n_tasks.
    steps_per_epoch = len(dataloader) if use_temp_sampling else len(dataloader) // len(active_keys)

    m_ref = model.module if args.n_gpu > 1 else model
    sigma_ids = {id(m_ref.log_sigma2)}
    main_params  = [p for p in model.parameters() if id(p) not in sigma_ids]
    sigma_params = [p for p in model.parameters() if id(p) in sigma_ids]
    param_groups = [{"params": main_params, "lr": args.learning_rate}]
    if args.loss_weighting == "uncertainty" and sigma_params:
        param_groups.append({"params": sigma_params, "lr": args.learning_rate * 10})
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.01)
    # steps_per_epoch already accounts for consuming n_tasks batches per step;
    # len(dataloader) counts raw batches (4× more), which would make warmup 4× too long.
    max_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * 0.1),
        num_training_steps=max_steps
    )
    scaler = GradScaler("cuda", enabled=(args.autocast_dtype == torch.float16))

    # Normalized static weights: estimate initial loss per task, then freeze w_i = 1/L_i.
    if args.loss_weighting == "normalized":
        logger.info("Estimating initial losses for static weight normalization (50 batches)...")
        init_losses = _estimate_initial_losses(args, model, train_sets, active_keys, n_batches=50)
        raw_w = {k: 1.0 / max(v, 1e-8) for k, v in init_losses.items()}
        w_sum = sum(raw_w.values())
        static_weights = {k: v * len(active_keys) / w_sum for k, v in raw_w.items()}
        logger.info("  Initial losses : %s", {k: f"{v:.4f}" for k, v in init_losses.items()})
        logger.info("  Static weights : %s", {k: f"{v:.4f}" for k, v in static_weights.items()})
    else:
        static_weights = None

    # GradNorm: separate weight parameter + optimizer, not part of the main model.
    # For other modes these are None and never touched.
    if args.loss_weighting == "gradnorm":
        gn_weights = torch.ones(len(active_keys), device=args.device)
        gn_initial_losses: dict = {}
    else:
        gn_weights = gn_initial_losses = None

    # FAMO (Liu et al., NeurIPS 2023): log-weight vector z; w = softmax(z).
    # z is updated each step from log-ratios of consecutive losses — tasks
    # that reduce their loss faster get less weight, stalling tasks get more.
    # No extra backward pass needed; uses only the scalars already collected.
    if args.loss_weighting == "famo":
        famo_z = torch.zeros(len(active_keys), device=args.device)
        famo_prev_losses: dict = {}   # task_name → loss scalar from previous step
    else:
        famo_z = famo_prev_losses = None

    logger.info("=" * 70)
    logger.info("***** Running Training *****")
    logger.info("  Model         : %s", args.model_name_or_path)
    logger.info("  PEFT          : %s", args.peft_module)
    logger.info("  Loss weighting: %s", args.loss_weighting)
    logger.info("  Epochs        : %d", args.num_train_epochs)
    logger.info("  Batch size    : %d  |  LR: %.2e  |  GPUs: %d",
                args.train_batch_size, args.learning_rate, args.n_gpu)
    logger.info("  Steps/epoch   : ~%d  |  Total steps: ~%d",
                steps_per_epoch, steps_per_epoch * args.num_train_epochs)
    logger.info("  Tasks & train sizes:")
    for k, ds in train_sets:
        logger.info("    %-22s  %d examples", k, len(ds))
    logger.info("=" * 70)

    model.zero_grad()

    # Per-task BCE: compute pos_weight for imbalanced binary tasks.
    # pos_weight = neg/pos so the positive-class gradient is upscaled to match
    # the negative-class contribution — prevents "predict all-negative" collapse.
    _task_bce: dict = {}
    for key, ds in train_sets:
        if TASK_REGISTRY[key]["type"] != "binary":
            continue
        labels = [ex.label for ex in ds.examples]
        pos = sum(labels)
        neg = len(labels) - pos
        if pos > 0 and (neg / pos) > 2.0:
            pw = neg / pos
            _task_bce[key] = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pw], device=args.device, dtype=torch.float32)
            )
            logger.info("  pos_weight %-22s = %.1f  (pos=%d  neg=%d)", key, pw, int(pos), int(neg))
        else:
            _task_bce[key] = nn.BCEWithLogitsLoss()

    bce = nn.BCEWithLogitsLoss()   # used by eval helpers (no pos_weight for eval loss)

    def _bce(task_name, logits, labels):
        return _task_bce.get(task_name, bce)(logits, labels)

    ce = CrossEntropyLoss()

    best_score = -np.inf
    train_results, validation_results = {}, {}
    early_stopper = MultiTaskEarlyStopper(active_keys, patience=4)

    cumulative_tokens = 0   # running total across all epochs

    for epoch in range(args.num_train_epochs):
        torch.cuda.empty_cache()
        losses_per_task = {k: [] for k in active_keys}
        total_losses = []
        tokens_per_task = {k: 0 for k in active_keys}  # non-padding tokens per task this epoch

        n_tasks = len(active_keys)
        step = 0

        model.train()

        if use_temp_sampling:
            # ---- Temperature-based sampling: one batch → one optimizer step ----
            pad_id = (model.module if args.n_gpu > 1 else model).pad_token_id
            for batch in dataloader:
                task_name = CONFIG_TASKS[int(batch[2][0].item())]
                code_seq  = batch[0].to(args.device, non_blocking=True)
                task_ids  = batch[2].to(args.device)

                with autocast("cuda", dtype=args.autocast_dtype):
                    if task_name == "code_search":
                        nl_seq = batch[1].to(args.device, non_blocking=True)
                        bs = code_seq.size(0)
                        code_p, nl_p = _pad_seqs([code_seq, nl_seq], pad_id)
                        vecs   = model(torch.cat([code_p, nl_p], 0),
                                       task_ids.repeat(2))["code_search"]
                        scores = torch.einsum("ab,cb->ac", vecs[bs:], vecs[:bs])
                        loss   = ce(scores * 20, torch.arange(bs, device=scores.device))
                    else:
                        labels = batch[1].to(args.device, non_blocking=True).float()
                        logit  = model(code_seq, task_ids)[task_name]
                        loss   = _bce(task_name, logit.squeeze(), labels.squeeze())

                losses_per_task[task_name].append(loss.item())
                # Count non-padding tokens (code + NL for code_search)
                tokens_per_task[task_name] += int((code_seq != pad_id).sum().item())
                if task_name == "code_search":
                    tokens_per_task[task_name] += int((nl_seq != pad_id).sum().item())

                m = model.module if args.n_gpu > 1 else model

                if args.loss_weighting == "uncertainty":
                    s = m.log_sigma2[active_keys.index(task_name)].clamp(min=-2.0, max=0.693)
                    total_loss = torch.exp(-s) * loss + 0.5 * s
                elif args.loss_weighting == "gradnorm":
                    w = gn_weights[active_keys.index(task_name)].detach().abs()
                    total_loss = w * loss
                elif args.loss_weighting == "normalized":
                    total_loss = static_weights[task_name] * loss
                elif args.loss_weighting == "famo":
                    w = torch.softmax(famo_z, dim=0)[active_keys.index(task_name)]
                    total_loss = w * loss
                else:
                    total_loss = loss

                total_losses.append(total_loss.item())

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # FAMO per-task update: track each task's loss trajectory independently
                if args.loss_weighting == "famo":
                    curr_loss = loss.item()
                    task_idx  = active_keys.index(task_name)
                    if task_name in famo_prev_losses:
                        with torch.no_grad():
                            curr_t = torch.tensor([curr_loss], device=args.device)
                            prev_t = torch.tensor([famo_prev_losses[task_name]], device=args.device)
                            delta  = (torch.log(curr_t.clamp(min=1e-8))
                                      - torch.log(prev_t.clamp(min=1e-8)))
                            famo_z[task_idx] -= args.famo_gamma * delta
                    famo_prev_losses[task_name] = curr_loss

                step += 1
                if step % 32 == 0:
                    lr_now   = optimizer.param_groups[0]["lr"]
                    task_str = "  ".join(
                        f"{k}={np.mean(v):.4f}" for k, v in losses_per_task.items() if v
                    )
                    logger.info(
                        "E%d/%d  step %d/%d  total=%.4f  lr=%.2e  |  %s",
                        epoch + 1, args.num_train_epochs,
                        step, steps_per_epoch,
                        float(np.mean(total_losses)), lr_now, task_str,
                    )

        else:
            # ---- Round-robin: n_tasks batches per optimizer step ----
            data_iter = iter(dataloader)
            while True:
                try:
                    batches = [next(data_iter) for _ in range(n_tasks)]
                except StopIteration:
                    break

                # --------- BUILD COMBINED BATCH ----------
                # Concatenate all task inputs into one tensor for a single encoder pass.
                # For code_search: code inputs come first, NL appended immediately after;
                # cs_batch_size tells us where code ends and NL begins in the output.
                all_seqs, all_task_ids = [], []
                binary_data = []   # (task_name, labels)
                cs_batch_size = 0

                for b in batches:
                    task_name = CONFIG_TASKS[int(b[2][0].item())]
                    code_seq  = b[0].to(args.device, non_blocking=True)
                    all_seqs.append(code_seq)
                    all_task_ids.append(b[2].to(args.device))
                    tokens_per_task[task_name] += int((code_seq != pad_id).sum().item())
                    if task_name == "code_search":
                        nl_seq = b[1].to(args.device, non_blocking=True)
                        cs_batch_size = code_seq.size(0)
                        all_seqs.append(nl_seq)
                        all_task_ids.append(b[2].to(args.device))   # same task_id for NL
                        tokens_per_task[task_name] += int((nl_seq != pad_id).sum().item())
                    else:
                        binary_data.append((task_name, b[1].to(args.device, non_blocking=True).float()))

                # --------- ONE FORWARD PASS ----------
                # Sequences may have different lengths (code_length vs nl_length for
                # code_search NL). Pad to the same length before the single encoder pass.
                pad_id = (model.module if args.n_gpu > 1 else model).pad_token_id
                all_seqs = _pad_seqs(all_seqs, pad_id)
                with autocast("cuda", dtype=args.autocast_dtype):
                    outputs = model(torch.cat(all_seqs, 0), torch.cat(all_task_ids, 0))

                    per_task_losses = []
                    for task_name, labels in binary_data:
                        loss = _bce(task_name, outputs[task_name].squeeze(), labels.squeeze())
                        per_task_losses.append((task_name, loss))

                    if cs_batch_size > 0:
                        cs_vecs    = outputs["code_search"]
                        code_vecs  = cs_vecs[:cs_batch_size]
                        nl_vecs    = cs_vecs[cs_batch_size:]
                        scores     = torch.einsum("ab,cb->ac", nl_vecs, code_vecs)
                        loss       = ce(scores * 20, torch.arange(cs_batch_size, device=scores.device))
                        per_task_losses.append(("code_search", loss))

                # --------- COMBINE LOSSES WITH TASK WEIGHTS ----------
                m = model.module if args.n_gpu > 1 else model

                if args.loss_weighting == "gradnorm":
                    _gradnorm_step(m, per_task_losses, gn_weights,
                                   gn_initial_losses, active_keys, args.gradnorm_alpha)
                    optimizer.zero_grad()

                total_loss = 0.0
                for task_name, loss in per_task_losses:
                    losses_per_task[task_name].append(loss.item())
                    if args.loss_weighting == "uncertainty":
                        s = m.log_sigma2[active_keys.index(task_name)].clamp(min=-2.0, max=0.693)
                        total_loss = total_loss + torch.exp(-s) * loss + 0.5 * s
                    elif args.loss_weighting == "gradnorm":
                        w = gn_weights[active_keys.index(task_name)].detach().abs()
                        total_loss = total_loss + w * loss
                    elif args.loss_weighting == "normalized":
                        total_loss = total_loss + static_weights[task_name] * loss
                    elif args.loss_weighting == "famo":
                        w = torch.softmax(famo_z, dim=0)[active_keys.index(task_name)]
                        total_loss = total_loss + w * loss
                    else:  # uniform
                        total_loss = total_loss + loss
                # famo weights (softmax) already sum to 1 → no division needed
                if args.loss_weighting in ("uniform", "normalized"):
                    total_loss = total_loss / max(len(per_task_losses), 1)
                total_losses.append(total_loss.item())

                # --------- BACKWARD + STEP (scaled if fp16, no-op scale if bf16) ----------
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)  # so clipping sees true grads
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # FAMO z update: δ_i = log(L_i^t) − log(L_i^{t-1})
                # tasks reducing loss fast get less weight next step; stalling tasks get more.
                if args.loss_weighting == "famo":
                    curr_scalars = {k: l.item() for k, l in per_task_losses}
                    if famo_prev_losses:
                        with torch.no_grad():
                            curr = torch.tensor([curr_scalars[k] for k in active_keys], device=args.device)
                            prev = torch.tensor([famo_prev_losses[k] for k in active_keys], device=args.device)
                            delta = torch.log(curr.clamp(min=1e-8)) - torch.log(prev.clamp(min=1e-8))
                            famo_z -= args.famo_gamma * (delta - delta.mean())
                    famo_prev_losses.update(curr_scalars)

                step += 1
                if step % 32 == 0:
                    lr_now  = optimizer.param_groups[0]["lr"]
                    task_str = "  ".join(
                        f"{k}={np.mean(v):.4f}" for k, v in losses_per_task.items() if v
                    )
                    logger.info(
                        "E%d/%d  step %d/%d  total=%.4f  lr=%.2e  |  %s",
                        epoch + 1, args.num_train_epochs,
                        step, steps_per_epoch,
                        float(np.mean(total_losses)), lr_now, task_str,
                    )

        # ---- Epoch train summary ----
        epoch_tokens = sum(tokens_per_task.values())
        cumulative_tokens += epoch_tokens
        train_results.setdefault('total_train_loss', []).append(round(np.mean(total_losses), 4))
        train_results.setdefault('tokens_epoch', []).append(epoch_tokens)
        train_results.setdefault('tokens_cumulative', []).append(cumulative_tokens)
        for k in active_keys:
            if losses_per_task[k]:
                train_results.setdefault(f'{k}_train_loss', []).append(round(np.mean(losses_per_task[k]), 4))
            train_results.setdefault(f'{k}_tokens', []).append(tokens_per_task[k])

        m = model.module if args.n_gpu > 1 else model
        if args.loss_weighting == "uncertainty":
            task_weights = m.get_task_weights().tolist()
        elif args.loss_weighting == "gradnorm":
            task_weights = gn_weights.tolist()
        elif args.loss_weighting == "normalized":
            task_weights = [static_weights[k] for k in active_keys]
        elif args.loss_weighting == "famo":
            task_weights = torch.softmax(famo_z, dim=0).tolist()
        else:
            task_weights = [1.0 / len(active_keys)] * len(active_keys)

        logger.info("-" * 70)
        logger.info("Epoch %d/%d — Train Summary", epoch + 1, args.num_train_epochs)
        logger.info("  %-22s  %8s  %8s  %12s", "Task", "Loss", "Weight", "Tokens")
        logger.info("  %-22s  %8s  %8s  %12s", "----", "----", "------", "------")
        for k, w in zip(active_keys, task_weights):
            loss_val = np.mean(losses_per_task[k]) if losses_per_task[k] else float("nan")
            logger.info("  %-22s  %8.4f  %8.4f  %12d", k, loss_val, w, tokens_per_task[k])
        logger.info("  %-22s  %8.4f  %8s  %12d  (epoch)  cumulative=%d",
                    "TOTAL", np.mean(total_losses), "", epoch_tokens, cumulative_tokens)
        logger.info("-" * 70)

        # ===== Gradient norms + conflict (cosine similarity between per-task gradients) =====
        if len(active_keys) > 1:
            conflicts, grad_norms = _compute_gradient_conflicts(args, model, train_sets, active_keys)
            if grad_norms:
                logger.info("  Gradient Norms (L2, shared encoder params):")
                for t, norm in grad_norms.items():
                    train_results.setdefault(f'{t}_grad_norm', []).append(norm)
                    logger.info("    %-22s  grad_norm=%10.6f", t, norm)
                logger.info("-" * 70)
            if conflicts:
                logger.info("  Gradient Conflict Matrix (cosine similarity of encoder grads):")
                logger.info("  %-22s  %-22s  %10s  %s", "Task A", "Task B", "cos_sim", "")
                for (ti, tj), cos_sim in conflicts.items():
                    tag = "CONFLICT" if cos_sim < 0 else "aligned"
                    train_results.setdefault(f'conflict_{ti}_vs_{tj}', []).append(cos_sim)
                    logger.info("  %-22s  %-22s  %+10.4f  [%s]", ti, tj, cos_sim, tag)
                logger.info("-" * 70)

        # ===== Validation =====
        eval_results = evaluate(args, model, tokenizer, eval_loaders)
        for k, res in eval_results.items():
            update_validation_results(res, validation_results)

        if early_stopper.early_stop(validation_results):
            logger.info("Early stopping triggered at epoch %d — halting training.", epoch + 1)
            break

        # Combined score
        scores = []
        for k in active_keys:
            if TASK_REGISTRY[k]["type"] == "binary":
                scores.append(eval_results[k]["f1"])
            else:
                scores.append(eval_results[k]["mrr"])
        combined = float(np.mean(scores))

        _log_metrics_table("Eval", epoch + 1, eval_results, combined)

        # Save best checkpoint + run tests
        if combined >= best_score:
            best_score = combined
            save_trainable_params(model, f"./adapters/{args.output_model_name}/model.bin")
            logger.info("  >> New best (%.4f) — checkpoint saved.", best_score)

            test_res = test_model(args, model, tokenizer, test_loaders)
            _log_metrics_table("Test", epoch + 1, test_res)

    # Final test after training
    test_res = test_model(args, model, tokenizer, test_loaders)
    _log_metrics_table("Final Test", args.num_train_epochs, test_res)

    # ===== Gradient conflict summary across all epochs =====
    conflict_keys = [(ti, tj) for i, ti in enumerate(active_keys)
                     for j, tj in enumerate(active_keys) if j > i]
    conflict_data = {(ti, tj): train_results.get(f'conflict_{ti}_vs_{tj}', [])
                     for ti, tj in conflict_keys}
    if any(v for v in conflict_data.values()):
        logger.info("=" * 70)
        logger.info("Gradient Conflict Summary (aggregated over %d epochs)", epoch + 1)
        logger.info("  %-22s  %-22s  %8s  %6s  %8s  %s",
                    "Task A", "Task B", "mean", "std", "neg_freq", "verdict")
        logger.info("  %-22s  %-22s  %8s  %6s  %8s  %s",
                    "------", "------", "----", "---", "--------", "-------")
        for (ti, tj), vals in conflict_data.items():
            if not vals:
                continue
            arr = np.array(vals)
            mean_sim  = arr.mean()
            std_sim   = arr.std()
            neg_freq  = (arr < 0).mean()   # fraction of epochs with conflict
            if mean_sim < -0.05:
                verdict = "CONFLICT"
            elif mean_sim > 0.05:
                verdict = "aligned"
            else:
                verdict = "neutral"
            logger.info("  %-22s  %-22s  %+8.4f  %6.4f  %7.1f%%  [%s]",
                        ti, tj, mean_sim, std_sim, neg_freq * 100, verdict)
        logger.info("=" * 70)

    return train_results, validation_results


# ================== EVALUATION PRIMITIVES ==================

def _encode_binary(args, model, dl, task_name):
    """Collect logits and labels for a binary-classification dataloader.
    Returns (logits, labels, eval_loss) — all CPU tensors except eval_loss (float).
    """
    bce = nn.BCEWithLogitsLoss()
    logits_list, labels_list, loss_sum, nb = [], [], 0.0, 0
    with torch.no_grad():
        for batch in dl:
            inputs   = batch[0].to(args.device, non_blocking=True)
            labels   = batch[1].to(args.device, non_blocking=True).float().squeeze()
            task_ids = batch[2].to(args.device)
            with autocast("cuda", dtype=args.autocast_dtype):
                logit = model(inputs, task_ids)[task_name].squeeze()
                loss  = bce(logit, labels)
            loss_sum += float(loss.mean().item())
            nb += 1
            logits_list.append(logit.float().detach().cpu().view(-1))
            labels_list.append(labels.float().detach().cpu().view(-1))
    return torch.cat(logits_list), torch.cat(labels_list), round(loss_sum / max(nb, 1), 4)


def _binary_metrics(task_name, logits, labels, eval_loss=None):
    """Compute classification metrics from CPU logits/labels tensors."""
    preds     = (torch.sigmoid(logits) > 0.5).to(torch.int32).numpy()
    labels_np = labels.to(torch.int32).numpy()
    result = {
        "task":      task_name,
        "acc":       round(float(np.mean(labels_np == preds)), 4),
        "f1":        round(f1_score(labels_np, preds, average="binary"), 4),
        "recall":    round(recall_score(labels_np, preds), 4),
        "precision": round(precision_score(labels_np, preds, zero_division=0), 4),
    }
    if eval_loss is not None:
        result["eval_loss"] = eval_loss
    return result


def _encode_retrieval(args, model, dl):
    """Encode code+NL pairs for a code_search dataloader.
    Returns (code_vecs, nl_vecs) as numpy arrays, one row per example.
    """
    pad_id = (model.module if hasattr(model, "module") else model).pad_token_id
    code_vecs, nl_vecs = [], []
    with torch.no_grad():
        with autocast("cuda", dtype=args.autocast_dtype):
            for b in dl:
                bs   = b[0].size(0)
                code = b[0].to(args.device)
                nl   = b[1].to(args.device)
                # code_length and nl_length may differ — pad to same length
                code, nl = _pad_seqs([code, nl], pad_id)
                combined = torch.cat([code, nl], 0)
                task_ids = b[2].repeat(2).to(args.device)
                vecs     = model(combined, task_ids)["code_search"]
                code_vecs.append(vecs[:bs].cpu().numpy())
                nl_vecs.append(vecs[bs:].cpu().numpy())
    return np.concatenate(code_vecs, 0), np.concatenate(nl_vecs, 0)


def _encode_codebase(args, model, dl):
    """Encode only the code side of a codebase DataLoader (NL side is empty/ignored)."""
    code_vecs = []
    with torch.no_grad():
        with autocast("cuda", dtype=args.autocast_dtype):
            for b in dl:
                code_ids = b[0].to(args.device)
                task_ids = b[2].to(args.device)
                vecs     = model(code_ids, task_ids)["code_search"]
                code_vecs.append(vecs.cpu().numpy())
    return np.concatenate(code_vecs, 0)


def _retrieval_mrr(code_vecs, nl_vecs, code_urls, nl_urls):
    """Compute MRR given encoded vectors and URL lists."""
    scores   = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1)[:, ::-1]
    ranks = []
    for url, row in zip(nl_urls, sort_ids):
        rank, found = 0, False
        for idx in row[:1000]:
            if not found:
                rank += 1
            if code_urls[idx] == url:
                found = True
        ranks.append(1 / rank if found else 0)
    return round(float(np.mean(ranks)), 4)


def _log_metrics_table(phase, epoch, results, combined=None):
    """Print a compact aligned table of per-task metrics to the logger.

    phase    — label shown in the header ('Eval', 'Test', 'Final Test', …)
    epoch    — epoch number (int)
    results  — {task_name: metrics_dict} as returned by evaluate / test_model
    combined — optional combined score float to show in the footer
    """
    HDR = f"  {'Task':<22}  {'Acc':>7}  {'F1':>7}  {'Recall':>8}  {'Precision':>10}  {'Loss':>7}  {'MRR':>7}"
    SEP = "  " + "-" * 68
    logger.info("-" * 70)
    logger.info("%s — Epoch %d", phase, epoch)
    logger.info(HDR)
    logger.info(SEP)
    for task, res in results.items():
        if "mrr" in res:
            logger.info("  %-22s  %7s  %7s  %8s  %8s  %7s  %7.4f",
                        task, "-", "-", "-", "-", "-", res["mrr"])
        else:
            loss_str = f"{res['eval_loss']:.4f}" if "eval_loss" in res else "-"
            logger.info("  %-22s  %7.4f  %7.4f  %8.4f  %10.4f  %7s  %7s",
                        task,
                        res.get("acc", float("nan")),
                        res.get("f1",  float("nan")),
                        res.get("recall",    float("nan")),
                        res.get("precision", float("nan")),
                        loss_str, "-")
    logger.info(SEP)
    if combined is not None:
        logger.info("  %-22s  combined = %.4f", "", combined)
    logger.info("-" * 70)


# ================== EVALUATE / TEST ==================

def evaluate(args, model, tokenizer, loaders_dict):
    """Evaluate all active tasks (with loss).
    loaders_dict: {task_name: DataLoader}
    Returns:      {task_name: metrics_dict}
    """
    model.eval()
    results = {}
    for task, dl in loaders_dict.items():
        if TASK_REGISTRY[task]["type"] == "binary":
            logits, labels, loss = _encode_binary(args, model, dl, task)
            results[task] = _binary_metrics(task, logits, labels, eval_loss=loss)
        else:
            code_vecs, nl_vecs = _encode_retrieval(args, model, dl)
            nl_urls = [ex.url for ex in dl.dataset.examples]
            if getattr(args, 'codebase_file', None):
                cb_ds  = TextDataset_code_search(tokenizer, args, args.codebase_file)
                cb_dl  = DataLoader(cb_ds, sampler=SequentialSampler(cb_ds),
                                    batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
                cb_code_vecs = _encode_codebase(args, model, cb_dl)
                code_urls    = [ex.url for ex in cb_ds.examples]
            else:
                cb_code_vecs = code_vecs
                code_urls    = nl_urls
            results[task] = {"task": task, "mrr": _retrieval_mrr(cb_code_vecs, nl_vecs, code_urls, nl_urls)}
    return results


def test_model(args, model, tokenizer, loaders_dict):
    """Test all active tasks (no loss).
    loaders_dict: {task_name: DataLoader}
    Returns:      {task_name: metrics_dict}
    """
    model.eval()
    results = {}
    for task, dl in loaders_dict.items():
        logger.info("  Testing %s (%d examples)", task, len(dl.dataset))
        if TASK_REGISTRY[task]["type"] == "binary":
            logits, labels, _ = _encode_binary(args, model, dl, task)
            results[task] = _binary_metrics(task, logits, labels)
        else:
            code_vecs, nl_vecs = _encode_retrieval(args, model, dl)
            nl_urls = [ex.url for ex in dl.dataset.examples]
            if getattr(args, 'codebase_file', None):
                cb_ds  = TextDataset_code_search(tokenizer, args, args.codebase_file)
                cb_dl  = DataLoader(cb_ds, sampler=SequentialSampler(cb_ds),
                                    batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
                cb_code_vecs = _encode_codebase(args, model, cb_dl)
                code_urls    = [ex.url for ex in cb_ds.examples]
            else:
                cb_code_vecs = code_vecs
                code_urls    = nl_urls
            results[task] = {"task": task, "mrr": _retrieval_mrr(cb_code_vecs, nl_vecs, code_urls, nl_urls)}
    return results


# ================== MAIN ==================
def main():
    """Main entry point for training, evaluation, and testing."""
    parser = argparse.ArgumentParser()

    # --- original args ---
    parser.add_argument("--output_dir", default='./', type=str)

    # data files
    parser.add_argument("--train_data_file_vul", default="./datasets/dataset_vulnerabilty/train.jsonl", type=str)
    parser.add_argument("--eval_data_file_vul",  default="./datasets/dataset_vulnerabilty/valid.jsonl",  type=str)
    parser.add_argument("--test_data_file_vul",  default="./datasets/dataset_vulnerabilty/test.jsonl",   type=str)

    parser.add_argument("--train_data_file_clone", default="./datasets/dataset_clone/train.txt", type=str)
    parser.add_argument("--eval_data_file_clone",  default="./datasets/dataset_clone/valid.txt",  type=str)
    parser.add_argument("--test_data_file_clone",  default="./datasets/dataset_clone/test.txt",   type=str)

    parser.add_argument("--train_data_file_flaky", default="./datasets/dataset_flakytest/train.json", type=str)
    parser.add_argument("--eval_data_file_flaky",  default="./datasets/dataset_flakytest/valid.json", type=str)
    parser.add_argument("--test_data_file_flaky",  default="./datasets/dataset_flakytest/test.json",  type=str)

    parser.add_argument("--train_data_file_CodeSearch", default="./datasets/code_search/train.jsonl", type=str)
    parser.add_argument("--eval_data_file_CodeSearch",  default="./datasets/code_search/valid.jsonl",  type=str)
    parser.add_argument("--test_data_file_CodeSearch",  default="./datasets/code_search/test.jsonl",   type=str)

    parser.add_argument("--codebase_file", default=None, type=str,
                        help="Codebase jsonl for code search evaluation (1000-sample pool). "
                             "If omitted, MRR is computed over the query set only.")
    parser.add_argument("--data_file_clone", default="./datasets/dataset_clone/data.jsonl", type=str,
                        help="Code pool for clone detection: jsonl with 'idx' and 'func' fields")

    parser.add_argument("--model_name_or_path", default='Qwen/Qwen3-Embedding-4B', type=str)
    parser.add_argument("--peft_module", default="parallel_adapter", type=str,
                        help="adapter | parallel_adapter | lora | prefix | full")
    parser.add_argument("--bottleneck_dim", default=64, type=int, help="Bottleneck dimension for adapter/parallel_adapter")
    parser.add_argument("--lora_r", default=16, type=int, help="Rank for LoRA layers")
    parser.add_argument("--output_model_name", default="qwen_parallel_4tasks", type=str)

    parser.add_argument("--nl_length", default=128, type=int)
    parser.add_argument("--code_length", default=512, type=int)

    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval",  action='store_true')
    parser.add_argument("--do_test",  action='store_true')

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size",  default=8, type=int)

    parser.add_argument("--n_groups", default=3, type=int)
    parser.add_argument("--train_data_rate_vul",         default=1.0, type=float)
    parser.add_argument("--train_data_rate_clone",       default=0.2, type=float)
    parser.add_argument("--train_data_rate_code_search", default=1.0, type=float)
    parser.add_argument("--train_data_rate_flaky",       default=1.0, type=float)

    parser.add_argument("--learning_rate",  default=1e-4, type=float)
    parser.add_argument("--max_grad_norm",  default=1.0,  type=float)
    parser.add_argument("--num_train_epochs", default=3,  type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_device', default='0', type=str,
                        help='GPU id to use when only one GPU is available')

    # --- task selection ---
    parser.add_argument("--tasks", dest="tasks_csv", type=str, default="",
                        help="CSV of task names. Example: --tasks vul_detection,clone_detection")
    parser.add_argument("--train_vul",         action='store_true')
    parser.add_argument("--train_clone",       action='store_true')
    parser.add_argument("--train_code_search", action='store_true')
    parser.add_argument("--train_flaky",       action='store_true')

    # --- loss weighting ---
    parser.add_argument("--loss_weighting", default="normalized", type=str,
                        choices=["uncertainty", "gradnorm", "uniform", "normalized", "famo"],
                        help="MTL loss weighting strategy.\n"
                             "  normalized:  frozen inverse-loss weights computed once before training;\n"
                             "               w_i = 1/L_i(init), normalised so mean weight = 1.\n"
                             "  famo:        Liu et al. (NeurIPS 2023) — softmax weights updated from\n"
                             "               log-ratios of consecutive losses; no extra backward pass.\n"
                             "  uncertainty: Kendall et al. (NeurIPS 2018) — exp(-s)*L + 0.5*s.\n"
                             "  gradnorm:    Chen et al. (NeurIPS 2018) — equalises gradient norms;\n"
                             "               requires extra backward passes (higher memory cost).\n"
                             "  uniform:     equal fixed weights, no learning.")
    parser.add_argument("--gradnorm_alpha", default=1.5, type=float,
                        help="GradNorm asymmetry hyperparameter alpha (default 1.5).")
    parser.add_argument("--famo_gamma", default=0.02, type=float,
                        help="FAMO step size for z update (default 0.02, as in the original paper).")
    # --- task sampling ---
    parser.add_argument("--sampling_temperature", default=0.0, type=float,
                        help="Temperature T for task sampling: prob ∝ N_i^T.\n"
                             "  T=0   → round-robin (BatchSchedulerSampler, default).\n"
                             "  T=0.5 → mT5-style sqrt(N_i); large datasets downscaled.\n"
                             "  T=1   → proportional to dataset size.\n"
                             "  T>1   → super-linear; largest dataset dominates.")

    # --- quick limits for smoke tests ---
    parser.add_argument("--max_train_samples", default=None, type=int,
                        help="Cap each training dataset to N examples. Useful for smoke tests.")
    parser.add_argument("--max_eval_samples", default=None, type=int,
                        help="Cap each eval/test dataset to N examples.")

    torch.cuda.empty_cache()
    args = parser.parse_args()
    set_seed(seed=args.seed)

    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.n_gpu > 1:
        device = torch.device("cuda:0")
    elif args.n_gpu == 1:
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.autocast_dtype = autocast_dtype

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    args.tasks = _resolve_active_tasks(args)
    config.tasks = args.tasks
    config.loss_weighting = args.loss_weighting

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Always use right-padding for encoding — decoder models default to
    # left-padding for generation but that is wrong for mean-pooling encoders.
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        # Decoder models (Qwen, DeepSeek, CodeLlama) have no native pad token.
        # Use eos as pad; model.__init__ picks up pad_token_id from config below.
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure config carries the resolved pad_token_id so MultiTaskModel_MTL
    # can store it at init time without architecture-specific guessing.
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    try:
        base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                               trust_remote_code=True)
    except (ValueError, OSError, RuntimeError) as e:
        logger.warning("AutoModel failed (%s); retrying with AutoModelForSeq2SeqLM.", e)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                                           trust_remote_code=True)

    # ---- Extract encoder for seq2seq models BEFORE injecting adapters ----
    # Must happen first so PEFT targets the encoder directly, not the full
    # encoder-decoder wrapper (which would inject adapters into the decoder too,
    # and break the PeftModel wrapper path for LoRA/Prefix).
    if getattr(config, "is_encoder_decoder", False):
        base_model = base_model.encoder

    # ---- PEFT injection targets, keyed by config.model_type ----
    # Parallel-adapter targets must be whole sub-modules whose input and output
    # are both hidden_size — OpenDelta silently skips projections that have a
    # different input dim (e.g. mlp.down_proj whose input = intermediate_size).
    _PEFT_PATTERNS = {
        # Encoder-only (RoBERTa-style): UniXcoder, CodeBERT, …
        # key added alongside query/value: vulnerability detection relies on
        # learning which tokens attend together, not just query/value projections.
        "roberta":     {"adapter":          ["attention", r"\d+\.output"],
                        "parallel_adapter": ["attention", r"\d+\.output"],
                        "lora":             ["query", "key", "value", "attention.output.dense"]},
        # T5-family (CodeT5+ 770M reports model_type='t5')
        "t5":          {"adapter":          ["SelfAttention", "DenseReluDense"],
                        "parallel_adapter": ["SelfAttention", "DenseReluDense"],
                        "lora":             ["q", "k", "v"]},
        # Encoder-decoder (CodeT5+) — not in PEFT's built-in mapping.
        "codet5p":     {"adapter":          ["mlp", "attn"],
                        "parallel_adapter": ["mlp", "attn"],
                        "lora":             ["attn"]},          # 2B override below
        "llama":       {"adapter":          ["self_attn", "mlp"],
                        "parallel_adapter": ["self_attn", "mlp"],
                        "lora":             ["q_proj", "k_proj", "v_proj"]},
        "mistral":     {"adapter":          ["self_attn", "mlp"],
                        "parallel_adapter": ["self_attn", "mlp"],
                        "lora":             ["q_proj", "k_proj", "v_proj"]},
        # DeepSeek v2/v3 — not in PEFT's built-in mapping.
        "deepseek_v2": {"adapter":          ["self_attn", "mlp"],
                        "parallel_adapter": ["self_attn", "mlp"],
                        "lora":             ["q_proj", "k_proj", "v_proj"]},
        # Qwen2 / Qwen3 — not in PEFT's built-in mapping.
        "qwen2":       {"adapter":          ["self_attn", "mlp"],
                        "parallel_adapter": ["self_attn", "mlp"],
                        "lora":             ["q_proj", "k_proj", "v_proj"]},
        "qwen3":       {"adapter":          ["self_attn", "mlp"],
                        "parallel_adapter": ["self_attn", "mlp"],
                        "lora":             ["q_proj", "k_proj", "v_proj"]},
    }

    model_type = config.model_type.lower()
    if model_type not in _PEFT_PATTERNS:
        if args.peft_module != "full":
            raise ValueError(
                f"No PEFT patterns registered for model_type='{model_type}'. "
                f"Add an entry to _PEFT_PATTERNS or use --peft_module full."
            )
        _entry = {}
    else:
        _entry = dict(_PEFT_PATTERNS[model_type])
        # CodeT5p-2B uses a fused QKV projection — needs more specific LoRA targets
        if model_type == "codet5p" and "2b" in args.model_name_or_path.lower():
            _entry["lora"] = ["attn.qkv_proj", "attn.out_proj"]

    adapter_patterns          = _entry.get("adapter", [])
    parallel_adapter_patterns = _entry.get("parallel_adapter", [])
    lora_patterns             = _entry.get("lora", [])

    if args.peft_module == "parallel_adapter":
        delta_model = ParallelAdapterModel(backbone_model=base_model, modified_modules=parallel_adapter_patterns, bottleneck_dim=args.bottleneck_dim)
        delta_model.freeze_module(exclude=["deltas", "lm_head"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    elif args.peft_module == "adapter":
        delta_model = AdapterModel(backbone_model=base_model, modified_modules=adapter_patterns, bottleneck_dim=args.bottleneck_dim)
        delta_model.freeze_module(exclude=["deltas", "lm_head"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    elif args.peft_module == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            # None → PEFT auto-detects from its built-in mapping (covers roberta,
            # llama, mistral, …). Explicit list required for models not in that
            # mapping (qwen2/3, deepseek_v2, codet5p).
            target_modules=lora_patterns or None,
            lora_dropout=0.1,
            bias="none",
        )
        delta_model = get_peft_model(base_model, lora_config)
        delta_model.print_trainable_parameters()
    elif args.peft_module == "prefix":
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            num_virtual_tokens=20,
        )
        delta_model = get_peft_model(base_model, prefix_config)
        delta_model.print_trainable_parameters()
    elif args.peft_module == "full":
        delta_model = base_model

    # OpenDelta (adapter/parallel_adapter) injects into base_model in-place →
    # pass base_model.  PEFT (lora/prefix) stores params in the PeftModel wrapper
    # itself (prefix encoder lives in delta_model, not base_model) → pass delta_model.
    if args.peft_module in ("lora", "prefix"):
        encoder = delta_model
    else:
        encoder = base_model

    MTLmodel = MultiTaskModel_MTL(encoder, config).to(args.device)

    # NOTE: torch.nn.DataParallel is used for simplicity but has known limitations
    # (GIL contention, GPU-0 memory imbalance). For large-scale training, migrating
    # to DistributedDataParallel (torchrun) is recommended.
    if args.n_gpu > 1:
        MTLmodel = torch.nn.DataParallel(MTLmodel, dim=0)

    logger.info("Model:\n%s", MTLmodel)

    if args.do_train:
        train_results, valid_results = train(args, MTLmodel, tokenizer)
        logger.info("Train results:\n%s", pprint.pformat(train_results))
        logger.info("Validation results:\n%s", pprint.pformat(valid_results))

    # Optional standalone eval/test using the same dynamic helpers
    if args.do_eval:
        eval_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="eval"))
        eval_results = evaluate(args, MTLmodel, tokenizer, eval_loaders)
        logger.info("\n***** Eval results (selected tasks) *****")
        for k, res in eval_results.items():
            logger.info("  [%s] %s", k, str(res))

    if args.do_test:
        test_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="test"))
        test_res = test_model(args, MTLmodel, tokenizer, test_loaders)
        logger.info("\n***** Test results (selected tasks) *****")
        for k, res in test_res.items():
            logger.info("  [%s] %s", k, str(res))


if __name__ == "__main__":
    main()
