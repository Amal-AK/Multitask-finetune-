#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from opendelta import AdapterModel , LoraModel , ParallelAdapterModel , PrefixModel
import argparse
import logging
import os
import pprint
import torch
import numpy as np
from model import *  # expects MultiTaskModel_MTL, dataset classes, etc.
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import torch.nn as nn
import transformers
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss 
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
    RobertaConfig, RobertaTokenizer, RobertaModel,
    AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
)
from utilities import *  
from sklearn.metrics import recall_score, precision_score, f1_score

# ================== ENV & LOGGING ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
global CONFIG_TASKS
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
    Return list[(key, dataset_or_dataloader)] for the requested split, preserving order.
    For 'train' we return datasets (caller concatenates and wraps with BatchSchedulerSampler).
    For 'eval'/'test' we return DataLoaders.
    """
    outputs = []
    for key in active_keys:
        meta = TASK_REGISTRY[key]
        path = getattr(args, meta[f"{split}_arg"])
        ds = meta["dataset"](tokenizer, args, path)
        if split == "train":
            outputs.append((key, ds))
        else:
            dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
            outputs.append((key, dl))
    return outputs

def _concat_train_loader(train_sets, args):
    concat = ConcatDataset([ds for _, ds in train_sets])
    dl = DataLoader(
        dataset=concat,
        sampler=BatchSchedulerSampler(dataset=concat, batch_size=args.train_batch_size),
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    return dl

def _taskid_to_pretty(task_id: int) -> str:
    return CONFIG_TASKS[task_id] if 0 <= task_id < len(CONFIG_TASKS) else str(task_id)



# ================== TRAIN ==================
def train(args, model, tokenizer):
    """ Train on any subset of tasks (2–4) selected via flags/CSV. """
    active_keys = _resolve_active_tasks(args)   # e.g., ["vul", "clone"] or ["vul","clone","code_search"]
    active_types = {k: TASK_REGISTRY[k]["type"] for k in active_keys}

    # Build train loaders (concat into one with scheduler)
    train_sets = _build_split_dataloaders(tokenizer, args, active_keys, split="train")
    dataloader = _concat_train_loader(train_sets, args)

    # Build eval & test loaders per task
    eval_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="eval"))
    test_loaders = dict(_build_split_dataloaders(tokenizer, args, active_keys, split="test"))

    # Optimizer/scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    max_steps = len(dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * 0.1),
        num_training_steps=max_steps
    )
    scaler = GradScaler(enabled=(args is not None and hasattr(args, "autocast_dtype") and args.autocast_dtype == torch.float16))

    logger.info("***** Running training *****")
    for k, ds in train_sets:
        logger.info(f"  Train examples {k} = {len(ds)}")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)

    model.zero_grad()
    bce = nn.BCEWithLogitsLoss()
    ce = CrossEntropyLoss()

    best_score = -np.inf
    train_results, validation_results = {}, {}
    early_stopper = MultiTaskEarlyStopper(active_keys, patience=4)  # use selected tasks

    for epoch in range(args.num_train_epochs):
        torch.cuda.empty_cache()
        losses_per_task = {k: [] for k in active_keys}
        total_losses = []

        data_iter = iter(dataloader)
        n_tasks = len(active_keys)  # take one batch per active task per "step"
        step = 0
        while True:
            batches = []
            try:
                for _ in range(n_tasks):
                    batches.append(next(data_iter))
            except StopIteration:
                break

            model.train()

            per_task_losses = []

            # --------- FORWARD + LOSS (mixed precision) ----------
            with autocast(dtype=args.autocast_dtype):
                for b in batches:
                    task_id = int(b[2][0].item())
                    pretty = _taskid_to_pretty(task_id)
                    is_retrieval = (pretty == "code_search")

                    if not is_retrieval:
                        code_inputs = b[0].to(args.device, non_blocking=True)
                        labels = b[1].to(args.device, non_blocking=True).float()
                        logits = model(code_inputs=code_inputs, task=task_id)
                        loss = bce(logits.squeeze(), labels.squeeze())
                    else:
                        code_inputs = b[0].to(args.device, non_blocking=True)
                        nl_inputs   = b[1].to(args.device, non_blocking=True)
                        code_vec = model(code_inputs=code_inputs, task=task_id)
                        nl_vec   = model(nl_inputs=nl_inputs, task=task_id)
                        scores = torch.einsum("ab,cb->ac", nl_vec.squeeze(), code_vec.squeeze())
                        loss = ce(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

                    per_task_losses.append((task_id, loss))

            # Combine with learned weights (in fp32)
            weights = model.module.task_weights if args.n_gpu > 1 else model.task_weights
            task_weights = torch.softmax(weights, dim=0)

            total_loss = 0.0
            for task_id, loss in per_task_losses:
                if task_id not in args.id2active: 
                    continue
                total_loss = total_loss + loss * task_weights[args.id2active[task_id]]
                key = _taskid_to_pretty(task_id)
                if key in losses_per_task:
                    losses_per_task[key].append(loss.item())
            total_losses.append(total_loss.item())

            # --------- BACKWARD + STEP (scaled if fp16, no-op scale if bf16) ----------
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # so clipping sees true grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()


            step += 1
            if step % 32 == 0:
                logger.info(
                    "Epoch %d Step %d Total %.3f %s",
                    epoch, step, float(np.mean(total_losses)),
                    " ".join([f"{k}:{np.mean(v):.3f}" for k, v in losses_per_task.items() if v])
                )

        # Log epoch train stats
        train_results.setdefault('total_train_loss', []).append(round(np.mean(total_losses), 3))
        for k in active_keys:
            if losses_per_task[k]:
                train_results.setdefault(f'{k}_train_loss', []).append(round(np.mean(losses_per_task[k]), 3))

        # Task weights view
        logger.info("\n***** task weights *****")
        weights_vec = model.module.get_task_weights() if args.n_gpu > 1 else model.get_task_weights()
        print(weights_vec)

        # ===== Validation (dynamic over selected tasks) =====
        logger.info("\n***** Running evaluation *****")
        eval_results = evaluate_dynamic(args, model, eval_loaders)
        for k, res in eval_results.items():
            update_validation_results(res, validation_results)

        # Early stopping over selected tasks
        if early_stopper.early_stop(validation_results):
            print("\nFreezing shared parameters!\n")
            encoder_params = model.module.encoder.parameters() if args.n_gpu > 1 else model.encoder.parameters()
            for p in encoder_params:
                p.requires_grad = False

        # Combined score: average F1 for binary + MRR for retrieval
        scores = []
        for k in active_keys:
            if TASK_REGISTRY[k]["type"] == "binary":
                if "vul" in k :
                    scores.append(eval_results[k]["eval_acc"])
                else :
                    scores.append(eval_results[k]["f1_score"])
            else:
                scores.append(eval_results[k]["eval_mrr"])
        combined = float(np.mean(scores))
        logger.info(f"\n***** Combined eval score (avg over {active_keys}) = {combined:.4f} *****")

        # Save best + run tests (dynamic)
        if combined >= best_score:
            best_score = combined
            names = save_trainable_params(model, "./adapters/"+args.output_model_name+"/model.bin")
            print("Sample of saved keys:", names)
            
            logger.info("\n***** Running Test *****")
            for k, dl in test_loaders.items():
                res = test_dynamic(args, model, k, dl)
                logger.info("  [test:%s] %s", k, str(res))

    # Final test after training
    for k, dl in test_loaders.items():
        res = test_dynamic(args, model, k, dl)
        logger.info("  [test:%s] %s", k, str(res))

    return train_results, validation_results











    

# ================== EVALUATION (VALIDATION) ==================
def evaluate_dynamic(args, model, eval_loaders_dict):
    """
    eval_loaders_dict: { key: dataloader } for active tasks
    Returns: { key: metrics_dict }
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    out = {}

    def eval_binary(dl, pretty_name):
        eval_loss = 0.0
        nb = 0
        logits_list, labels_list = [], []

        with torch.no_grad():
            for batch in dl:
                inputs = batch[0].to(args.device, non_blocking=True)
                label  = batch[1].to(args.device, non_blocking=True).float().squeeze()
                taskid = int(batch[2][0].item())

                with autocast(dtype=args.autocast_dtype):
                    logit  = model(code_inputs=inputs, task=taskid).squeeze()
                    loss   = bce(logit, label)

                eval_loss += float(loss.mean().item()); nb += 1
                # bf16-safe: keep as torch tensors, cast to float32
                logits_list.append(logit.float().detach().cpu().view(-1))
                labels_list.append(label.float().detach().cpu().view(-1))

        # Concatenate as torch tensors
        logits_t = torch.cat(logits_list, dim=0)          # (N,)
        labels_t = torch.cat(labels_list, dim=0)          # (N,)

        # Threshold in torch; convert to numpy afterwards
        probs_t = torch.sigmoid(logits_t)                 # (N,)
        preds_np = (probs_t > 0.5).to(torch.int32).cpu().numpy()
        labels_np = labels_t.to(torch.int32).cpu().numpy()

        return {
            "task": pretty_name,
            "eval_loss": round(eval_loss / max(nb, 1), 4),
            "eval_acc": round(float(np.mean(labels_np == preds_np)), 4),
            "f1_score": round(f1_score(labels_np, preds_np, average="binary"), 4),
            "recall":   round(recall_score(labels_np, preds_np), 4),
            "precision":round(precision_score(labels_np, preds_np, zero_division=0), 4),
        }

    def eval_retrieval(dl):
        # Use same loader for queries & code (matching your previous approach)
        code_vecs, nl_vecs = [], []
        with torch.no_grad():
            with autocast(dtype=args.autocast_dtype):
                for b in dl:
                    nl_inputs = b[1].to(args.device)
                    nl_vec = model(nl_inputs=nl_inputs, task=int(b[2][0].item())).squeeze()
                    if nl_vec.dim() == 1: nl_vec = nl_vec.unsqueeze(0)
                    nl_vecs.append(nl_vec.cpu().numpy())
                for b in dl:
                    code_inputs = b[0].to(args.device)
                    code_vec = model(code_inputs=code_inputs, task=int(b[2][0].item())).squeeze()
                    if code_vec.dim() == 1: code_vec = code_vec.unsqueeze(0)
                    code_vecs.append(code_vec.cpu().numpy())
        code_vecs = np.concatenate(code_vecs, 0); nl_vecs = np.concatenate(nl_vecs, 0)
        scores = np.matmul(nl_vecs, code_vecs.T)
        sort_ids = np.argsort(scores, axis=-1)[:, ::-1]
        nl_urls = [ex.url for ex in dl.dataset.examples]
        code_urls = [ex.url for ex in dl.dataset.examples]
        ranks = []
        for url, row in zip(nl_urls, sort_ids):
            rank = 0; found = False
            for idx in row[:1000]:
                if not found: rank += 1
                if code_urls[idx] == url: found = True
            ranks.append(1/rank if found else 0)
        return {"task": "code_search", "eval_mrr": round(float(np.mean(ranks)), 4)}

    for k, dl in eval_loaders_dict.items():
        t = TASK_REGISTRY[k]["type"]
        pretty = k
        logger.info("  Num examples %s = %d", pretty, len(dl.dataset))
        if t == "binary":
            out[k] = eval_binary(dl, pretty)
        else:
            out[k] = eval_retrieval(dl)
    return out


# ================== TEST (ORIGINAL + DYNAMIC WRAPPER) ==================
def test(args, model, test_dataloader):
    """Binary classification test (kept from your original, slightly cleaned)."""
    logits_list = []
    labels_list = []
    model.eval()

    for batch in test_dataloader:
        inputs = batch[0].to(args.device, non_blocking=True)
        label = batch[1].to(args.device, non_blocking=True).float()

        with torch.no_grad():
            with autocast(dtype=args.autocast_dtype):
                task_name_pretty = _taskid_to_pretty(int(batch[2][0].item()))
                logit = model(code_inputs=inputs, task=int(batch[2][0].item())).squeeze()

        logits_list.append(logit.float().detach().cpu().view(-1))
        labels_list.append(label.float().detach().cpu().view(-1))

    logits_t = torch.cat(logits_list, dim=0)
    labels_t = torch.cat(labels_list, dim=0)

    probs_t = torch.sigmoid(logits_t)
    preds_np = (probs_t > 0.5).to(torch.int32).cpu().numpy()
    labels_np = labels_t.to(torch.int32).cpu().numpy()

    acc = np.mean(labels_np == preds_np)
    recall = recall_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, average="binary")

    return {
        "task": task_name_pretty,
        "test_acc": round(acc, 4),
        "test_f1_score": round(f1, 4),
        "test_recall": round(recall, 4),
        "test_precision": round(precision, 4),
    }




def test_code_search(args, model, query_dataloader , code_dataloader ,eval_when_training=False):
    """Retrieval test (kept from your original, slightly cleaned)."""
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    with torch.no_grad():
        with autocast(dtype=args.autocast_dtype):
            for batch in query_dataloader:  
                nl_inputs = batch[1].to(args.device)
                nl_vec = model(nl_inputs=nl_inputs , task=int(batch[2][0].item())).squeeze()
                if nl_vec.dim() == 1:
                    nl_vec = nl_vec.unsqueeze(0)
                nl_vecs.append(nl_vec.cpu().numpy()) 

            for batch in code_dataloader:
                code_inputs = batch[0].to(args.device)    
                code_vec = model(code_inputs=code_inputs, task=int(batch[2][0].item())).squeeze()
                if code_vec.dim() == 1:
                    code_vec = code_vec.unsqueeze(0)
                code_vecs.append(code_vec.cpu().numpy())  

    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1)[:,::-1]    
    
    nl_urls = [ex.url for ex in query_dataloader.dataset.examples]
    code_urls = [ex.url for ex in code_dataloader.dataset.examples]

    ranks = []
    for url, row in zip(nl_urls, sort_ids):
        rank = 0; found = False
        for idx in row[:1000]:
            if not found: rank += 1
            if code_urls[idx] == url:
                found = True
        ranks.append(1/rank if found else 0)

    result = {
        "task" :  "code_search", 
        "eval_mrr": round(float(np.mean(ranks)), 4)
    }

    print(result , "\n\n")
    return result


def test_dynamic(args, model, key, dataloader):
    """Convenience test dispatcher for any task key in TASK_REGISTRY."""
    t = TASK_REGISTRY[key]["type"]
    if t == "binary":
        return test(args, model, dataloader)
    else:
        # for code_search we reuse the same dl for queries and code (dataset symmetrical)
        return test_code_search(args, model, dataloader, dataloader)


# ================== MAIN ==================
def main():
    """Main entry point for training, evaluation, and testing."""
    parser = argparse.ArgumentParser()

    # --- original args ---
    parser.add_argument("--output_dir", default='./', type=str,
                        help="Where predictions and checkpoints are written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="Number of classes for the classification model")

    # data files
    parser.add_argument("--train_data_file_vul", default="./datasets/dataset_vulnerabilty/train.jsonl", type=str)
    parser.add_argument("--eval_data_file_vul", default="./datasets/dataset_vulnerabilty/valid.jsonl", type=str)
    parser.add_argument("--test_data_file_vul", default="./datasets/dataset_vulnerabilty/test.jsonl", type=str)

    parser.add_argument("--data_file_clone", default="./datasets/dataset_clone/data.jsonl", type=str)
    parser.add_argument("--train_data_file_clone", default="./datasets/dataset_clone/train.txt", type=str)
    parser.add_argument("--eval_data_file_clone", default="./datasets/dataset_clone/valid.txt", type=str)
    parser.add_argument("--test_data_file_clone", default="./datasets/dataset_clone/test.txt", type=str)

    parser.add_argument("--train_data_file_flaky", default="./datasets/dataset_flakytest/train.json", type=str)
    parser.add_argument("--eval_data_file_flaky", default="./datasets/dataset_flakytest/valid.json", type=str)
    parser.add_argument("--test_data_file_flaky", default="./datasets/dataset_flakytest/test.json", type=str)

    parser.add_argument("--train_data_file_CodeSearch", default="./datasets/code_search/train.jsonl", type=str)
    parser.add_argument("--eval_data_file_CodeSearch", default="./datasets/code_search/valid.jsonl", type=str)
    parser.add_argument("--test_data_file_CodeSearch", default="./datasets/code_search/test.jsonl", type=str)

    parser.add_argument("--codebase_file", default=None, type=str,
                        help="Optional codebase jsonl")

    parser.add_argument("--model_name_or_path", default='Qwen/Qwen3-Embedding-4B', type=str)
    parser.add_argument("--config_name", default="Qwen/Qwen3-Embedding-4B", type=str)
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen3-Embedding-4B", type=str)
    parser.add_argument("--peft_module", default="parallel_adapter", type=str , help="[adapter , parallel_adapter , lora , prefix]")
    parser.add_argument("--bottleneck_dim", default=64, type=int, help="Bottleneck dimension for adapter/parallel_adapter")
    parser.add_argument("--lora_r", default=16, type=int, help="Rank for LoRA layers")
    parser.add_argument("--output_model_name", default="qwen_parallel_4tasks", type=str)
    
    parser.add_argument("--nl_length", default=128, type=int)
    parser.add_argument("--code_length", default=512, type=int)

    parser.add_argument("--do_train", type=bool , default=True)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)

    parser.add_argument("--n_groups", default=3, type=int)
    parser.add_argument("--train_data_rate_vul", default=1.0, type=float)
    parser.add_argument("--train_data_rate_clone", default=0.2, type=float)
    parser.add_argument("--train_data_rate_code_search", default=1.0, type=float)
    parser.add_argument("--train_data_rate_flaky", default=1.0, type=float)

    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', default=-1 ,type=int)
    parser.add_argument('--cuda_device', default='3', type=str, help='cuda device id')

    # --- task selection ---
    parser.add_argument("--tasks", dest="tasks_csv", type=str, default="",
                        help="CSV from {vul,clone,code_search,flaky}. Example: --tasks vul,clone")
    parser.add_argument("--train_vul", default=False , type=bool)
    parser.add_argument("--train_clone", default=False , type=bool)
    parser.add_argument("--train_code_search", default=False ,type=bool)
    parser.add_argument("--train_flaky", default=False , type=bool)

    torch.cuda.empty_cache()
    args = parser.parse_args()
    set_seed(seed=args.seed)

    device = torch.device("cuda:"+args.cuda_device if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1  # torch.cuda.device_count() if you want
    args.device = device
    torch.backends.cudnn.benchmark = True 
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.autocast_dtype = autocast_dtype
    
    # Keep full task list to preserve dataset task IDs; we only *select* a subset for loops.
    config = AutoConfig.from_pretrained(args.model_name_or_path , trust_remote_code=True)
    args.tasks = _resolve_active_tasks(args)         # returns pretties
    active_pretties = args.tasks
    args.id2active = {CONFIG_TASKS.index(p): i for i, p in enumerate(active_pretties)}
    config.tasks = active_pretties  
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    
    try:
        base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)
    except Exception:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    
    
    # ---- PEFT / Delta ----
    
    n_layers = 24
    try:
        if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "block"):
            n_layers = len(base_model.encoder.block)  # T5/CodeT5p
        elif hasattr(base_model, "layers"):
            n_layers = len(base_model.layers)         # LLaMA/DeepSeek style
        elif hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            n_layers = len(base_model.model.layers)
    except Exception:
        pass
    
    if "codet5p" in args.model_name_or_path.lower() :  
        adapter_patterns = ["mlp", "attn"]
        lora_patterns = ["attn"]  
        if "2b" in args.model_name_or_path.lower() : 
            lora_patterns = ["attn.qkv_proj", "attn.out_proj"]   
        parallel_adapater_patterns =  ["mlp", "attn"]

            
    elif "codellama" in args.model_name_or_path.lower() : 
        adapter_patterns = ["self_attn", "mlp"]
        parallel_adapater_patterns  = ["self_attn", "mlp"]
        lora_patterns = ["q_proj", "v_proj"]
        
    elif  "deepseek" in args.model_name_or_path.lower() : 
        adapter_patterns = ["self_attn", "mlp"]
            
        # for parallel adapter we need to add each projection twice
        parallel_adapater_patterns = ["self_attn.o_proj", "mlp.down_proj"]
        
        lora_patterns = ["q_proj", "v_proj"]
        
        
    elif "unixcoder" in args.model_name_or_path.lower() :
        adapter_patterns = ['attention', '[r](\d)+\.output']
        parallel_adapater_patterns = ['attention', '[r](\d)+\.output']
        lora_patterns = ['attention']
        
    elif "qwen" in args.model_name_or_path.lower():
        adapter_patterns = ["self_attn", "mlp"]
        parallel_adapater_patterns = ["self_attn.o_proj", "mlp.down_proj"]
        # LoRA: couvrir les deux variantes (qkv fusionné ou séparé)
        lora_patterns = ["q_proj", "v_proj"]
        
    else : 
        raise ValueError("Unknown model name or path for PEFT patterns: {}".format(args.model_name_or_path))
    
    if args.peft_module == "parallel_adapter":
        delta_model = ParallelAdapterModel(backbone_model=base_model , modified_modules= parallel_adapater_patterns , bottleneck_dim= args.bottleneck_dim)
    elif args.peft_module == "adapter":
        delta_model = AdapterModel(backbone_model=base_model,modified_modules= adapter_patterns , bottleneck_dim=args.bottleneck_dim )
    elif args.peft_module == "lora":
        delta_model = LoraModel(backbone_model=base_model  ,modified_modules=lora_patterns, lora_r = args.lora_r)   #modified_modules=["q_proj", "v_proj"]
    elif args.peft_module == "prefix":
        delta_model = PrefixModel(backbone_model=base_model)
    elif args.peft_module == "full":
        delta_model = base_model


    if args.peft_module != "full":
        delta_model.freeze_module(exclude=["deltas" , "lm_head"])
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    
    if "codet5" in args.model_name_or_path.lower():
        try:
            base_model = base_model.module.encoder
        except Exception:
            base_model = base_model.encoder
    
    MTLmodel = MultiTaskModel_MTL(base_model, config).to(args.device)
        
        
    if args.n_gpu > 1:
        MTLmodel = torch.nn.DataParallel(MTLmodel, dim=0)

    if args.do_train:
        train_results , valid_results = train(args, MTLmodel, tokenizer)
        print("\n Train results : \n")
        pprint.pprint(train_results )
        print("\n Validation results : \n")
        pprint.pprint( valid_results )

    # Optional standalone eval/test using the same dynamic helpers
    if args.do_eval:
        # Build eval loaders only for selected subset
        eval_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="eval"))
        eval_results = evaluate_dynamic(args, MTLmodel, eval_loaders)
        logger.info("\n***** Eval results (selected tasks) *****")
        for k, res in eval_results.items():
            logger.info("  [%s] %s", k, str(res))

    if args.do_test:
        # Build test loaders only for selected subset
        test_loaders = dict(_build_split_dataloaders(tokenizer, args, args.tasks, split="test"))
        logger.info("\n***** Test results (selected tasks) *****")
        for k, dl in test_loaders.items():
            res = test_dynamic(args, MTLmodel, k, dl)
            logger.info("  [%s] %s", k, str(res))


if __name__ == "__main__":
    main()
