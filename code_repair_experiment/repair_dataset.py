"""Data loader for the code_repair task (CodeXGLUE Bugs2Fix / code-refinement, small split).

Self-contained — not part of the main utilities.py — since this experiment is isolated
from the main 4-task codebase per request.
"""
import json
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("name")

# Global task id for this experiment's 5-task setup (0-3 match the main codebase's
# vul_detection/clone_detection/code_search/flakiness_detect; 4 is the new task).
CODE_REPAIR_TASK_ID = 4


class InputFeatures_code_repair:
    def __init__(self, buggy_ids, fixed_ids):
        self.buggy_ids = buggy_ids
        self.fixed_ids = fixed_ids
        self.task = CODE_REPAIR_TASK_ID


class TextDataset_code_repair(Dataset):
    """Buggy Java function -> fixed Java function (seq2seq pair).

    Mirrors TextDataset_code_search's two-token-sequence shape: __getitem__ returns
    (buggy_ids, fixed_ids, task_id) instead of (code_ids, label, task_id).
    """
    def __init__(self, tokenizer, args, file_path=None, is_test=None, lang=None):
        self.examples = []
        logger.info("Preparing the code repair Dataset...\n")
        is_train = "train" in file_path
        limit = getattr(args, "max_train_samples", None) if is_train else getattr(args, "max_eval_samples", None)

        data = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                data.append(js)
                if limit and len(data) >= limit:
                    break

        for js in data:
            buggy_enc = tokenizer(js["buggy"], max_length=args.code_length,
                                  padding="max_length", truncation=True)
            fixed_enc = tokenizer(js["fixed"], max_length=args.code_length,
                                  padding="max_length", truncation=True)
            self.examples.append(InputFeatures_code_repair(buggy_enc["input_ids"], fixed_enc["input_ids"]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].buggy_ids),
            torch.tensor(self.examples[i].fixed_ids),
            torch.tensor(self.examples[i].task),
        )
