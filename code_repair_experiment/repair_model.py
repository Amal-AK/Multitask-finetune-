"""5-task MTL model: the main codebase's 4 classification/retrieval tasks (vul_detection,
clone_detection, code_search, flakiness_detect) plus 1 true sequence-generation task
(code_repair), added to test whether the paper's findings (PEFT vs full FT, single- vs
multi-task) generalize beyond classification/retrieval — per reviewer request.

Self-contained — does not import or modify the main model.py — since this experiment is
isolated from the main codebase per request. The pooling-task forward(input_ids, task_ids)
interface intentionally matches the main model.py's so that run.py's _encode_binary /
_binary_metrics / _encode_retrieval / _retrieval_mrr helpers can be reused unmodified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS_STR = {
    0: "vul_detection",
    1: "clone_detection",
    2: "code_search",
    3: "flakiness_detect",
    4: "code_repair",
}
TASK_ID = {v: k for k, v in TASKS_STR.items()}

_POOLING_TASKS = ("vul_detection", "clone_detection", "code_search", "flakiness_detect")


class MultiTaskModelWithRepair(nn.Module):
    """encoder: T5Stack, PEFT-adapted in-place by the caller before construction.
    full_model: the complete T5ForConditionalGeneration encoder is extracted from
        (full_model.encoder IS `encoder` — same object reference), used only for
        the code_repair generation path. Adapters injected into either view affect
        both, since they share the same encoder submodule.
    """
    def __init__(self, encoder, full_model, config):
        super().__init__()
        self.encoder    = encoder
        self.full_model = full_model
        self.config     = config

        self.hidden_size  = config.d_model
        self.pad_token_id = config.pad_token_id

        # CodeT5+ is encoder-decoder -> mean pooling over encoder hidden states
        # (matches the main model.py's choice for "t5"/"codet5p" model types).
        self._pooling = "mean"

        # Uncertainty-style weighting param kept for parity with the main model.py;
        # this experiment only uses "normalized" loss weighting, so log_sigma2 is
        # unused but harmless to keep (covers all 4 pooling tasks; code_repair's
        # weight is tracked separately in run.py since it's not in this dict).
        self.log_sigma2 = nn.Parameter(torch.zeros(len(_POOLING_TASKS)))

        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self._head_out_dims: dict = {}
        head_dropout = getattr(config, "head_dropout", 0.1)

        for task in ("vul_detection", "clone_detection", "flakiness_detect"):
            self.task_heads[task] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(p=head_dropout),
                nn.Linear(self.hidden_size // 2, 1),
            )
            self._head_out_dims[task] = 1

        emb_dim = getattr(config, "code_search_emb_dim", self.hidden_size)
        self.task_heads["code_search"] = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, emb_dim),
        )
        self._head_out_dims["code_search"] = emb_dim

    def _encode(self, input_ids):
        attention_mask = input_ids.ne(self.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).to(last_hidden.dtype).expand_as(last_hidden)
        return (last_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, task_ids, fixed_ids=None):
        """Pooling-task forward (identical interface/semantics to the main model.py's
        MultiTaskModel_MTL.forward) PLUS an optional code_repair path.

        fixed_ids is only passed for code_repair batches (input_ids=buggy_ids in that
        case). Routing the repair loss through this single forward() — rather than a
        separate forward_repair() called directly on the model — is required for
        nn.DataParallel: it only scatters/gathers across .forward(), so a method called
        directly on the wrapped model would silently run on one GPU only.
        """
        if fixed_ids is not None:
            return {"code_repair_loss": self._repair_loss(input_ids, fixed_ids).unsqueeze(0)}

        hidden = self._encode(input_ids)
        results = {}
        for task_name, head in self.task_heads.items():
            tid = TASK_ID[task_name]
            idx = (task_ids == tid).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                results[task_name] = hidden.new_empty(0, self._head_out_dims[task_name])
            else:
                pred = head(hidden[idx])
                if task_name == "code_search":
                    pred = F.normalize(pred, p=2, dim=1)
                results[task_name] = pred
        return results

    def _repair_loss(self, buggy_ids, fixed_ids):
        """Teacher-forced seq2seq loss for code repair.

        Delegates to the full T5ForConditionalGeneration's own forward, which
        internally handles decoder-input shift_right, lm_head scaling
        (tie_word_embeddings), and cross-entropy with -100 ignore_index.
        self.full_model.encoder is the SAME object as self.encoder, so whatever
        PEFT adapters were injected into self.encoder are used here too.
        """
        attn_mask = buggy_ids.ne(self.pad_token_id)
        labels = fixed_ids.masked_fill(fixed_ids == self.pad_token_id, -100)
        out = self.full_model(input_ids=buggy_ids, attention_mask=attn_mask, labels=labels)
        return out.loss

    def forward_repair(self, buggy_ids, fixed_ids):
        """Single-GPU convenience wrapper around _repair_loss — used only by eval/test
        (_encode_repair_exact_match), which always calls model.module directly (single
        GPU), so DataParallel scatter/gather doesn't apply there anyway."""
        return self._repair_loss(buggy_ids, fixed_ids)

    @torch.no_grad()
    def generate_repair(self, buggy_ids, max_length=256):
        """Greedy decode for eval (exact-match scoring)."""
        attn_mask = buggy_ids.ne(self.pad_token_id)
        return self.full_model.generate(
            input_ids=buggy_ids, attention_mask=attn_mask,
            max_length=max_length, num_beams=1,
        )

    def get_task_weights(self):
        return torch.exp(-self.log_sigma2)
