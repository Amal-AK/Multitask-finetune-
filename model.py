import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS_STR = {
    0: "vul_detection",
    1: "clone_detection",
    2: "code_search",
    3: "flakiness_detect",
}
# Reverse mapping: task name → global integer ID
TASK_ID = {v: k for k, v in TASKS_STR.items()}
#----------------------------------------------------------------------------------------



class  MultiTaskModel_MTL(nn.Module):   
    def __init__(self, encoder , config ):
        super(MultiTaskModel_MTL, self).__init__()
        self.encoder = encoder
        self.config = config 
        
        hs = getattr(config, "hidden_size", None)
        if hs is None:
            hs = getattr(config, "d_model", None)

        # CodeT5p-2B: nested encoder config
        if hs is None and hasattr(config, "encoder"):
            enc_cfg = getattr(config, "encoder")
            # enc_cfg may be a dict or an object
            hs = getattr(enc_cfg, "n_embd", None) if not isinstance(enc_cfg, dict) else enc_cfg.get("n_embd", None)

        # last-resort: read from actual encoder module
        if hs is None:
            enc_mod = getattr(self.encoder, "encoder", self.encoder)
            if hasattr(enc_mod, "wte"):                         # CodeT5p encoder
                hs = enc_mod.wte.weight.shape[1]
            elif hasattr(enc_mod, "embed_tokens"):              # T5-style
                hs = enc_mod.embed_tokens.weight.shape[1]

        if hs is None:
            raise AttributeError("Could not infer encoder hidden size (set config.d_model or encoder.n_embd).")

        self.hidden_size = hs

        # ---- pad_token_id ----
        # Priority: explicit config value → type-based default → 0
        _model_type = getattr(config, "model_type", "").lower()
        _pad = getattr(config, "pad_token_id", None)
        if _pad is None:
            # RoBERTa-family uses 1; everything else uses 0
            # (decoder models alias eos→pad, T5 uses 0)
            _pad = 1 if _model_type in {"roberta", "bert", "deberta", "deberta-v2",
                                        "xlm-roberta", "camembert", "electra"} else 0
        self.pad_token_id = _pad

        # ---- pooling strategy ----
        # cls        — encoder-only (BERT/RoBERTa): rich [CLS] pooler embedding
        # last_token — decoder-only (causal LM): the last non-padding token has
        #              attended to all prior tokens via causal self-attention;
        #              earlier tokens are blind to later context, so mean pooling
        #              would average in under-informed representations
        # mean       — encoder-decoder (T5/CodeT5+): full bidirectional context,
        #              mean pool over non-padding encoder hidden states
        _ENCODER_ONLY   = {"roberta", "bert", "deberta", "deberta-v2", "albert",
                           "xlm-roberta", "electra", "camembert"}
        _ENCODER_DECODER = {"t5", "mt5", "bart", "mbart", "pegasus", "codet5p"}
        # Decoder-only models used as encoders (with adapters, not for generation):
        # mean pooling outperforms last_token here because the model wasn't trained
        # for last-token encoding and code functions vary widely in length.
        _MEAN_DECODER = {"qwen2", "qwen3", "llama", "mistral", "deepseek_v2",
                         "codellamaconfig", "starcoder2", "phi", "phi3"}
        if _model_type in _ENCODER_ONLY:
            self._pooling = "cls"
        elif _model_type in _ENCODER_DECODER or _model_type in _MEAN_DECODER:
            self._pooling = "mean"
        else:
            self._pooling = "last_token"

        # Uncertainty-based task weighting (Kendall et al., NeurIPS 2018).
        # Parametrise as log σ²_i; effective weight = exp(−s_i).
        # Full loss: Σ_i [ exp(−s_i)·L_i + 0.5·s_i ]
        # The +0.5·s_i term prevents collapse: s_i→∞ makes the penalty dominate
        # and pushes s_i back, so no task weight can silently go to zero.
        self.log_sigma2 = nn.Parameter(torch.zeros(len(config.tasks)))
        self.task_heads = nn.ModuleDict()
        self._head_out_dims: dict = {}   # task_name → output dimension for empty-tensor fallback
        
        # head_dropout: single consistent value, configurable via config
        head_dropout = getattr(config, "head_dropout", 0.1)

        for task in config.tasks:
            if task == "vul_detection":
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(p=head_dropout),
                    nn.Linear(self.hidden_size // 2, 1),
                )
                self._head_out_dims[task] = 1
            elif task in ("clone_detection", "flakiness_detect"):
                self.task_heads[task] = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(p=head_dropout),
                    nn.Linear(self.hidden_size // 2, 1),
                )
                self._head_out_dims[task] = 1
            elif task == "code_search":
                emb_dim = getattr(config, "code_search_emb_dim", self.hidden_size)
                self.task_heads[task] = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, emb_dim),
                )
                self._head_out_dims[task] = emb_dim
            else:
                raise ValueError(f"Unknown task: {task}")


        
    
    def _encode(self, input_ids):
        """Encode a batch of token ids → one vector per sample.

        Three pooling strategies, selected at init from config.model_type:
          cls        — encoder-only (RoBERTa/BERT): pooler_output; falls back
                       to raw [CLS] hidden state if no pooler head is present.
          last_token — decoder-only (causal LMs): hidden state of the last
                       non-padding token, which has attended to all context.
          mean       — encoder-decoder (T5/CodeT5+): mean over non-padding
                       encoder hidden states (full bidirectional context).
        """
        attention_mask = input_ids.ne(self.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        if self._pooling == "cls":
            # Prefer explicit pooler output; fall back to raw [CLS] if absent
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0, :]
            return pooled

        last_hidden = outputs.last_hidden_state   # (B, T, H)

        if self._pooling == "last_token":
            # Index of the last real (non-padding) token, 0-based, shape (B,)
            seq_lengths = attention_mask.sum(dim=1) - 1
            return last_hidden[
                torch.arange(last_hidden.size(0), device=last_hidden.device),
                seq_lengths,
            ]

        # mean pooling — cast mask to hidden dtype to stay in mixed precision
        mask_exp = attention_mask.unsqueeze(-1).to(last_hidden.dtype).expand_as(last_hidden)
        return (last_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, task_ids):
        """
        input_ids : (B, seq_len)  — all inputs for this step, concatenated across tasks.
                    For code_search, append NL inputs right after code inputs;
                    the caller tracks the split point (cs_batch_size).
        task_ids  : (B,) int tensor — global task ID per sample (same value for
                    both code and NL halves of a code_search batch).

        Returns dict {task_name: tensor} for every active task head.
        Tasks absent from this batch return an empty (0, out_dim) tensor so that
        nn.DataParallel's gather step sees uniform dict keys across all GPU replicas.
        Code-search vectors are L2-normalised; all others are raw logits.
        """
        hidden = self._encode(input_ids)          # ONE encoder pass for all tasks

        results = {}
        for task_name, head in self.task_heads.items():
            tid = TASK_ID[task_name]
            idx = (task_ids == tid).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                # Keep dict key present for DataParallel gather
                results[task_name] = hidden.new_empty(0, self._head_out_dims[task_name])
            else:
                pred = head(hidden[idx])
                if task_name == "code_search":
                    pred = F.normalize(pred, p=2, dim=1)
                results[task_name] = pred
        return results
    
    
    
    def get_task_weights(self):
        """Effective task weights exp(−log_sigma²) — not normalised, for logging."""
        return torch.exp(-self.log_sigma2).detach()
    






