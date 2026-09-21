"""
Jev-clone: a non-autoregressive typed-decision model.

Architecture: small transformer encoder (trained from scratch — no
pretrained weights, since the sandbox can't reach HuggingFace's hub) that
takes tokenized text (the "state") and returns, in a SINGLE forward pass:
  - a Noul decision: is_spam (bool)
  - a Score: calibrated confidence in [0, 1]

This mirrors Jev's actual interface shape (typed questions -> typed
answers with confidence), just built at toy scale on real labeled data.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class JevCloneEncoder(nn.Module):
    """
    Non-autoregressive encoder + typed decision heads.

    encode() runs the shared transformer trunk ONCE per input and returns a
    single pooled state vector -- no token-by-token generation, which is
    the core structural difference from an autoregressive LLM. Every typed
    head (Noul, Choice, Score) reads from that SAME pooled vector, so one
    encoder forward pass can answer multiple typed questions about the same
    state, matching Jev's "many typed questions evaluated in parallel from
    one state" interface shape.

    - noul_head: always present. Single logit for a binary decision
      (is_spam in the spam pipeline).
    - choice_head: present only if num_choice_classes is given. Logits over
      a fixed real label set (e.g. the 77 BANKING77 intent categories).
    - score_head: present only if enable_score=True. Single logit whose
      sigmoid is a continuous Score in [0, 1] (e.g. STS-B similarity).

    Each head is a fixed-shape linear projection -- it cannot emit anything
    outside its declared output shape. This is the "cannot produce a type
    error" guarantee: it's structural, not learned.
    """

    def __init__(self, vocab_size, d_model=96, nhead=6, num_layers=3, max_len=48,
                 dropout=0.15, num_choice_classes=None, enable_score=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)

        self.noul_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.choice_head = None
        if num_choice_classes is not None:
            self.choice_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_choice_classes)
            )
        self.score_head = None
        if enable_score:
            self.score_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )

        # Per-head calibration temperature (Guo et al. 2017). noul/choice
        # use it directly on logits; score's "confidence" calibration is a
        # separate, weaker notion -- see calibrate_multitask.py.
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)
        self.choice_temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def encode(self, ids, mask):
        """Run the shared trunk once. Returns one pooled state vector per example."""
        x = self.embed(ids)
        x = self.pos(x)
        key_padding_mask = mask == 0  # True where padded
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean-pool over real (non-pad) tokens -> one state vector per example
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
        return self.pool_norm(pooled)

    def forward(self, ids, mask):
        """Backward-compatible single-task entry point: returns the noul logit."""
        pooled = self.encode(ids, mask)
        return self.noul_head(pooled).squeeze(-1)

    def forward_all(self, ids, mask):
        """
        ONE encoder forward pass -> every typed head that exists on this
        model, read off the same pooled state. This is the literal
        "many typed questions in parallel from one state" claim.
        """
        pooled = self.encode(ids, mask)
        out = {"noul_logit": self.noul_head(pooled).squeeze(-1)}
        if self.choice_head is not None:
            out["choice_logits"] = self.choice_head(pooled)
        if self.score_head is not None:
            out["score_logit"] = self.score_head(pooled).squeeze(-1)
        return out

    def predict(self, ids, mask):
        """Typed inference: returns {is_spam, confidence} — no text generated."""
        with torch.no_grad():
            logit = self.forward(ids, mask)
            calibrated_logit = logit / self.temperature
            prob_spam = torch.sigmoid(calibrated_logit)
            is_spam = prob_spam > 0.5
            # Confidence = probability mass on the predicted class
            confidence = torch.where(is_spam, prob_spam, 1 - prob_spam)
        return is_spam, confidence, prob_spam

    def predict_choice(self, ids, mask):
        """Typed inference for the Choice head: {class_idx, confidence}."""
        with torch.no_grad():
            pooled = self.encode(ids, mask)
            logits = self.choice_head(pooled) / self.choice_temperature
            probs = torch.softmax(logits, dim=-1)
            confidence, class_idx = probs.max(dim=-1)
        return class_idx, confidence, probs

    def predict_score(self, ids, mask):
        """Typed inference for the Score head: a single value in [0, 1]."""
        with torch.no_grad():
            pooled = self.encode(ids, mask)
            score = torch.sigmoid(self.score_head(pooled).squeeze(-1))
        return score
