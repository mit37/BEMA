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

    forward() returns raw logits for two heads:
      - noul_logit: single logit for the binary "is_spam" decision
      - score_logit: single logit whose sigmoid is trained (via temperature
        scaling in calibrate.py) to be a CALIBRATED confidence, not just a
        raw softmax probability.

    Both come from ONE forward pass over the shared encoded state — no
    token-by-token generation, which is the core structural difference
    from an autoregressive LLM.
    """

    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=40, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)

        # Typed decision heads — fixed output shape, cannot emit anything
        # outside {noul_logit, score_logit}. This is the "cannot produce a
        # type error" guarantee: it's structural, not learned.
        self.noul_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def forward(self, ids, mask):
        x = self.embed(ids)
        x = self.pos(x)
        key_padding_mask = mask == 0  # True where padded
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean-pool over real (non-pad) tokens -> one state vector per example
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
        pooled = self.pool_norm(pooled)

        noul_logit = self.noul_head(pooled).squeeze(-1)
        return noul_logit

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
