"""The sequence model (05_sequence_models): the slices of one series, in order, through a sequence encoder.

Inputs per slice: the pooled CNN features (through LayerNorm and an optional projection), the CNN logits (of an
ensemble: the logits of every model side by side, `logits_in`) and side inputs (the gaps to the neighbours). With
`residual` the output is the CNN logits plus a correction, and the correction starts at zero, so the untrained model
equals the CNN.

Encoders (`arch`); padding never reaches a real slice in any of them:
- "gru" / "lstm": bidirectional, on packed sequences, so the backward direction starts at the last real slice.
- "tcn": dilated 1D convolutions over the slices, not causal; padding is zeroed after every LayerNorm, just before
  every convolution, so the end of a series looks like the end of the scan.
- "transformer": pre-norm self-attention with a key padding mask and a learned bias per head for the distance
  between two slices (in slices) instead of absolute positions.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2)  # receptive field +-66 slices: every slice sees the whole series (<= 60)
MAX_LEN = 64  # transformer: longest series the distance bias covers


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """(B, T, C), mask (B, T, 1) 1 on real slices -> (B, T, C)."""
        h = self.conv((self.norm(x) * mask).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.gelu(h))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads, self.p = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias):
        """(B, T, C), bias (B, H, T, T) added to the attention scores (-inf on padded keys) -> (B, T, C)."""
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.p if self.training else 0.0)
        x = x + self.dropout(self.out(a.transpose(1, 2).reshape(b, t, c)))
        return x + self.dropout(self.ff(self.norm2(x)))


class Encoder(nn.Module):
    """One sequence encoder: (B, T, width), lengths (B,) -> (B, T, out_width); positions past a length are padding
    (any values). `hidden`: per direction for gru / lstm, the width for tcn / transformer; `layers` only for gru /
    lstm / transformer (tcn: one block per dilation)."""

    def __init__(self, arch: str, width: int, hidden: int, layers: int):
        super().__init__()
        self.arch = arch
        if arch in ("gru", "lstm"):
            rnn = nn.GRU if arch == "gru" else nn.LSTM
            setattr(self, arch, rnn(width, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                    dropout=0.1 if layers > 1 else 0.0))  # self.gru / self.lstm
            if arch == "lstm":  # forget gate bias 1 (gate order i, f, g, o), the other bias vector 0 there
                for name, p in self.lstm.named_parameters():
                    if name.startswith("bias_"):
                        with torch.no_grad():
                            p[hidden:2 * hidden] = 1.0 if name.startswith("bias_ih") else 0.0
            self.out_width = 2 * hidden
        elif arch == "tcn":
            self.inp = nn.Linear(width, hidden)
            self.blocks = nn.ModuleList(TCNBlock(hidden, d, 0.1) for d in TCN_DILATIONS)
            self.norm = nn.LayerNorm(hidden)
            self.out_width = hidden
        elif arch == "transformer":
            self.n_heads = 4
            self.inp = nn.Linear(width, hidden)
            self.blocks = nn.ModuleList(TransformerBlock(hidden, self.n_heads, 4 * hidden, 0.1) for _ in range(layers))
            self.distance_bias = nn.Parameter(torch.zeros(self.n_heads, 2 * MAX_LEN - 1))
            self.norm = nn.LayerNorm(hidden)
            self.out_width = hidden
        else:
            raise ValueError(f"unknown arch {arch}")

    def forward(self, x, lengths):
        t = x.shape[1]
        if self.arch in ("gru", "lstm"):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = getattr(self, self.arch)(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
            return out
        real = torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None]  # (B, T)
        x = self.inp(x)
        if self.arch == "tcn":
            mask = real[..., None].to(x.dtype)
            for block in self.blocks:
                x = block(x, mask)
        else:
            i = torch.arange(t, device=x.device)
            bias = self.distance_bias[:, i[None, :] - i[:, None] + MAX_LEN - 1]  # (H, T, T), key minus query
            padding = torch.zeros(real.shape, dtype=x.dtype, device=x.device).masked_fill(~real, float("-inf"))
            bias = bias[None] + padding[:, None, None, :]
            for block in self.blocks:
                x = block(x, bias)
        return self.norm(x)


class SequenceModel(nn.Module):
    """`two_branch` (with features and logits; after the 1st place of 2019): branch 1 reads the (projected) features
    alone and predicts the CNN logits plus a correction (`aux`, for a second loss: deep supervision); branch 2 reads
    the logits input and branch 1's logits and gives the output. Each branch has its own encoder of `arch`."""

    def __init__(self, n_features: int, use_features: bool = True, use_logits: bool = True, proj: int = 256,
                 n_side: int = 0, hidden: int = 256, layers: int = 2, dropout: float = 0.3, residual: bool = True,
                 n_labels: int = 6, n_logits_in: int = None, arch: str = "gru", two_branch: bool = False):
        super().__init__()
        assert not two_branch or (use_features and use_logits and residual), "two branches: features, logits, residual"
        self.use_features, self.use_logits, self.residual, self.two_branch = use_features, use_logits, residual, two_branch
        self.aux = None
        n_logits = (n_logits_in or n_labels) if use_logits else 0
        width = n_side + n_logits
        if use_features:
            steps = [nn.LayerNorm(n_features)] + ([nn.Linear(n_features, proj)] if proj else []) + [nn.Dropout(dropout)]
            self.features = nn.Sequential(*steps)
            width += proj or n_features
        if two_branch:
            self.encoder1 = Encoder(arch, n_side + (proj or n_features), hidden, layers)
            self.head1 = nn.Linear(self.encoder1.out_width, n_labels)
            nn.init.zeros_(self.head1.weight), nn.init.zeros_(self.head1.bias)
            width = n_side + n_logits + n_labels
        self.encoder = Encoder(arch, width, hidden, layers)
        self.head = nn.Linear(self.encoder.out_width, n_labels)
        if residual:
            nn.init.zeros_(self.head.weight), nn.init.zeros_(self.head.bias)
        self._register_load_state_dict_pre_hook(self._old_keys)

    @staticmethod
    def _old_keys(state_dict, prefix, *args):
        """Weights saved before the encoders were a module of their own (gru.* -> encoder.gru.*, also lstm / tcn /
        transformer)."""
        old = tuple(prefix + p for p in ("gru.", "lstm.", "inp.", "blocks.", "norm.", "distance_bias"))
        for key in [k for k in state_dict if k.startswith(old)]:
            state_dict[prefix + "encoder." + key[len(prefix):]] = state_dict.pop(key)

    def forward(self, features, logits, side, lengths, logits_in=None):
        """(B, T, d), (B, T, 6), (B, T, n_side), (B,) -> logits (B, T, 6); positions past a length are padding.
        `logits` is the base of the residual head; `logits_in` (B, T, n_logits_in), if given, the logits input.
        Two branches: branch 1's logits are kept in `self.aux`."""
        logits_in = logits if logits_in is None else logits_in
        side = [side] if side.shape[-1] else []
        if self.two_branch:
            self.aux = logits + self.head1(self.encoder1(torch.cat([self.features(features)] + side, -1), lengths))
            x = torch.cat([logits_in, self.aux] + side, -1)
        else:
            parts = ([self.features(features)] if self.use_features else []) + ([logits_in] if self.use_logits else [])
            x = torch.cat(parts + side, -1)
        out = self.head(self.encoder(x, lengths))
        return logits + out if self.residual else out
