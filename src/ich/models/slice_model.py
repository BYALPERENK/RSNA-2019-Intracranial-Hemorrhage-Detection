"""The slice model (04_slice_models): a timm backbone, global average pooling, dropout and one linear layer to the six labels.

`forward` returns the logits and the pooled features before the dropout, the embedding that the context models of 05 read.
"""
import timm
import torch
from torch import nn


class SliceModel(nn.Module):
    def __init__(self, backbone: str, in_chans: int, pretrained: bool = True, n_labels: int = 6, dropout: float = 0.0):
        super().__init__()
        # timm adapts the pretrained first convolution to `in_chans` (summed for 1 channel, repeated and scaled for more)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, n_labels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.head(self.dropout(features)), features
