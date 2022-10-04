import torch
from torch import nn
from torch.cuda.amp import autocast

class ConvFuser(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    @autocast(False)
    def forward(self, inputs):
        return super().forward(torch.cat(inputs, dim=1))
