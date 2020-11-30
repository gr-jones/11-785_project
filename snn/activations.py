from torch import nn

import functional as F


class SpikeActivation(nn.Module):
    def __init__(self):
        super(SpikeActivation, self).__init__()

    def forward(self, x):
        return F.SpikeActivation(x)
