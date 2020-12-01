from torch import nn

import snn.functional as F


class SpikeActivation(nn.Module):
    def __init__(self):
        super(SpikeActivation, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return F.SpikeActivation.apply(x)
