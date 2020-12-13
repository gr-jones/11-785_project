from torch import nn


class SpikeCELoss(nn.Module):
    def __init__(self, xi=0.4, tau_s=5):
        super(SpikeCELoss, self).__init__()
        self.xi = xi
        self.tau_s = tau_s
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        loss = self.celoss(-x / (self.xi * self.tau_s), target)
        return loss
