from torch import nn

import functional as F
from activations import SpikeActivation


class SpikingLinearLayer(nn.Module):
    SPIKEPROP = 0
    EVENTPROP = 1

    def __init__(self, input_dim, output_dim, T=20, dt=1,
                 tau_m=20.0, tau_s=5.0, mu=0.1, backprop=SPIKEPROP):
        super(SpikingLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steps = self.T // self.dt
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, mu, mu)

        if backprop == SPIKEPROP:
            self.slinear = F.SpikingLinearSpikeProp
        elif backprop == EVENTPROP:
            self.slinear = F.SpikingLinearEventProp
        else:
            raise Exception(
                'SpikingLinearLayer: invalid selection of backprop algorithm!')

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.slinear.apply(x, self.weight, self.steps, self.dt,
                                  self.tau_m, self.tau_s, self.training)


class SNN(nn.Module):
    # single layer SNN
    def __init__(self, input_dim, output_dim, backprop):
        super(SNN, self).__init__()
        self.slinear1 = SpikingLinear(
            input_dim,
            output_dim,
            backprop=backprop)
        self.outact = SpikeActivation()

    def forward(self, input):
        u = self.slinear1(input)
        u = self.outact(u)
        return u
