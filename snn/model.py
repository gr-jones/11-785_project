import torch
from torch import nn

import torch.nn.functional as F

import snn.functional as snnF
from snn.activations import SpikeActivation

import random


class SpikingLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, T=20, dt=1,
                 tau_m=20.0, tau_s=5.0, mu=0.1):
        super(SpikingLinearLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steps = T // dt
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, mu, mu)

        self.slinear = snnF.SpikingLinearEventProp

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.slinear.apply(x, self.weight, self.steps, self.dt,
                                  self.tau_m, self.tau_s, self.training)


class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, T=20, dt=1,
                 tau_m=20.0, tau_s=5.0, mu=0.1):
        super(SNN, self).__init__()

        self.slinear1 = SpikingLinearLayer(input_dim, output_dim)
        self.outact = SpikeActivation()

        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.mu = mu

    def forward(self, input):
        u = self.slinear1(input)
        u = self.outact(u)
        return u


class SNU(nn.Module):
    def __init__(self, input_size, output_size, decay):
        super(SNU, self).__init__()

        self.snucell = SNUCell(input_size, output_size, decay, useBias=True)
        self.activation = SpikeActivation()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x, hidden=None, output=None):
        '''
        Args:
            x (tensor): (batch_size, input_size)
            hidden (tensor): (batch_size, hidden_size)
            output (tensor): (batch_size, hidden_size)
        Return:
            output (tensor): (batch_size, hidden_size)
        '''

        outputs = []
        for i in range(x.shape[2]):
            output, hidden = self.snucell(x[:, :, i], hidden, output)
            outputs.append(output.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)
        return self.activation(outputs)


class SNUCell(nn.Module):
    '''
        Single cell of SNU, operate at the timestep level

        weight (tensor): (input_size, hidden_size)
            the input weights [W in paper eqn(3)]

        bias (tensor): (1, hidden_size)
            the bias added to state [b in paper eqn(3)]

        decay (scalar): (*)
            voltage decay values [fancy l in eqn(3)]

        hidden (tensor): (batch_size, hidden_size)
            voltage value [s_t in paper eqn(3)]

        output (tensor): (batch_size, hidden_size)     
            previou output value [y_t in eqn(3)]
    '''

    def __init__(self, input_size, hidden_size, decay, useBias=True):
        super(SNUCell, self).__init__()

        # Initialize weight: W
        self.weight = nn.Parameter(torch.Tensor(input_size, hidden_size))
        nn.init.normal_(self.weight, 0.1, 0.1)

        # Initialize bias: b
        self.bias = None
        if useBias:
            self.bias = nn.Parameter(torch.Tensor(1, hidden_size))
            nn.init.normal_(self.bias, 0.1, 0.1)

        # state decay factor: l(tau)
        self.decay = decay

        # Note that hidden and output are managed by SNU
        self.activation = torch.relu

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x, hidden=None, output=None):
        '''
        Args:
            x (tensor): (batch_size, input_size)
            hidden (tensor): (batch_size, hidden_size)
            output (tensor): (batch_size, hidden_size)
        Return:
            output (tensor): (batch_size, hidden_size)
        '''

        if hidden is None:
            # Assume hidden is 0 and output is 0 initially
            hidden = self.activation(x @ self.weight)
        else:
            hidden = self.activation(
                x @ self.weight + self.decay * hidden * (1 - output))

        if self.bias is None:
            output = snnF.ThresholdActivation.apply(hidden)
        else:
            output = snnF.ThresholdActivation.apply(hidden + self.bias)

        return output, hidden
