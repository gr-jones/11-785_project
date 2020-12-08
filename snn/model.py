import torch
from torch import nn

import torch.nn.functional as F

import snn.functional as snnF
from snn.activations import SpikeActivation

import random

class SpikingLinearLayer(nn.Module):
    SPIKEPROP = 0
    EVENTPROP = 1

    def __init__(self, input_dim, output_dim, T=20, dt=1,
                 tau_m=20.0, tau_s=5.0, mu=0.1,
                 backprop=SPIKEPROP):
        super(SpikingLinearLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steps = T // dt
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, mu, mu)

        if backprop == SpikingLinearLayer.SPIKEPROP:
            self.slinear = snnF.SpikingLinearSpikeProp
        elif backprop == SpikingLinearLayer.EVENTPROP:
            self.slinear = snnF.SpikingLinearEventProp
        else:
            raise Exception(
                'SpikingLinearLayer: invalid selection of backprop algorithm!')

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.slinear.apply(x, self.weight, self.steps, self.dt,
                                  self.tau_m, self.tau_s, self.training)


class SNN(nn.Module):
    SPIKEPROP = 0
    EVENTPROP = 1

    # single layer SNN
    def __init__(self, input_dim, output_dim, T=20, dt=1,
                 tau_m=20.0, tau_s=5.0, mu=0.1, backprop=SPIKEPROP):
        super(SNN, self).__init__()
        self.slinear1 = SpikingLinearLayer(
            input_dim,
            output_dim,
            backprop=backprop)
        self.outact = SpikeActivation()

        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.mu = mu
        self.backprop = 'EVENTPROP' if backprop == 1 else 'SPIKEPROP'

    def forward(self, input):
        u = self.slinear1(input)
        u = self.outact(u)
        return u


class SNU(nn.Module):
    def __init__(self, input_size, hidden_size, decay, num_layers=1):
        super(SNU, self).__init__()

        self.snucell = SNUCell(input_size, hidden_size, decay)

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

        for i in range(x.shape[2]):
            output, hidden = self.snucell(x[:,:,i], hidden, output)

        return output


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

    def __init__(self, input_size, hidden_size, decay):
        super(SNUCell, self).__init__()
        
        # Initialize weight: W
        self.weight = nn.Parameter(torch.rand(
            (input_size, hidden_size), 
            requires_grad=True))

        # Initialize bias: b
        self.bias = nn.Parameter(torch.rand(
            (1, hidden_size), 
            requires_grad=True))

        # state decay factor: l(tau)
        self.decay = decay

        # Note that hidden and output are managed by SNU

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
        try:
            # Save on performance by asking for forgiveness instead of 
            # permission: try to calculate assuming hidden is not None
            
            hidden = F.relu(
                x @ self.weight + self.decay * hidden * (1 - output))

        except TypeError:
            # If we get a TypeError, it's most likely because hidden was None
            if not hidden:
                # Assume hidden is 0 and output is 0 initially
                hidden = F.relu(x @ self.weight)
            else:
                # State is not none but we got a TypeError. Something's wrong, 
                # so raise the error again
                raise

        output = F.threshold(hidden + self.bias, 1, 0)

        return output, hidden