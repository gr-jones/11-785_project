import numpy as np
import torch
from torch import nn

import snn.functional as F
from snn.activations import SpikeActivation, ThresholdActivation


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
            self.slinear = F.SpikingLinearSpikeProp
        elif backprop == SpikingLinearLayer.EVENTPROP:
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


class SNU_Net(nn.Module):
    '''
    This class defines a single Spiking Neural Unit Network (SNU_Net).

    Args:
        weight_xh (tensor):         the input weights [W in paper eqn(3)] - (input_size, num_neurons)
        weight_decay (tensor):      voltage decay values [fancy l in paper eqn(3)] = (num_neurons, num_neurons)
        s_t (tensor):               voltage value [s_t in paper eqn(3)] - (num_neurons, 1)
        reset_signal (tensor):      signal to keep or discard previous voltage value [1 - y_t in eqn(3)] - (num_neurons, 1)

        time_duration (int):        time length of a single trial (sequence length)
    '''

    def __init__(self, input_size, num_neurons, threshold_level, 
                    time_duration, device):
        super(SNU_Net,self).__init__()
        
        # Initialize SNU variables - variables that appear in eqn (3) of the paper
        self.weight_xh = nn.Parameter(torch.rand((input_size, num_neurons), device=device, requires_grad=True))
        self.weight_decay = torch.rand((num_neurons, num_neurons), device=device, requires_grad=True) # not sure what to initialize these to 
        self.s_t = torch.zeros((num_neurons, 1), device=device)
        self.reset_signal = torch.zeros((num_neurons,1), device=device)

        # Construct output activation function h
        self.h = ThresholdActivation()

        # Other parameters 
        self.time_duration = time_duration
        self.ones = torch.ones((num_neurons,1), device=device)

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)

    def forward(self, input, hidden = None):
        '''
        Args:
            input (tensor): (input_size, time_duration)
        Return:
            output (tensor): (time_steps, class_size)
        '''
        output = []
        for t in range(self.time_duration):
            # retrieve input spikes for given time step (t)
            x_t = input[t]

            print("-------------------------")
            print("x_t.shape = ", x_t.shape)
            print("self.weight_xh.shape = ", self.weight_xh.shape)
            print("self.weight_decay.shape = ", self.weight_decay.shape)
            print("self.s_t.shape = ", self.s_t.shape)
            print("self.reset_signal.shape = ", self.reset_signal.shape)
            print("-------------------------")

            # calculate voltage_state (s_t in the paper)
            self.s_t = torch.nn.ReLU(x_t@self.weight_xh.T() + self.weight_decay*self.s_t*self.reset_signal)

            # determine if voltage is greater than threshold value. If so, y_t = 1, otherwise y_t = 0
            y_t = self.h(self.s_t)

            # store output 
            output.append(y_t)

            # calculate reset signal value
            self.reset_signal = self.ones - y_t

        return torch.cat(output)