import torch
import torch.nn.functional as F

from torch.autograd import Function


class SpikingLinearEventProp(Function):
    @staticmethod
    def forward(ctx, x, w, steps, dt, tau_m, tau_s, training):
        # Leaky integrate and fire
        N = x.shape[0]
        K = w.shape[0]
        c1 = 1 - dt / tau_m
        c2 = 1 - dt / tau_s

        V = torch.zeros(N, K, device=x.device)
        I = torch.zeros(N, K, steps, device=x.device)
        output = torch.zeros(N, K, steps, device=x.device, requires_grad=True)

        while True:
            for i in range(1, steps):
                V = c1 * V + (1-c1) * I[:, :, i-1]
                I[:, :, i] = c2 * I[:, :, i-1] + F.linear(x[:, :, i-1], w)
                output[:, :, i] = V > 1
                V = (1-output[:, :, i]) * V

            if training:
                is_silent = output.sum(2).min(0)[0] == 0
                w.data[is_silent] += 0.1
                if is_silent.sum() == 0:
                    break
            else:
                break

        # save for EventProp backward pass
        ctx.save_for_backward(x, w, I, output)
        ctx.c1 = c1
        ctx.c2 = c2

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and constants
        x, w, I, output = ctx.saved_tensors
        c1 = ctx.c1  # 1 - dt / tau_m
        c2 = ctx.c2  # 1 - dt / tau_s

        N = x.shape[0]
        K = w.shape[0]
        steps = output.shape[2]

        # EventProp exact gradient calculation
        lV = torch.zeros(N, K, device=x.device)
        lI = torch.zeros(N, K, device=x.device)

        grad_x = torch.zeros(N, x.shape[1], steps, device=x.device)
        grad_w = torch.zeros(N, *w.shape, device=x.device)

        for i in range(steps-2, -1, -1):
            delta = lV - lI

            grad_x[:, :, i] = F.linear(delta, w.t())
            denom = I[:, :, i] - 1 + 1e-10

            lV = c1 * lV + output[:, :, i+1] * \
                (lV + grad_output[:, :, i+1]) / denom
            lI = lI + (1-c2) * delta

            grad_w -= x[:, :, i].unsqueeze(1) * lI.unsqueeze(2)

        return grad_x, grad_w, None, None, None, None, None


class SpikeActivation(Function):
    @staticmethod
    def forward(ctx, x):
        T = x.shape[2]

        idx = torch.arange(
            T, 0, -1, dtype=torch.float32, device=x.device)
        idx = idx.unsqueeze(0).unsqueeze(0)

        first_spike_times = torch.argmax(idx*x, dim=2).float()

        ctx.save_for_backward(first_spike_times.clone())
        ctx.T = T

        first_spike_times[first_spike_times == 0] = T-1

        first_spike_times.requires_grad = True

        return first_spike_times

    @staticmethod
    def backward(ctx, grad_output):
        first_spike_times = ctx.saved_tensors[0]

        k = F.one_hot(first_spike_times.long(), ctx.T)

        return k * grad_output.unsqueeze(-1)


class ThresholdActivation(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        output = (x > 0).float()
        output.requires_grad = True

        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Use derivative of tanh to approximate derivative of threshold function
        '''
        # retrieve input values
        x = ctx.saved_tensors[0]  # voltage_state

        # calculate derivative of voltage_state [1 - (tanh(x))^2]
        dx = 1 - x.tanh() ** 2

        # calculate gradient w.r.t. voltage_state
        grad_x = dx * grad_output

        return grad_x
