import torch
import torch.nn.functional as F

from torch.autograd import Function


class SpikingLinearSpikeProp(Function):
    @staticmethod
    def forward(ctx, x, w, steps, dt, tau_m, tau_s, training):
        # Leaky integrate and fire
        N = x.shape[0]
        K = w.shape[0]
        c1 = 1 - dt / tau_m
        c2 = 1 - dt / tau_s

        V = torch.zeros(N, K, device=x.device)
        I = torch.zeros(N, K, steps, device=x.device)
        output = torch.zeros(N, K, steps, device=x.device)

        while True:
            for i in range(1, steps):
                V = c1 * V + (1-c1) * I[:,:,i-1]
                I[:,:,i] = c2 * I[:,:,i-1] + F.linear(x[:,:,i-1], w)
                output[:,:,i] = V > 1
                V = (1-output[:,:,i]) * V

            if training:
                is_silent = output.sum(2).min(0)[0] == 0
                self.weight.data[is_silent] += 0.1
                if is_silent.sum() == 0:
                    break
            else:
                break

        # Save for SpikeProp backward pass
        raise Exception('Not yet implemented')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and constants
        raise Exception('Not yet implemented!')

        # SpikeProp approximate gradient calculation
        raise Exception('Not yet implemented!')

        return grad_x, grad_w, None, None, None, None, None


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
        output = torch.zeros(N, K, steps, device=x.device)

        while True:
            for i in range(1, steps):
                V = c1 * V + (1-c1) * I[:,:,i-1]
                I[:,:,i] = c2 * I[:,:,i-1] + F.linear(x[:,:,i-1], w)
                output[:,:,i] = V > 1
                V = (1-output[:,:,i]) * V

            if training:
                is_silent = output.sum(2).min(0)[0] == 0
                self.weight.data[is_silent] += 0.1
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
        c1 = ctx.c1 # 1 - dt / tau_m
        c2 = ctx.c2 # 1 - dt / tau_s

        N = x.shape[0]
        K = w.shape[0]
        steps = output.shape[2]

        # EventProp exact gradient calculation
        lV = torch.zeros(N, K, device=x.device)
        lI = torch.zeros(N, K, device=x.device)

        grad_x = torch.zeros(N, K, steps, device=x.device)
        grad_w = torch.zeros(N, K, steps, device=x.device)

        for i in range(steps-2, -1, -1):
            delta = lV - lI
            grad_x[:,:,i] = F.linear(delta, w.T())
            denom = I[:,:,i] - 1 + 1e-10

            lV = c1 * lV + output[:,:,i+1] * (lV + grad_output[:,:,i+1])/denom
            lI = lI + (1-c2) * delta

            grad_w -= x[:,:,i].unsqueeze(1) * lI.unsqueeze(2)

        return grad_x, grad_w, None, None, None, None, None


class SpikeActivation(Function):
    @staticmethod
    def forward(ctx, x):
        N = x.shape[2]

        idx = torch.arange(N, 0, -1, dtype=torch.float32, device=x.device)
        idx = idx.unsqueeze(0).unsqueeze(0)

        first_spike_times = torch.argmax(idx*x, dim=2)

        ctx.save_for_backward(first_spike_times.clone())
        ctx.N = N

        first_spike_times[first_spike_times == 0] = N-1

        return first_spike_times

    @staticmethod
    def backward(ctx, grad_output):
        first_spike_times = ctx.saved_tensors[0]

        k = F.one_hot(first_spike_times, ctx.N)

        return k * grad_output.unsqueeze(-1)
