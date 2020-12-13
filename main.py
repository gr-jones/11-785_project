import sys

import torch

from torch import optim
from torch import nn

from torchvision import datasets, transforms

from snn.model import SNN, SNU
from snn.loss import SpikeCELoss
from util import plot_and_save_performance


def load_data(batch_size):
    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)

    test_dataset = datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


def encode_data(data, t_min=2, t_max=12, T=20):
    spike_data = t_min + (t_max - t_min) * (1 - data).view(data.shape[0], -1)
    spike_data = nn.functional.one_hot(spike_data.long(), int(T)).float()
    return spike_data


# def encode_data_repeat(data, T=20):
#     spike_data = (data.view(data.shape[0], -1) > 0.5).float()
#     spike_data = spike_data.unsqueeze(2).repeat(1,1,T)
#     return spike_data


# def encode_data_rate(data, T=5):
#     rates = data.view(data.shape[0], -1)
#     rates = rates.unsqueeze(2).repeat(1,1,T)
#     spikes = torch.bernoulli(rates)
#     return spikes


def print_grads(named_parameters, fliter0=False):
    for n, p in named_parameters:
        if p.requires_grad:
            avg_grad = p.grad.abs().mean()
            if fliter0 and not avg_grad:
                continue
            print('parameter [%s]: %f' % (n, avg_grad))


def train(model, criterion, optimizer, loader, alpha=0.01, beta=2.0,
          useCuda=False):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    model.train()

    num_batches = len(loader)

    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (x, target) in enumerate(loader):
        optimizer.zero_grad()

        if useCuda:
            x, target = x.cuda(), target.cuda()
        x = encode_data(x)

        output = model(x)
        # if batch_idx % 100 == 0:
        #     print(output[0,:])

        loss = criterion(output, target)

        if type(model) == SNU:
            predictions = output.detach().argmax(dim=1)
        else:
            if alpha != 0:
                target_first_spike_times = output.gather(1, target.view(-1, 1))
                loss += alpha * (torch.exp(target_first_spike_times / (
                    beta * 5)) - 1).mean()
            predictions = output.detach().argmin(dim=1)
        # if batch_idx % 100 == 0:
        #     print(predictions[0:10])
        #     print(target[0:10])

        batch_samples = len(target)
        batch_correct = (predictions == target).sum().item()
        batch_loss = loss.item()
        if type(model) == SNU:
            batch_loss = 1 / batch_loss

        total_samples += batch_samples
        total_correct += batch_correct
        total_loss += batch_loss * batch_samples

        loss.backward()

        # print_grads(model.named_parameters())

        optimizer.step()

        if batch_idx % 100 == 0:
            print('\tBatch %3d/%d: \tAcc %5.2f  Loss %.3f' % (
                batch_idx, num_batches, 100 * batch_correct / batch_samples,
                batch_loss))

    train_acc = 100 * total_correct / total_samples
    train_loss = total_loss / total_samples

    print('\t' + '-' * 37)
    print('\t\tTrain: \tAcc %5.2f  Loss %.3f' % (train_acc, train_loss))

    return train_acc, train_loss


def test(model, criterion, loader, useCuda=False):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    model.eval()

    for batch_idx, (data, target) in enumerate(loader):
        if useCuda:
            data, target = data.cuda(), target.cuda()

        data = encode_data(data)

        output = model(data)

        loss = criterion(output, target)

        if type(model) == SNU:
            predictions = output.detach().argmax(dim=1)
            loss = 1 / loss
        else:
            predictions = output.detach().argmin(dim=1)

        total_correct += (predictions == target).sum().item()
        total_loss += loss.item() * len(target)
        total_samples += len(target)

    test_acc = 100 * total_correct / total_samples
    test_loss = total_loss / total_samples

    print('\t\t Test: \tAcc %5.2f  Loss %.3f' % (test_acc, test_loss))

    return test_acc, test_loss


def main(batch_size=128, num_epochs=50, exactGrad=True, useCuda=True):
    if exactGrad:
        print('#' * 45 + '\nRunning Eventprop:\n' + '-' * 45)
        model = SNN(
            input_dim=784,
            output_dim=10,
            T=20,
            dt=1,
            tau_m=20.0,
            tau_s=5.0,
            mu=0.1)
        criterion = SpikeCELoss(xi=0.4, tau_s=5)
    else:
        print('#' * 45 + '\nRunning SNUP:\n' + '-' * 45)
        model = SNU(
            input_size=784,
            output_size=10,
            decay=.95)
        criterion = SpikeCELoss(xi=0.2, tau_s=5)

    if useCuda:
        model = model.cuda()

    

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1)

    train_loader, test_loader = load_data(batch_size)

    train_performance = []
    test_performance = []
    for epoch in range(1, num_epochs+1):
        print('Epoch %d/%d' % (epoch, num_epochs))
        train_performance.append(train(
            model, criterion, optimizer, train_loader, useCuda=useCuda))

        with torch.no_grad():
            test_performance.append(test(
                model, criterion, test_loader, useCuda=useCuda))
        scheduler.step()

    return train_performance, test_performance


if __name__ == '__main__':
    if len(sys.argv) > 1:
        apprx_performance = main(exactGrad=False)
        exact_performance = main(exactGrad=True)
        plot_and_save_performance(exact_performance, apprx_performance)
    else:
        main(exactGrad=False)
