
import random
import torch

import numpy as np

from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from snn.model import SNN, SNU_Net
from snn.loss import SpikeCELoss


def loadData(batch_size):
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


def encode_data(data, t_min=2.0, t_max=12.0, T=20):
    spike_data = t_min + (t_max - t_min) * (data < 0.5).view(data.shape[0], -1)
    spike_data = F.one_hot(spike_data.long(), int(T))
    return spike_data


def train(model, criterion, optimizer, loader, alpha=0.01, beta=2.0):
    total_correct = 0
    total_loss = 0
    total_samples = 0
    model.train()

    num_batches = len(loader)

    for batch_idx, (x, target) in enumerate(loader):
        # x, target = x.cuda(), target.cuda()
        x = encode_data(x)

        output = model(x)

        loss = criterion(output, target)

        if alpha != 0:
            target_first_spike_times = output.gather(1, target.view(-1, 1))
            loss += alpha * (torch.exp(target_first_spike_times / (
                beta * model.tau_s)) - 1).mean()

        predictions = output.data.min(1, keepdim=True)[1]

        batch_correct = predictions.eq(
            target.data.view_as(predictions)).sum().item()
        batch_loss = loss.item() * len(target)
        batch_samples = len(target)

        total_correct += batch_correct
        total_loss += batch_loss
        total_samples += batch_samples

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('\tBatch %3d/%d: \tAcc %4.2f  Loss %.3f' % (
                batch_idx, num_batches, 100 * batch_correct / batch_samples,
                batch_loss / batch_samples))

    print('\t\tTrain: \tAcc %.2f  Loss %.3f' % (
        100 * total_correct / total_samples, total_loss / total_samples))


def test(model, loader):
    total_correct = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            spike_data = encode_data(data)

            first_post_spikes = model(spike_data)
            predictions = first_post_spikes.data.min(1, keepdim=True)[1]
            total_correct += predictions.eq(
                target.data.view_as(predictions)).sum().item()
            total_samples += len(target)

        print('\t\t Test: \tAcc %4.2f' % (100 * total_correct / total_samples))


def main(batch_size=1, num_epochs=3):

    # model = SNN(
    #     input_dim=784,
    #     output_dim=10,
    #     T=20,
    #     dt=1,
    #     tau_m=20.0,
    #     tau_s=5.0,
    #     mu=0.1,
    #     backprop=SNN.EVENTPROP).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SNU_Net(input_size=784,
                    num_neurons=10, 
                    threshold_level=1, 
                    time_duration=20, 
                    device=device)

    criterion = SpikeCELoss(T=20, xi=0.4, tau_s=5.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1)

    train_loader, test_loader = loadData(batch_size)

    for epoch in range(1, num_epochs+1):
        print('Epoch %d/%d' % (epoch, num_epochs))
        train(model, criterion, optimizer, train_loader)
        test(model, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()
