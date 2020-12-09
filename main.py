import matplotlib
import matplotlib.pyplot as plt

import pandas as pd  

import random
import torch

import numpy as np

from torch import optim

from torch import nn

import torch.nn.functional as F
from torchvision import datasets, transforms

from snn.model import SNN, SNU
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
    spike_data = t_min + (t_max - t_min) * (1 - data).view(data.shape[0], -1)
    spike_data = F.one_hot(spike_data.long(), int(T)).float()
    return spike_data

# def encode_data_eventprop(data, t_min=2.0, t_max=12.0, T=20):
#     spike_data = t_min + (t_max - t_min) * (data < 0.5).view(data.shape[0], -1)
#     spike_data = F.one_hot(spike_data.long(), int(T))
#     return spike_data

# def encode_data_repeat(data, T=20):
#     spike_data = (data.view(data.shape[0], -1) > 0.5).float()
#     spike_data = spike_data.unsqueeze(2).repeat(1,1,T)
#     return spike_data


# def encode_data_rate(data, T=5):
#     rates = data.view(data.shape[0], -1)
#     rates = rates.unsqueeze(2).repeat(1,1,T)
#     spikes = torch.bernoulli(rates)
#     return spikes


def print_grads(named_parameters):
    for n, p in named_parameters:
            if p.requires_grad:
                avg_grad = p.grad.abs().mean()
                if avg_grad:
                    print('parameter [%s]: %f' % (n, avg_grad))


def train(model, criterion, optimizer, loader, useCuda=False):
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

        loss = criterion(-output, target)

        predictions = output.detach().argmax(dim=1)
        # if batch_idx % 100 == 0:
        #     print(predictions[0:10])
        #     print(target[0:10])

        batch_samples = len(target)
        batch_correct = (predictions == target).sum().item()
        batch_loss = loss.item() * batch_samples
        
        total_samples += batch_samples
        total_correct += batch_correct
        total_loss += batch_loss

        loss.backward()

        # print_grads(model.named_parameters())

        optimizer.step()

        if batch_idx % 100 == 0:
            print('\tBatch %3d/%d: \tAcc %5.2f  Loss %.3f' % (
                batch_idx, num_batches, 100 * batch_correct / batch_samples,
                batch_loss / batch_samples))

    print('\t\tTrain: \tAcc %5.2f  Loss %.3f' % (
        100 * total_correct / total_samples, total_loss / total_samples))
    
    train_accuracy = total_correct / total_samples
    train_loss = total_loss / total_samples

    return train_accuracy, train_loss


def test(model, loader, useCuda=False):
    total_correct = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if useCuda:
                data, target = data.cuda(), target.cuda()

            spike_data = encode_data(data)

            first_post_spikes = model(spike_data)
            predictions = first_post_spikes.data.max(1, keepdim=True)[1]
            total_correct += predictions.eq(
                target.data.view_as(predictions)).sum().item()
            total_samples += len(target)

        print('\t\t Test: \tAcc %5.2f' % (100 * total_correct / total_samples))

    test_accuracy = total_correct / total_samples

    return test_accuracy


###############################################################################
# Old training and testing functions for event prop below:

# def train(model, criterion, optimizer, loader, alpha=0.01, beta=2.0):
#     total_correct = 0
#     total_loss = 0
#     total_samples = 0
#     model.train()

#     num_batches = len(loader)

#     for batch_idx, (x, target) in enumerate(loader):
#         x, target = x.cuda(), target.cuda()
#         x = encode_data(x)

#         output = model(x)

#         loss = criterion(output, target)

#         if alpha != 0:
#             target_first_spike_times = output.gather(1, target.view(-1, 1))
#             loss += alpha * (torch.exp(target_first_spike_times / (
#                 beta * model.tau_s)) - 1).mean()

#         predictions = output.data.min(1, keepdim=True)[1]

#         batch_correct = predictions.eq(
#             target.data.view_as(predictions)).sum().item()
#         batch_loss = loss.item() * len(target)
#         batch_samples = len(target)

#         total_correct += batch_correct
#         total_loss += batch_loss
#         total_samples += batch_samples

#         optimizer.zero_grad()
#         loss.backward()

#         optimizer.step()

#         if batch_idx % 100 == 0:
#             print('\tBatch %3d/%d: \tAcc %4.2f  Loss %.3f' % (
#                 batch_idx, num_batches, 100 * batch_correct / batch_samples,
#                 batch_loss / batch_samples))

#     print('\t\tTrain: \tAcc %.2f  Loss %.3f' % (
#         100 * total_correct / total_samples, total_loss / total_samples))


# def test(model, loader):
#     total_correct = 0
#     total_samples = 0
#     model.eval()

#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(loader):
#             data, target = data.cuda(), target.cuda()
#             spike_data = encode_data(data)

#             first_post_spikes = model(spike_data)
#             predictions = first_post_spikes.data.min(1, keepdim=True)[1]
#             total_correct += predictions.eq(
#                 target.data.view_as(predictions)).sum().item()
#             total_samples += len(target)

#         print('\t\t Test: \tAcc %4.2f' % (100 * total_correct / 
#               total_samples))
###############################################################################

def plot_and_save_performance(train_accuracy, train_loss, test_accuracy, algo, num_epochs):
    
    # plot network performance and save plot
    x = range(num_epochs)

    plt.plot(x, train_accuracy, '-', label='Training Accuracy')
    plt.plot(x, test_accuracy, '-', label='Test Accuracy')
    plt.legend(loc='best')

    plt.xlabel("Epoch")
    plt.ylabel("Classification Accuracy %")
    plt.title("Network Performance Using Approximate Gradient")
    
    plt.savefig(".".join([algo,'png'])) # save network performance plot
    plt.show()
    
    # plot training loss and save plot 
    plt.plot(x, train_loss, '-', label='Training Loss')
    plt.legend(loc='best')

    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("Network Training Loss Using Approximate Gradient")

    filename = "".join([algo,'_loss'])
    plt.savefig(".".join([filename,'png'])) # save training loss plot
    plt.show()
    
    # save train and test data to csv file
    dict = {'train_accuracy': train_accuracy,
            'train_loss': train_loss, 
            'test_accuracy': test_accuracy}   
        
    df = pd.DataFrame(dict)  

    # saving the dataframe  
    df.to_csv(".".join([algo,'csv'])) 



def main(batch_size=128, num_epochs=50, useCuda=True):

    # Uncomment if using EventProp algo
    # algo = 'EventProp'
    # model = SNN(
    #     input_dim=784,
    #     output_dim=10,
    #     T=20,
    #     dt=1,
    #     tau_m=20.0,
    #     tau_s=5.0,
    #     mu=0.1,
    #     backprop=SNN.EVENTPROP).cuda()
    

    # Uncomment if using SpikeProp/gradient approximate algo
    algo = 'GradApprox'
    model = SNU(
        input_size=784,
        hidden_size=10,
        decay=1)
    
    if useCuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1)

    train_loader, test_loader = loadData(batch_size)

    train_accuracy = []
    train_loss = []
    test_accuracy = []
    for epoch in range(1, num_epochs+1):
        print('Epoch %d/%d' % (epoch, num_epochs))
        train_acc, loss = train(model, criterion, optimizer, train_loader, useCuda)
        test_acc = test(model, test_loader, useCuda)

        train_accuracy.append(train_acc)
        train_loss.append(loss)
        test_accuracy.append(test_acc)

        scheduler.step()
    
    plot_and_save_performance(train_accuracy, train_loss, test_accuracy, algo, num_epochs)

 

if __name__ == '__main__':
    main()
