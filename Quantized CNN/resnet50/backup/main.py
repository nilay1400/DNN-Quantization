import os
import zipfile
import torch
import io

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn

from models.resnet import resnet18, resnet34, resnet50

import quanto
import matplotlib.pyplot as plt

from fi import FI
import pandas

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "resnet-fi-70-127int.csv"

Sufficient_no_faults = 200

#model = resnet50(pretrained=True)
#model = torch.load('qresnet-70-127modelint.pth')

#model.eval()

def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader

transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
        ]
    )
dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

evens = list(range(0, len(dataset), 10))
trainset_1 = torch.utils.data.Subset(dataset, evens)

data = val_dataloader()
"""
layer_weights = model.conv1.weight._data.numpy()

# # Plot the distribution of weights
plt.hist(layer_weights.flatten(), bins=50)
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Distribution of Weights')
plt.show()


import timeit
correct = 0
total = 0

model.eval()
start_time = timeit.default_timer()
with torch.no_grad():
    for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(timeit.default_timer() - start_time)
print('Accuracy of the network on the 10000 test images: %.4f %%' % (
    100 * correct / total))
"""

# # print(model.conv1.weight)

#quanto.quantize(model, weights=quanto.qint8, activations=None)
#quanto.freeze(model)

# torch.load('qresnet-80-127.pth')

# print(model.conv1.weight)

# b = io.BytesIO()
# torch.save(model.state_dict(), b)
# b.seek(0)
# state_dict2 = torch.load(b)

#torch.save(model, 'qresnet-70-127modelint.pth')
#loaded_state_dict2 = torch.load('qresnet-80-127.pth')
#model.load_state_dict(loaded_state_dict2)

# import timeit
# correct = 0
# total = 0

# model.eval()
# start_time = timeit.default_timer()
# with torch.no_grad():
#     for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
#         images, labels = images.to("cpu"), labels.to("cpu")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(timeit.default_timer() - start_time)
# print('Accuracy of the golden quantized network on the 10000 test images: %.4f %%' % (
#     100 * correct / total))


def test():
    #loaded_state_dict2 = torch.load('qresnet-80-127.pth')


# # model2 = resnet50(pretrained=True)
# print(model.conv1.weight)
    #model.load_state_dict(loaded_state_dict2)
    # print(model.conv1.weight._data)
    model = torch.load('qresnet-70-127modelint.pth')
    weights = model.conv1.weight._data
    fi = FI(weights)
    index, bit = fi.fault_position()
    new_weights = fi.inject(index, bit)
    model.conv1.weight._data = new_weights

#layer_weights = model.conv1.weight._data.numpy()

# Plot the distribution of weights
#plt.hist(layer_weights.flatten(), bins=50)
#plt.xlabel('Weight Value')
#plt.ylabel('Frequency')
#plt.title('Distribution of Weights')
#plt.show()


    import timeit
    correct = 0
    total = 0

    model.eval()
    start_time = timeit.default_timer()
    with torch.no_grad():
        for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(timeit.default_timer() - start_time)
    print('Accuracy: %.4f %%' % (100 * correct / total))
    return(100. * correct / total)

for k in range(Sufficient_no_faults):
    print(f"{Sufficient_no_faults - k} faults to inject")
    accuracy = test()
    acc_dict["Accuracy"].append(accuracy)
    acc_dict["noFI"].append(k)


data = pandas.DataFrame(acc_dict)
data.to_csv(csv_acc)
avg_accuracy = sum(acc_dict["Accuracy"])/len(acc_dict["Accuracy"])
print('Average Faulty Accuracy: %.4f %%' % (avg_accuracy))
print('Accuracy Drop: %.4f %%' % (93.5897 - avg_accuracy))   