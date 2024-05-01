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

from resnet import resnet50

import quanto

# model = resnet50(pretrained=True)

# model.eval()

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
# print('Accuracy of the network on the 10000 test images: %.4f %%' % (
#     100 * correct / total))

# # print(model.conv1.weight)

# quanto.quantize(model, weights=quanto.qint8, activations=None)
# quanto.freeze(model)

qmodel3 = torch.load('qresnet50.pth')

print(qmodel3.conv1.weight)

# b = io.BytesIO()
# torch.save(model.state_dict(), b)
# b.seek(0)
# state_dict2 = torch.load(b)

# torch.save(model, 'qresnet50.pth')



# loaded_state_dict2 = torch.load('qresnet50.pth')


# # model2 = resnet50(pretrained=True)
# model.load_state_dict(loaded_state_dict2)
# print(model.conv1.weight)

import timeit
correct = 0
total = 0

qmodel3.eval()
start_time = timeit.default_timer()
with torch.no_grad():
    for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = qmodel3(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(timeit.default_timer() - start_time)
print('Accuracy of the reloaded quantized network on the 10000 test images: %.4f %%' % (
    100 * correct / total))