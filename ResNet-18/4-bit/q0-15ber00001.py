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
import random
import timeit

from models.resnet import resnet18, resnet34, resnet50
#from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# import quanto
import matplotlib.pyplot as plt

from fi015 import FI
from fi0152 import FI2
from dmr015 import DMR
from dmr0152 import DMR2
import pandas
import csv

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "resnet18-0-15-fi50-ber00001.csv"


output_results_file = open("output_resnet18_0-15_ber00001", "w")
output_results = csv.DictWriter(output_results_file,
                                        [
                                            "fault_id",
                                            "img_id",
                                            "predicted",
                                            *[f"p{i}" for i in range(10)],
                                        ]
                                        )
output_results.writeheader()
output_results_file.close()

output_results_file = open("output_resnet18_0-15_ber00001", "a")
output_results = csv.DictWriter(output_results_file, ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]])

Sufficient_no_faults = 20
BER = 0.00001

model = torch.load('qresnet18-0-15.pth')

layer_list = ['conv1', 'layer1[0].conv1', 'layer1[0].conv2', 'layer1[1].conv1', 'layer1[1].conv2', 'layer2[0].conv1', 'layer2[0].conv2', 'layer2[1].conv1', 'layer2[1].conv2', 'layer3[0].conv1', 'layer3[0].conv2', 'layer3[1].conv1', 'layer3[1].conv2', 'layer4[0].conv1', 'layer4[0].conv2', 'layer4[1].conv1', 'layer4[1].conv2', 'fc']
first = ['conv1', 'layer1[0].conv1', 'layer1[0].conv2', 'layer1[1].conv1', 'layer1[1].conv2', 'layer2[0].conv1', 'layer2[0].conv2', 'layer2[1].conv1', 'layer2[1].conv2', 'layer3[0].conv1', 'layer3[0].conv2', 'layer3[1].conv1', 'layer3[1].conv2', 'layer4[0].conv1', 'layer4[0].conv2', 'layer4[1].conv1', 'layer4[1].conv2']
second = ['fc']


def get_nested_attr(obj, attr):
    try:
        # Split the string by '.' to get individual attributes and indices
        parts = attr.split('.')
        for part in parts:
            # Check if part is indexed (e.g., 'features[0]')
            if '[' in part and ']' in part:
                # Split by '[' and extract the index
                part, idx = part.split('[')
                idx = int(idx[:-1])  # Convert '0]' to 0
                obj = getattr(obj, part)[idx]
            else:
                obj = getattr(obj, part)
        return obj
    except AttributeError as e:
        print(f"Error: {e}")
        return None

for l in layer_list:
    print(l)
    weights = get_nested_attr(model, l).weight._data
    if l in first : dmr = DMR(weights)
    if l in second : dmr = DMR2(weights)
    new_weights = dmr.set()
    get_nested_attr(model, l).weight._data = new_weights
    
original_tmr_weights = []
for i in layer_list:
    original_tmr_weights.append(get_nested_attr(model, i).weight._data)


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
        shuffle= False,
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


def no_faults():
    number =[]
    for i in layer_list:
        weights = get_nested_attr(model, i).weight._data
        if i in first: fi = FI(weights)
        elif i in second: fi = FI2(weights)
        nn = fi.param(weights)
        # print(nn)
        number.append(nn)
    total = sum(number) * 6
    return(total)

def generate_fault_list(n):
    no_faults_each_iteration = int(BER * n)
    print("each iteration:", no_faults_each_iteration)
    for j in range(no_faults_each_iteration):
        layer = random.choice(layer_list)
        weights = get_nested_attr(model, layer).weight._data
        if layer in first: fi = FI(weights)
        elif layer in second: fi = FI2(weights)
        index, bit = fi.fault_position()
        fault_dict['Iteration'].append(k)
        fault_dict['Layer'].append(layer)
        fault_dict['Index'].append(index)
        fault_dict['Bit'].append(bit)



def test(n):
    #model = torch.load('qvgg11-0-7.pth')
    layer_count = 0
    for i in layer_list:
        get_nested_attr(model, i).weight._data = original_tmr_weights[layer_count] 
        layer_count += 1
    

    p = 0
    for t in range(int(BER * n)):
       
        layer = random.choice(layer_list)
        weights = get_nested_attr(model, layer).weight._data
        if layer in first : fi = FI(weights)
        if layer in second : fi = FI2(weights)
        index, bit = fi.fault_position()
        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights
        p += 1
        #print("which fault", k, p)
    start_time1 = timeit.default_timer()
    for l in layer_list:
        print(l)
        weights = get_nested_attr(model, l).weight._data
        if l in first : dmr = DMR(weights)
        if l in second : dmr = DMR2(weights)
        new_weights = dmr.protect()
        get_nested_attr(model, l).weight._data = new_weights
    
    correct = 0
    total = 0
    img_id = 0

    model.eval()
    start_time = timeit.default_timer()
    with torch.no_grad():
        for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(outputs.size(0)):  # Iterate over the batch size
                image_output = outputs[i]
                prediction = predicted[i]
                probs = {f"p{i}": "{:.2f}".format(float(image_output[i])) for i in range(10)}
                csv_output = {
                    "fault_id": k, "img_id": img_id, "predicted": prediction.item(),
                }
                csv_output.update(probs)
                img_id += 1
                output_results.writerow(csv_output)
    print(timeit.default_timer() - start_time)
    print("Total time:", timeit.default_timer() - start_time1)
    print('Accuracy: %.4f %%' % (
    100 * correct / total))
    return(100. * correct / total)

n = no_faults()
print(n)
for k in range(Sufficient_no_faults):
    print(f"{Sufficient_no_faults - k} faults to inject")
    accuracy = test(n)
    acc_dict["Accuracy"].append(accuracy)
    acc_dict["noFI"].append(k)
   
data = pandas.DataFrame(acc_dict)
data.to_csv(csv_acc)
avg_accuracy = sum(acc_dict["Accuracy"])/len(acc_dict["Accuracy"])
print('Average Faulty Accuracy: %.4f %%' % (avg_accuracy))
print('Accuracy Drop: %.4f %%' % (92.6683 - avg_accuracy))    
