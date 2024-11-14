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

from models.inception import inception_v3

import quanto
import matplotlib.pyplot as plt

from fi import FI
from fi2 import FI2
from fi3 import FI3
import pandas
import csv

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "inception-normal-fi50-ber00001.csv"

output_results_file = open("output_inception_normal_ber00001", "w")
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

output_results_file = open("output_inception_normal_ber00001", "a")
output_results = csv.DictWriter(output_results_file, ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]])

Sufficient_no_faults = 20
BER = 0.00001

#fault_list = pandas.read_csv('resnet18_fault_list50_ber0001.csv')

model = torch.load('qinception-normal.pth')

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

# Function to dynamically get the attribute
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

layer_list = ['Conv2d_1a_3x3.conv', 'Mixed_5b.branch1x1.conv', 'Mixed_5b.branch5x5_1.conv', 
              'Mixed_5b.branch5x5_2.conv', 'Mixed_5b.branch5x5_1.conv', 'Mixed_5b.branch3x3dbl_1.conv',
                'Mixed_5b.branch3x3dbl_2.conv', 'Mixed_5b.branch3x3dbl_3.conv', 'Mixed_5b.branch_pool.conv', 
                'Mixed_5c.branch1x1.conv', 'Mixed_5c.branch5x5_1.conv', 'Mixed_5c.branch5x5_2.conv', 
                'Mixed_5c.branch5x5_1.conv', 'Mixed_5c.branch3x3dbl_1.conv', 'Mixed_5c.branch3x3dbl_2.conv', 
                'Mixed_5c.branch3x3dbl_3.conv', 'Mixed_5c.branch_pool.conv', 'Mixed_5d.branch1x1.conv', 
                'Mixed_5d.branch5x5_1.conv', 'Mixed_5d.branch5x5_2.conv', 'Mixed_5d.branch3x3dbl_1.conv', 
                'Mixed_5d.branch3x3dbl_2.conv', 'Mixed_5d.branch3x3dbl_3.conv', 'Mixed_5d.branch_pool.conv', 
                'Mixed_6a.branch3x3.conv', 'Mixed_6a.branch3x3dbl_1.conv', 'Mixed_6a.branch3x3dbl_2.conv', 
                'Mixed_6a.branch3x3dbl_3.conv', 'Mixed_6b.branch1x1.conv', 'Mixed_6b.branch7x7_1.conv', 
                'Mixed_6b.branch7x7_2.conv', 'Mixed_6b.branch7x7_3.conv', 'Mixed_6b.branch7x7dbl_1.conv', 
                'Mixed_6b.branch7x7dbl_2.conv', 'Mixed_6b.branch7x7dbl_3.conv', 'Mixed_6b.branch7x7dbl_4.conv', 
                'Mixed_6b.branch7x7dbl_5.conv', 'Mixed_6b.branch_pool.conv', 'Mixed_6c.branch1x1.conv',
                'Mixed_6c.branch7x7_1.conv', 'Mixed_6c.branch7x7_2.conv', 'Mixed_6c.branch7x7_3.conv', 
                'Mixed_6c.branch7x7dbl_1.conv', 'Mixed_6c.branch7x7dbl_2.conv', 'Mixed_6c.branch7x7dbl_3.conv', 
                'Mixed_6c.branch7x7dbl_4.conv', 'Mixed_6c.branch7x7dbl_5.conv', 'Mixed_6c.branch_pool.conv',
                'Mixed_6d.branch1x1.conv',
                'Mixed_6d.branch7x7_1.conv', 'Mixed_6d.branch7x7_2.conv', 'Mixed_6d.branch7x7_3.conv', 
                'Mixed_6d.branch7x7dbl_1.conv', 'Mixed_6d.branch7x7dbl_2.conv', 'Mixed_6d.branch7x7dbl_3.conv', 
                'Mixed_6d.branch7x7dbl_4.conv', 'Mixed_6d.branch7x7dbl_5.conv', 'Mixed_6d.branch_pool.conv',
                'Mixed_6e.branch1x1.conv',
                'Mixed_6e.branch7x7_1.conv', 'Mixed_6e.branch7x7_2.conv', 'Mixed_6e.branch7x7_3.conv', 
                'Mixed_6e.branch7x7dbl_1.conv', 'Mixed_6e.branch7x7dbl_2.conv', 'Mixed_6e.branch7x7dbl_3.conv', 
                'Mixed_6e.branch7x7dbl_4.conv', 'Mixed_6e.branch7x7dbl_5.conv', 'Mixed_6e.branch_pool.conv',
                'Mixed_7a.branch3x3_1.conv', 'Mixed_7a.branch3x3_2.conv', 'Mixed_7a.branch7x7x3_1.conv',
                'Mixed_7a.branch7x7x3_2.conv', 'Mixed_7a.branch7x7x3_3.conv', 'Mixed_7a.branch7x7x3_4.conv',
                'Mixed_7b.branch1x1.conv', 'Mixed_7b.branch3x3_1.conv', 'Mixed_7b.branch3x3_2a.conv',
                'Mixed_7b.branch3x3_2b.conv', 'Mixed_7b.branch3x3dbl_1.conv', 'Mixed_7b.branch3x3dbl_2.conv',
                'Mixed_7b.branch3x3dbl_3a.conv', 'Mixed_7b.branch3x3dbl_3b.conv', 'Mixed_7b.branch_pool.conv',
                'Mixed_7c.branch1x1.conv', 'Mixed_7c.branch3x3_1.conv', 'Mixed_7c.branch3x3_2a.conv',
                'Mixed_7c.branch3x3_2b.conv', 'Mixed_7c.branch3x3dbl_1.conv', 'Mixed_7c.branch3x3dbl_2.conv',
                'Mixed_7c.branch3x3dbl_3a.conv', 'Mixed_7c.branch3x3dbl_3b.conv', 'Mixed_7c.branch_pool.conv',
                'fc']
first = ['Conv2d_1a_3x3.conv', 'Mixed_5b.branch1x1.conv', 'Mixed_5b.branch5x5_1.conv', 
                'Mixed_5b.branch5x5_1.conv', 'Mixed_5b.branch3x3dbl_1.conv', 'Mixed_5b.branch5x5_2.conv', 
                'Mixed_5c.branch5x5_2.conv', 'Mixed_5d.branch5x5_2.conv',
                'Mixed_5b.branch3x3dbl_2.conv', 'Mixed_5b.branch3x3dbl_3.conv', 'Mixed_5b.branch_pool.conv', 
                'Mixed_5c.branch1x1.conv', 'Mixed_5c.branch5x5_1.conv',  
                'Mixed_5c.branch5x5_1.conv', 'Mixed_5c.branch3x3dbl_1.conv', 'Mixed_5c.branch3x3dbl_2.conv', 
                'Mixed_5c.branch3x3dbl_3.conv', 'Mixed_5c.branch_pool.conv', 'Mixed_5d.branch1x1.conv', 
                'Mixed_5d.branch5x5_1.conv', 'Mixed_5d.branch3x3dbl_1.conv', 
                'Mixed_5d.branch3x3dbl_2.conv', 'Mixed_5d.branch3x3dbl_3.conv', 'Mixed_5d.branch_pool.conv', 
                'Mixed_6a.branch3x3.conv', 'Mixed_6a.branch3x3dbl_1.conv', 'Mixed_6a.branch3x3dbl_2.conv', 
                'Mixed_6a.branch3x3dbl_3.conv', 'Mixed_6b.branch1x1.conv', 'Mixed_6b.branch7x7_1.conv', 
                'Mixed_6b.branch7x7dbl_1.conv', 
                'Mixed_6b.branch_pool.conv', 'Mixed_6c.branch1x1.conv',
                'Mixed_6c.branch7x7_1.conv',  
                'Mixed_6c.branch7x7dbl_1.conv', 'Mixed_6c.branch_pool.conv',
                'Mixed_6d.branch1x1.conv',
                'Mixed_6d.branch7x7_1.conv',  
                'Mixed_6d.branch7x7dbl_1.conv', 'Mixed_6d.branch_pool.conv',
                'Mixed_6e.branch1x1.conv',
                'Mixed_6e.branch7x7_1.conv', 
                'Mixed_6e.branch7x7dbl_1.conv', 'Mixed_6e.branch_pool.conv',
                'Mixed_7a.branch3x3_1.conv', 'Mixed_7a.branch3x3_2.conv', 'Mixed_7a.branch7x7x3_1.conv',
                'Mixed_7a.branch7x7x3_4.conv',
                'Mixed_7b.branch1x1.conv', 'Mixed_7b.branch3x3_1.conv', 
                'Mixed_7b.branch3x3dbl_1.conv', 'Mixed_7b.branch3x3dbl_2.conv',
                 'Mixed_7b.branch_pool.conv',
                'Mixed_7c.branch1x1.conv', 'Mixed_7c.branch3x3_1.conv', 'Mixed_7c.branch3x3dbl_1.conv', 'Mixed_7c.branch_pool.conv']
second = ['fc']
with_padding = ['Mixed_6b.branch7x7_2.conv', 
                'Mixed_6b.branch7x7_3.conv', 
                'Mixed_6b.branch7x7dbl_2.conv', 'Mixed_6b.branch7x7dbl_3.conv',
                'Mixed_6b.branch7x7dbl_4.conv', 'Mixed_6b.branch7x7dbl_5.conv', 'Mixed_6c.branch7x7_2.conv', 
                'Mixed_6c.branch7x7_3.conv', 'Mixed_6c.branch7x7dbl_2.conv', 'Mixed_6c.branch7x7dbl_3.conv', 
                'Mixed_6c.branch7x7dbl_4.conv', 'Mixed_6c.branch7x7dbl_5.conv', 'Mixed_6d.branch7x7_2.conv', 
                'Mixed_6d.branch7x7_3.conv', 
                'Mixed_6d.branch7x7dbl_2.conv', 'Mixed_6d.branch7x7dbl_3.conv', 
                'Mixed_6d.branch7x7dbl_4.conv', 'Mixed_6d.branch7x7dbl_5.conv', 'Mixed_6e.branch7x7_2.conv', 
                'Mixed_6e.branch7x7_3.conv', 'Mixed_6e.branch7x7dbl_2.conv', 'Mixed_6e.branch7x7dbl_3.conv', 
                'Mixed_6e.branch7x7dbl_4.conv', 'Mixed_6e.branch7x7dbl_5.conv', 'Mixed_7a.branch7x7x3_2.conv', 
                'Mixed_7a.branch7x7x3_3.conv', 'Mixed_7b.branch3x3_2a.conv', 'Mixed_7b.branch3x3_2b.conv',
                'Mixed_7b.branch3x3dbl_3a.conv', 'Mixed_7b.branch3x3dbl_3b.conv', 'Mixed_7c.branch3x3_2a.conv',
                'Mixed_7c.branch3x3_2b.conv', 'Mixed_7c.branch3x3dbl_2.conv',
                'Mixed_7c.branch3x3dbl_3a.conv', 'Mixed_7c.branch3x3dbl_3b.conv',]

original_weights = []
for i in layer_list:
   original_weights.append(get_nested_attr(model, i).weight._data)

def no_faults():
    number =[]
 
    for i in layer_list:
        weights = get_nested_attr(model, i).weight._data
        fi = FI3(weights)
        
        nn = fi.param(weights)
        # print(nn)
        number.append(nn)
    total = sum(number) * 8
    return(total)

def generate_fault_list(n):
    
    no_faults_each_iteration = int(BER * n)
    print("each iteration:", no_faults_each_iteration)
    for j in range(no_faults_each_iteration):
        layer = random.choice(layer_list)
        # layer = 'features[15]'
        # print(layer)
        weights = get_nested_attr(model, layer).weight._data
        if layer in first: fi = FI(weights)
        elif layer in second: fi = FI2(weights)
        index, bit = fi.fault_position()
        fault_dict['Iteration'].append(k)
        fault_dict['Layer'].append(layer)
        fault_dict['Index'].append(index)
        fault_dict['Bit'].append(bit)

def test(n):
    layer_count = 0
    for i in layer_list:
        get_nested_attr(model, i).weight._data = original_weights[layer_count] 
        layer_count += 1

    p = 0
    #int(BER * n)
    for t in range(int(BER * n)):
        
        layer = random.choice(layer_list)
 
        weights = get_nested_attr(model, layer).weight._data
        # if layer in first : fi = FI(weights)
        # if layer in second : fi = FI2(weights)
        # if layer in with_padding : fi = FI3(weights)
        fi = FI3(weights)
        index, bit = fi.fault_position()
        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights
        p += 1
        #print("which fault", k, p)


    import timeit
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
    print('Total Time:', timeit.default_timer() - start_time)
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
print('Accuracy Drop: %.4f %%' % (93.7099 - avg_accuracy))    
