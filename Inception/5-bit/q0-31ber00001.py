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

from models.inception import inception_v3

# import quanto
import matplotlib.pyplot as plt

from fi031 import FI
#from fi0312 import FI2
from dmr031 import DMR
#from dmr0312 import DMR2
import pandas
import csv

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "Inception-0-31-fi50-ber00001.csv"


output_results_file = open("output_Inception_0-31_ber00001", "w")
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

output_results_file = open("output_Inception_0-31_ber00001", "a")
output_results = csv.DictWriter(output_results_file, ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]])

Sufficient_no_faults = 20
BER = 0.00001


model = torch.load('qinception-0-31.pth')

layer_list = ['Conv2d_1a_3x3.conv', 'Mixed_5b.branch1x1.conv', 'Mixed_5b.branch5x5_1.conv', 
              'Mixed_5b.branch5x5_2.conv', 'Mixed_5b.branch3x3dbl_1.conv',
                'Mixed_5b.branch3x3dbl_2.conv', 'Mixed_5b.branch3x3dbl_3.conv', 'Mixed_5b.branch_pool.conv', 
                'Mixed_5c.branch1x1.conv', 'Mixed_5c.branch5x5_1.conv', 'Mixed_5c.branch5x5_2.conv', 
                'Mixed_5c.branch3x3dbl_1.conv', 'Mixed_5c.branch3x3dbl_2.conv', 
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
    dmr = DMR(weights)
    
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
        fi = FI(weights)
        
        nn = fi.param(weights)
        # print(nn)
        number.append(nn)
    total = sum(number) * 7
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
        get_nested_attr(model, i).weight._data = original_tmr_weights[layer_count] 
        layer_count += 1
    
    p = 0
    for t in range(int(BER * n)):
        layer = random.choice(layer_list)

        weights = get_nested_attr(model, layer).weight._data
        fi = FI(weights)
        
        index, bit = fi.fault_position()

        #bit = fault_list['Bit'][k+t]
        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights
        p += 1
    start_time1 = timeit.default_timer()

    for l in layer_list:
        print(l)
        weights = get_nested_attr(model, l).weight._data
        dmr = DMR(weights)
        
        new_weights = dmr.protect()
        get_nested_attr(model, l).weight._data = new_weights)


    
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
print('Accuracy Drop: %.4f %%' % (93.2492 - avg_accuracy))    
