import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import quanto

import random
from fi import FI
from fi2 import FI2
import pandas
import csv

# Setting random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

acc_dict = {"noFI": [], "Accuracy": []}
csv_acc = "alex-normal-fi50-ber00001.csv"

# fault_dict = {"Iteration": [], "Layer": [], "Index": [], "Bit": []}
# csv_fault = "resnet18_fault_list50_ber003.csv"

output_results_file = open("output_alex_normal_ber00001", "w")
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

output_results_file = open("output_alex_normal_ber00001", "a")
output_results = csv.DictWriter(output_results_file, ["fault_id", "img_id", "predicted", *[f"p{i}" for i in range(10)]])

Sufficient_no_faults = 20
BER = 0.00001

#fault_list = pandas.read_csv('resnet18_fault_list50_ber0001.csv')


#model = vgg11_bn(pretrained=True)

#model.eval()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(torch.__version__)
# print(DEVICE)


train_csv = pd.read_csv('./kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_csv = pd.read_csv('./kaggle/input/fashionmnist/fashion-mnist_test.csv')


# print(train_csv.shape)
# print(test_csv.shape)

# Customize training size here
inputSize = 8000
train_csv=train_csv[:inputSize]
# len(train_csv)


# print(train_csv.info())
# print(train_csv.head())


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):        
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        
        label, image = [], []
        
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        
        if self.transform is not None:
            # transfrom the numpy array to PIL image before the transform function
            pil_image = Image.fromarray(np.uint8(image)) 
            image = self.transform(pil_image)
            
        return image, label


AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform), 
    batch_size=100, shuffle=False)

test_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform), 
    batch_size=100, shuffle=False)


class fasion_mnist_alexnet(nn.Module):  
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out



def testt(model, device, test_loader):
    # model.eval()
    # test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset)  # loss之和除以data数量 -> mean
        # accuracy_val.append(100. * correct / len(test_loader.dataset))
        # print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            # test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        # print(test_loss)
        # print(correct)
        # print(accuracy_val)
        acc = 100. * correct / len(test_loader.dataset)
        return(acc)
# print(model)
# print(model.fc.weight._data)
# print(model.features[25].weight._data)
# 0, 4, 8, 11, 15, 18, 22, 25
# print(model.classifier[0].weight)
# 0, 3, 6

# layer = 'features[0]'

model = torch.load('qalex-normal.pth')

# print(model)
# print(model.conv1[0].weight._data)



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

layer_list = ['conv1[0]', 'conv2[0]', 'conv3[0]', 'conv4[0]', 'conv5[0]', 'fc1', 'fc2', 'fc3']
first = ['conv1[0]', 'conv2[0]', 'conv3[0]', 'conv4[0]', 'conv5[0]']
second = ['fc1', 'fc2', 'fc3']



# layer_count = 0
original_weights = []
for i in layer_list:
    original_weights.append(get_nested_attr(model, i).weight._data)
    # layer_count += 1

# layer_list = ['conv1', 'layer1[0].conv1', 'layer1[0].conv2', 'layer1[1].conv1', 'layer1[1].conv2', 'layer2[0].conv1', 'layer2[0].conv2', 'layer2[1].conv1', 'layer2[1].conv2', 'layer3[0].conv1', 'layer3[0].conv2', 'layer3[1].conv1', 'layer3[1].conv2', 'layer4[0].conv1', 'layer4[0].conv2', 'layer4[1].conv1', 'layer4[1].conv2', 'fc']
#layer = random.choice(layer_list)
#print(layer)
#layer_weights = get_nested_attr(model, layer).weight._data.numpy()

# # # Plot the distribution of weights
#plt.hist(layer_weights.flatten(), bins=50)
#plt.xlabel('Weight Value')
#plt.ylabel('Frequency')
#plt.title('Distribution of Weights')
#plt.show()

#import timeit
#correct = 0
#total = 0

#model.eval()
#start_time = timeit.default_timer()
#with torch.no_grad():
 #   for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
 #      images, labels = images.to("cpu"), labels.to("cpu")
 #      outputs = model(images)
  #      _, predicted = torch.max(outputs.data, 1)
  #      total += labels.size(0)
  #      correct += (predicted == labels).sum().item()
#print(timeit.default_timer() - start_time)
#print('Accuracy of the network on the 10000 test images: %.4f %%' % (
#    100 * correct / total))

    
# # print(model.conv1.weight)

#quanto.quantize(model, weights=quanto.qint8, activations=None)
#quanto.freeze(model)
#torch.save(model, 'vgg-q-normal.pth')

# torch.load('qresnet-80-127.pth')

#print(model.conv1.weight)

# b = io.BytesIO()
# torch.save(model.state_dict(), b)
# b.seek(0)
# state_dict2 = torch.load(b)


# loaded_state_dict2 = torch.load('qresnet-80-127.pth')
# model.load_state_dict(loaded_state_dict2)

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

def no_faults():
    number =[]
 
    for i in layer_list:
        weights = get_nested_attr(model, i).weight._data
        if i in first: fi = FI(weights)
        elif i in second: fi = FI2(weights)
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
    # model = torch.load('vgg-q-normal.pth')
    layer_count = 0
    for i in layer_list:
        get_nested_attr(model, i).weight._data = original_weights[layer_count] 
        layer_count += 1
    # loaded_state_dict2 = torch.load('qresnet-80-127.pth')


# # model2 = resnet50(pretrained=True)
# print(model.conv1.weight)
    # model.load_state_dict(loaded_state_dict2)
    # print(model.conv1.weight._data)
    p = 0
    for t in range(int(BER * n)):
        #layer = fault_list['Layer'][k+t]
        
        layer = random.choice(layer_list)
        #print(layer)
        
        weights = get_nested_attr(model, layer).weight._data
        if layer in first : fi = FI(weights)
        if layer in second : fi = FI2(weights)
        index, bit = fi.fault_position()
        #index = fault_list['Index'][k+t]
        #bit = fault_list['Bit'][k+t]
        new_weights = fi.inject(index, bit)
        get_nested_attr(model, layer).weight._data = new_weights
        p += 1
        #print("which fault", k, p)

#layer_weights = model.conv1.weight._data.numpy()

# Plot the distribution of weights
#plt.hist(layer_weights.flatten(), bins=50)
#plt.xlabel('Weight Value')
#plt.ylabel('Frequency')
#plt.title('Distribution of Weights')
#plt.show()


    import timeit

    # model.eval()
    start_time = timeit.default_timer()
    dataloader =test_loader
    model.eval()
    acc = testt(model, DEVICE, dataloader)
    # print(acc)ts.writerow(csv_output)
    print('Total Time:', timeit.default_timer() - start_time)
    print('Accuracy: %.4f %%' % (
    acc))
    return(acc)

n = no_faults()
print(n)
for k in range(Sufficient_no_faults):
    print(f"{Sufficient_no_faults - k} faults to inject")
    accuracy = test(n)
    acc_dict["Accuracy"].append(accuracy)
    acc_dict["noFI"].append(k)
    # generate_fault_list(n)

# data = pandas.DataFrame(fault_dict)
# data.to_csv(csv_fault)
data = pandas.DataFrame(acc_dict)
data.to_csv(csv_acc)
avg_accuracy = sum(acc_dict["Accuracy"])/len(acc_dict["Accuracy"])
print('Average Faulty Accuracy: %.4f %%' % (avg_accuracy))
print('Accuracy Drop: %.4f %%' % (95.3250 - avg_accuracy))    
