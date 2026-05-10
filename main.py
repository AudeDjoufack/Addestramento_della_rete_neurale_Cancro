import csv

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(os.path.exists("data/labels.csv"))
print(os.path.exists("data"))

csv_file = "data/labels.csv"
img_dir = "data"

df = pd.read_csv(csv_file)
#df.columns = df.iloc[0]   # usa la prima riga come header
#df = df[1:]               # rimuovi la prima riga dai dati
print(df.head())

#df = df[["filepath", "case category"]]
#df.columns = ["filename", "label"]

class BreastDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.data['case category'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filepath']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_str = self.data.iloc[idx]['case category']
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
batch_size=4
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('benign', 'malignant', 'normal')
#dataset = BreastDataset(df, img_dir="data", transform=transform)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_df = df[df["split"] == "train"]
test_df  = df[df["split"] == "test"]
train_dataset = BreastDataset(train_df, img_dir="data", transform=transform)
test_dataset  = BreastDataset(test_df, img_dir="data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))

print(classes)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels

print(labels)
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#images, labels = next(iter(loader))
#print(images.shape)
#print(labels)

#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.train()
dati0 = []
dati_loss0 = []
dati_accuracy0 = []
for epoch in range(50):  # loop over the dataset multiple times
    correct = 0
    total = 0
    loss_total = 0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            dati0.append((epoch, running_loss / 4))
            running_loss = 0.0
    accuracy = correct / total
    loss_media = loss_total / len(train_loader)
    dati_loss0.append((epoch, loss_media))
    dati_accuracy0.append((epoch, accuracy))

print('Finished Training')

with open("Dati_Training_loss.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati0:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)
with open("Dati_Training_total_loss.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati_loss0:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)
with open("Dati_Training_accuracy.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati_accuracy0:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)


PATH = './eco_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
net.eval()
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
#with torch.no_grad():
#    for data in test_loader:
#        images, labels = data
#        # calculate outputs by running images through the network
#        outputs = net(images)
#        # the class with the highest energy is what we choose as prediction
#        _, predicted = torch.max(outputs, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()

#print(f'Accuracy of the network on the 156 test images: {100 * correct // total} %')
dati1 = []
dati_loss1 = []
dati_accuracy1 = []
with torch.no_grad():
 for epoch in range(50):  # loop over the dataset multiple times
     correct = 0
     total = 0
     loss_total = 0
     running_loss = 0.0
     for i, data in enumerate(test_loader, 0):
         # get the inputs; data is a list of [inputs, labels]
         inputs, labels = data
         inputs = inputs.to(device)
         labels = labels.to(device)
         outputs = net(inputs)
         loss = criterion(outputs, labels)
         _, predicted = torch.max(outputs, 1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()
         loss_total +=loss.item()

         # print statistics
         running_loss += loss.item()
         if i % 5 == 4:    # print every 2000 mini-batches
             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
             dati1.append((epoch, running_loss / 4))
             running_loss = 0.0
     accuracy = correct / total
     loss_media = loss_total / len(test_loader)
     dati_loss1.append((epoch, accuracy))
     dati_accuracy1.append((epoch, loss_media))
print(f'Accuracy of the network on the 156 test images: {100 * correct // total} %')

with open("Dati_Test_loss.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati1:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)
with open("Dati_Test_total_loss.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati_loss1:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)
with open("Dati_Test_accuracy.csv", mode='w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for s in dati_accuracy1:
        writer.writerow([s[0], s[1]])
    #writer.writerow(dati)

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

