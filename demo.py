# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# load in training imgs, resize them with torchvision.transforms.Resize()
# take training imgs, put into separate datasets based upon img name
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resize = transforms.Resize((32, 32))

def main():

    # load model, setup net
    net = Net()
    net.to(device)
    # if the computer has cuda, load with cuda
    if torch.cuda.is_available():
        net.load_state_dict(torch.load('./model_best_train.pt'))
    else:
        net.load_state_dict(torch.load('./model_best_train.pt', map_location=torch.device('cpu')))

    # get image loader
    # for testing, change dog to cat to run w respective test img
    # this code is hella inefficient, just wanna have something working lol
    img = DataLoader(AnimalDataset('./animal_cat.jpg', transform=resize), batch_size=64, shuffle=True, num_workers=4)

    # get output
    for i, data in enumerate(img, 0):
        # run img thru nn
        output = net(data[0].to(device))

        # get predicted val
        _, predicted = torch.max(output, 1)

        # print output
        if predicted == 0:
            print("cat")
        else:
            print("dog")


# Our neural network definition
# m's are conv output channels
# f's are filter sizes
# fc's are fully connected layer output sizes for fcl1 and fcl2
# n is the maxpool size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5) # output is (33 - f1) by (33 - f1) by m1
        self.pool = nn.MaxPool2d(2)
        self.pool_output_dims = int((33 - 5) / 2)
        self.conv2 = nn.Conv2d(8, 12, 3) # output is (pool_output_dims - f2) ** 2 * m2
        self.conv2_output_size = ((self.pool_output_dims - 2) ** 2) * 12
        self.fc1 = nn.Linear(432, 25)
        self.fc2 = nn.Linear(25, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        # print('-a', self.conv2_output_size)
        x = x.transpose(1, 3)

        # convert x from numpy to cuda
        # x = x.to(device)

        # print('a', x.shape)
        x = self.conv1(x)
        # print('b', x.shape)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        # print('c', x.shape)
        x = x.view(-1, 12 * 6 * 6)
        # print('d', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('e', x.shape)
        return x

# crappy way of loading img
class AnimalDataset(Dataset):
    def __init__(self, file, transform=None):
        self.file = file
        self.transform = transform
        self.label = 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = Image.open(self.file)
        if self.transform:
            img = self.transform(img)
        img = np.array(img)
        return img.astype('float32'), self.label


if __name__ == "__main__":
    main()