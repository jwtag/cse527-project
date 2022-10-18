import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# load in training imgs, resize them with torchvision.transforms.Resize()
# take training imgs, put into separate datasets based upon img name
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

train_dir = 'model-data/train'
test_dir = 'model-data/test'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    learning_rate = 0.0018739
    momentum = 0.9

    data_transform = transforms.Compose([transforms.Resize((32, 32))])

    cat_files = [tf for tf in train_files if 'cat' in tf]
    cat_train_files = cat_files[:10000]
    cat_test_files = cat_files[10000:]
    dog_files = [tf for tf in train_files if 'dog' in tf]
    dog_train_files = dog_files[:10000]
    dog_test_files = dog_files[10000:]

    cats_train = CatDogDataset(cat_train_files, train_dir, transform=data_transform)
    dogs_train = CatDogDataset(dog_train_files, train_dir, transform=data_transform)

    catdogs_train = ConcatDataset([cats_train, dogs_train])

    trainloader = DataLoader(catdogs_train, batch_size=64, shuffle=True, num_workers=4)

    cats_test = CatDogDataset(cat_test_files, train_dir, transform=data_transform)
    dogs_test = CatDogDataset(dog_test_files, train_dir, transform=data_transform)

    catdogs_test = ConcatDataset([cats_test, dogs_test])

    testloader = DataLoader(catdogs_test, batch_size=64, shuffle=True, num_workers=4)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    train(trainloader, testloader, net, criterion, optimizer)


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


# Calculates the accuracy given a data loader and a neural network
def calculate_accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    return accuracy


def train(trainloader, testloader, net, criterion, optimizer):
    print("Starting to train")
    test_accuracies = []
    train_accuracies = []
    epochs = []
    epoch = 0
    best_train = 0
    best_test = 0
    while epoch < 75 and best_train < 100 and best_test < 100:  # loop over the dataset multiple times
        print("Epoch " + str(epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.to(device)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i)

        train_accuracy = calculate_accuracy(trainloader, net)
        test_accuracy = calculate_accuracy(testloader, net)
        print(train_accuracy, test_accuracy)

        # update stored model if new best
        if best_test < test_accuracy:
            best_test = test_accuracy
            torch.save(net.state_dict(), "model_best_test.pt")
        if best_train < train_accuracy:
            best_train = train_accuracy
            torch.save(net.state_dict(), "model_best_train.pt")
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        epochs.append(epoch)
        epoch += 1
    print('Finished Training')
    print('best train = ', best_train, "best test = ", best_test)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.plot(epochs, train_accuracies)
    plt.plot(epochs, test_accuracies)
    plt.legend(["Train", "Test"])
    plt.title("OG Model Accuracy per Epoch")
    plt.savefig("og_chart.png")
    plt.show()


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

        # preprocess imgs for significant speedup
        self.imgs = []
        for file in file_list:
            img = Image.open(os.path.join(self.dir, file))
            if self.transform:
                img = self.transform(img)
            img = np.array(img)
            img = img.astype('float32')
            if self.mode == 'train':
                self.imgs.append((img, self.label))
            else:
                self.imgs.append((img, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.imgs[idx]


if __name__ == "__main__":
    main()