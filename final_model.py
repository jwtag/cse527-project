import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from mutation_dataset import MutationDataset

train_dir = './model-data/train'
test_dir = './model-data/test'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

device = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    learning_rate = 0.00005
    momentum = 0.9

    print (train_dir)
    print (train_files)

    # turn the training mutation data into a "Dataset" struct for training.
    train_dataset = MutationDataset(train_files, train_dir)

    # setup training DataLoader.
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # turn the training mutation data into a "Dataset" struct for training.
    test_dataset = MutationDataset(test_files, test_dir)

    # setup test DataLoader.
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Train the model + test it afterwards.  If the model has the best specs seen so far, save it to disk.
    # TODO:  Do this.
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
        self.conv1 = nn.Conv2d(3, 15, 7) # output is (33 - f1) by (33 - f1) by m1
        self.pool = nn.MaxPool2d(2)
        self.pool_output_dims = int((65 - 7) / 2)
        self.conv2 = nn.Conv2d(15, 12, 5) # output is (pool_output_dims - f2) ** 2 * m2
        self.conv2_output_size = int((self.pool_output_dims + 1 - 5)/2)
        self.conv3 = nn.Conv2d(12, 8, 3)
        self.conv3_output_size = int((self.conv2_output_size + 1 - 3) / 2)
        self.fc1 = nn.Linear(self.conv3_output_size ** 2 * 8, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 80)
        self.fc4 = nn.Linear(80, 50)
        self.fc5 = nn.Linear(50, 2)

    def forward(self, x):
        # print('-a', self.conv2_output_size)
        x = x.transpose(1, 3)

        # convert x from numpy to cuda
        x = x.to(device)
        #print('pool output dims', self.pool_output_dims)
        #print('a', x.shape)
        x = self.conv1(x)
        #print('b', x.shape)
        x = self.pool(F.relu(x))
        #print('hey', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print('c', x.shape)
        #print(int((((self.pool_output_dims-2)/2) ** 2) * 12))
        x = x.view(-1, self.conv3_output_size ** 2 * 8)
        #print('d', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output


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
            print(data[0])
            print(data[1])

            # get the inputs; data is a list of [acid values array, treatment]
            # these should "just work" with tensors, which is good

            # convert data[0] (inputs) to torch-compatible format
            data = torch.tensor(data)

            inputs = data[0].to(device)
            labels = data[1].to(device)

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
    plt.title("Model Accuracy per Epoch")
    plt.savefig("chart.png")
    plt.show()


if __name__ == "__main__":
    main()