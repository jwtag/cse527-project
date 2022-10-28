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

device = "cpu" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    learning_rate = 0.00005
    momentum = 0.9
    batch_size = 128
    acid_seq_length = 288  # length of acid sequence.  NOTE:  THIS VARIES BY PROTEIN, WILL NEED TO EDIT!

    print (train_dir)
    print (train_files)

    # turn the training mutation data into a "Dataset" struct for training.
    train_dataset = MutationDataset(train_files, train_dir)

    # setup training DataLoader.
    # TODO:  Set batch_size to be programmatically equal to DNA seq len (since we're processing in DNA-seq-len batches).
    # train_dataset = the dataset we trained upon.
    # batch_size = the sizes of the batches of data being processed at-a-time
    # shuffle = shuffle the data to prevent any sort of ordering bias
    # drop_last = ignore all remaining elements when "<# elements> % <batch_size> != 0".  this prevents unintended
    #             processing errors.
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # turn the training mutation data into a "Dataset" struct for training.
    #test_dataset = MutationDataset(test_files, test_dir)

    # setup test DataLoader.  TODO:  figure out why it's having weird formatting errors with copied file
    # testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # setup the model.
    net = Net(acid_seq_length, batch_size)
    net.to(device)

    # setup the loss function used to evaluate the model's accuracy.
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # setup an optimizer to speed-up the model's performance.
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Train the model + test it afterwards.  If the model has the best specs seen so far, save it to disk.
    # TODO:  Do this.
    train(trainloader, trainloader, net, criterion, optimizer)  # TODO: make use the testloader once formatting issues are resolved:  trainloader, testloader, net, criterion, optimizer)


# Our neural network definition
class Net(nn.Module):
    def __init__(self, acid_seq_length, batch_size):
        super(Net, self).__init__()

        # this code is currently derived from
        # https://towardsdatascience.com/modeling-dna-sequences-with-pytorch-de28b0a05036

        # params
        num_filters = 32
        kernel_size = 5  # size of the kernel to look with at the data.  kernel is used to look at multiple datapoints at once.  arbitrarily choosing 5, may change in the future.

        # define the layers
        self.conv1 = nn.Conv1d(in_channels=batch_size, out_channels=acid_seq_length, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # honestly, figuring out the num in_features is a total PITA.  let's just duplicate this file for the two cases + set this value manually based upon the size of the data processed.
        self.linear1 = nn.Linear(in_features=284, out_features=batch_size)  # out_features = batch_size


    def forward(self, x):
        # permute to put channel in correct order.

        # TODO:  Maybe add more layers in the future, but this works for now.

        # apply the layers
        x = self.conv1(x)
        # print("conv1")
        # print(x.size())

        x = self.relu(x)
        # print("relu")
        # print(x.size())

        x = self.flatten(x)
        # print("flatten")
        # print(x.size())

        x = self.linear1(x)
        # print("linear1")
        # print(x.size())

        # permute the result.
        # The current channels are (batch size x acid seq len) when it should be (acid seq len x batch_size).
        # (w/o doing this, the loss function won't work...)
        x = x.permute(-1, 0)

        return x


# Calculates the accuracy given a data loader and a neural network
def calculate_accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test sequences: %d %%' % accuracy)
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

            # get the inputs; "data" is just an object containing a list of [acid values array, treatment label]
            # these should "just work" with tensors, which is good
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            print(outputs.shape, labels.shape)
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