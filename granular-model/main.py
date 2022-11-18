import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset_helper import DataCategory
from neural_network import Net
from datasets.granular_model_mutation_dataset import GranularModelMutationDataset


device = "cpu" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    learning_rate = 0.00005
    momentum = 0.9
    batch_size = 128
    use_binary_labels = True  # if the labels should not be drug-specific, but scoped to drug-type-specific instead.
                               # (ex: <drugname>101 = <drugname><drug><no drug><drug>)
    ini_filename = 'model-data/INI.csv'
    pi_filename = 'model-data/PI.csv'
    rti_filename = 'model-data/RTI.csv'

    create_and_save_nn(DataCategory.INI, ini_filename, learning_rate, momentum, batch_size, use_binary_labels)
    create_and_save_nn(DataCategory.PI, pi_filename, learning_rate, momentum, batch_size, use_binary_labels)
    create_and_save_nn(DataCategory.RTI, rti_filename, learning_rate, momentum, batch_size, use_binary_labels)


# creates, iterates, and saves a nn for the data for the specified category + file
def create_and_save_nn(category, filename, learning_rate, momentum, batch_size, use_binary_labels):
    # create train + test datasets

    # first, load the data into one dataset.
    all_data_dataset = GranularModelMutationDataset(filename, use_binary_labels)

    # randomly divide the dataset into training + test at an 80%/20% ratio.
    train_size = int(0.8 * len(all_data_dataset))
    test_size = len(all_data_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_data_dataset, [train_size, test_size])

    # setup train + test DataLoaders.

    # batch_size = the sizes of the batches of data being processed at-a-time
    # shuffle = shuffle the data to prevent any sort of ordering bias
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # setup the neural network.
    acid_seq_length = all_data_dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(device)

    # setup an optimizer to speed-up the model's performance.
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    # Train the model + refine it.  If the model has the best accuracy seen so far on the test data, it is saved to disk.
    train(category, trainloader, testloader, net, optimizer)


# Calculates the accuracy given a data loader and a neural network
# category = the Enum representing the drug category
# loader = dataloader
# loader_type = the type of data in the loader ("train" or "test")
# net = the neural network being evaluated
def calculate_accuracy(category, loader, loader_data_type, net):
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
    print('Accuracy of the network on the ' + str(category) + ' ' + loader_data_type + ' sequences: %d %%' % accuracy)
    return accuracy


# setup the loss function used to evaluate the model's accuracy.
def criterion(outputs, labels):
    loss_func = nn.CrossEntropyLoss()
    losses = 0.0
    for i, key in enumerate(outputs):
        losses += loss_func(outputs[key], labels[key])
    return losses


def train(category, trainloader, testloader, net, optimizer):
    print("Starting to train")
    test_accuracies = []
    train_accuracies = []
    epochs = []
    epoch = 0

    # stores accuracies of "best train" model.  this model can most accurately predict the training data.
    best_train = 0

    # stores accuracies of "best test" model.  this model can most accurately predict the testing data.
    best_test = 0

    # stores accuracies of "best balanced" model.  "best balanced" == instance where train & test are better than ever.
    # (we do this to have a model that isn't overfit)
    best_balanced_train = 0
    best_balanced_test = 0
    while epoch < 75 and best_train < 100 and best_test < 100:  # loop over the dataset multiple times
        print("Epoch " + str(epoch + 1))
        for i, data in enumerate(trainloader, 0):

            # get the inputs; "data" is just an object containing a list of [acid values array, treatment label]
            # these should "just work" with tensors, which is good
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

        # evaluate the training + testing accuracies of the model.
        train_accuracy = calculate_accuracy(trainloader, "training", net)
        test_accuracy = calculate_accuracy(testloader, "testing", net)
        print(train_accuracy, test_accuracy)

        # update stored model if new best
        if best_test < test_accuracy:
            best_test = test_accuracy
            torch.save(net.state_dict(), "../{}_model_best_test.pt".format(str(category)))
        if best_train < train_accuracy:
            best_train = train_accuracy
            torch.save(net.state_dict(), "../{}_model_best_train.pt".format(str(category)))
        if best_balanced_train <= train_accuracy and best_balanced_test <= test_accuracy:
            best_balanced_train = train_accuracy
            best_balanced_test = test_accuracy
            torch.save(net.state_dict(), "../{}_model_best_balanced.pt".format(str(category)))
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
    plt.savefig("{}_chart.png".format(str(category)))
    plt.show()


if __name__ == "__main__":
    main()