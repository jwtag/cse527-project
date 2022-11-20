# add the parent directory to the Python path so we can use dataset_helper.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# other imports.
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_helper import DataCategory
from config import GranularConfig
from neural_network import Net
from datasets.granular_model_mutation_dataset import GranularModelMutationDataset


def main():
    create_and_save_nn(DataCategory.INI, GranularConfig.ini_data_file)
    create_and_save_nn(DataCategory.PI, GranularConfig.pi_data_file)
    create_and_save_nn(DataCategory.RTI, GranularConfig.rti_data_file)


# creates, iterates, and saves a nn for the data for the specified category + file
def create_and_save_nn(category, filename):
    # create train + test datasets

    # first, load the data into one dataset.
    all_data_dataset = GranularModelMutationDataset(filename, GranularConfig.use_binary_labels)

    # randomly divide the dataset into training + test at an 80%/20% ratio.
    train_size = int(GranularConfig.training_data_proportion * len(all_data_dataset))
    test_size = len(all_data_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_data_dataset, [train_size, test_size])

    # setup train + test DataLoaders.

    # batch_size = the sizes of the batches of data being processed at-a-time
    # shuffle = shuffle the data to prevent any sort of ordering bias
    trainloader = DataLoader(train_dataset, batch_size=GranularConfig.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=GranularConfig.batch_size, shuffle=True, num_workers=4)


    # setup the neural network.
    acid_seq_length = all_data_dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(GranularConfig.device)


    # setup the loss function used to evaluate the model's accuracy.
    criterion = nn.CrossEntropyLoss()
    criterion.to(GranularConfig.device)


    # setup an optimizer to speed-up the model's performance.
    optimizer = optim.SGD(net.parameters(), lr=GranularConfig.learning_rate, momentum=GranularConfig.momentum)


    # Train the model + refine it.  If the model has the best accuracy seen so far on the datasets, it is saved to disk.
    train(category, trainloader, testloader, net, criterion, optimizer)


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
            inputs, labels = data[0].to(GranularConfig.device), data[1].to(GranularConfig.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the ' + str(category) + ' ' + loader_data_type + ' sequences: %d %%' % accuracy)
    return accuracy


def train(category, trainloader, testloader, net, criterion, optimizer):
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
    while epoch < GranularConfig.num_training_epochs and best_train < 100 and best_test < 100:  # loop over the dataset multiple times
        print("Epoch " + str(epoch + 1))
        for i, data in enumerate(trainloader, 0):

            # get the inputs; "data" is just an object containing a list of [acid values array, treatment label]
            # these should "just work" with tensors, which is good
            inputs = data[0].to(GranularConfig.device)
            labels = data[1].to(GranularConfig.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.to(GranularConfig.device)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i)

        # evaluate the training + testing accuracies of the model.
        train_accuracy = calculate_accuracy(category, trainloader, "training", net)
        test_accuracy = calculate_accuracy(category, testloader, "testing", net)
        print(train_accuracy, test_accuracy)

        # update stored model if new best
        if best_test < test_accuracy:
            best_test = test_accuracy
            torch.save(net.state_dict(), "../{}_model_best_test.pt".format(GranularConfig.current_configuration_write_file_prefix + '_' + str(category)))
        if best_train < train_accuracy:
            best_train = train_accuracy
            torch.save(net.state_dict(), "../{}_model_best_train.pt".format(GranularConfig.current_configuration_write_file_prefix + '_' + str(category)))
        if best_balanced_train <= train_accuracy and best_balanced_test <= test_accuracy:
            best_balanced_train = train_accuracy
            best_balanced_test = test_accuracy
            torch.save(net.state_dict(), "../{}_model_best_balanced.pt".format(GranularConfig.current_configuration_write_file_prefix + '_' + str(category)))
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


if __name__ == "__main__":
    main()