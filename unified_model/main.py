# add the dataset_helper to the Python path so we can use it.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# do other imports.
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_helper import DataCategory
from config import UnifiedConfig
from neural_network import Net

def main():
    # create train + test datasets

    # first, load all data into one dataset.
    all_data_dataset = UnifiedConfig.model_dataset_class(UnifiedConfig.model_data_file, UnifiedConfig.use_binary_labels)
    
    # randomly divide the dataset into training + test at the ratio defined in config.
    train_size = int(UnifiedConfig.training_data_proportion * len(all_data_dataset))
    test_size = len(all_data_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_data_dataset, [train_size, test_size])

    # setup train + test DataLoaders.

    # UnifiedConfig.batch_size = the sizes of the batches of data being processed at-a-time
    # shuffle = shuffle the data to prevent any sort of ordering bias
    trainloader = DataLoader(train_dataset, batch_size=UnifiedConfig.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=UnifiedConfig.batch_size, shuffle=True, num_workers=4)


    # setup the neural network.
    acid_seq_length = all_data_dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(UnifiedConfig.device)

    # setup an optimizer to speed-up the model's performance.
    optimizer = optim.SGD(net.parameters(), lr=UnifiedConfig.learning_rate, momentum=UnifiedConfig.momentum)


    # Train the model + refine it.  If the model has the best accuracy seen so far on the test data, it is saved to disk.
    train(trainloader, testloader, net, optimizer)

# Calculates the accuracy given a data loader and a neural network
# loader = dataloader
# loader_type = the type of data in the loader ("train" or "test")
# net = the neural network being evaluated
def calculate_accuracy(loader, loader_data_type, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data['mutation_seq'].to(UnifiedConfig.device), data['labels']
            outputs = net(inputs)

            # get the predicted drug for each drug type
            _, predicted_type_1 = torch.max(outputs[DataCategory.INI], 1)
            _, predicted_type_2 = torch.max(outputs[DataCategory.PI], 1)
            _, predicted_type_3 = torch.max(outputs[DataCategory.RTI], 1)

            # store if the predictions were correct
            total += labels[DataCategory.INI].size(0)
            correct += (predicted_type_1 == labels[DataCategory.INI]).sum().item()
            total += labels[DataCategory.PI].size(0)
            correct += (predicted_type_2 == labels[DataCategory.PI]).sum().item()
            total += labels[DataCategory.RTI].size(0)
            correct += (predicted_type_3 == labels[DataCategory.RTI]).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the ' + loader_data_type + ' sequences: %d %%' % accuracy)
    return accuracy


# setup the loss function used to evaluate the model's accuracy.
def criterion(outputs, labels):
    loss_func = nn.CrossEntropyLoss()
    losses = 0.0
    for i, key in enumerate(outputs):
        losses += loss_func(outputs[key], labels[key])
    return losses


def train(trainloader, testloader, net, optimizer):
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
    while epoch < UnifiedConfig.num_training_epochs and best_train < 100 and best_test < 100:  # loop over the dataset multiple times
        print("Epoch " + str(epoch + 1))
        for i, data in enumerate(trainloader, 0):

            # get the inputs; "data" is just an object containing a list of [acid values array, treatment label]
            # these should "just work" with tensors, which is good
            inputs = data['mutation_seq'].to(UnifiedConfig.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, data['labels'])
            loss.to(UnifiedConfig.device)
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
            torch.save(net.state_dict(), "../{}_unified_model_best_test.pt".format(UnifiedConfig.current_configuration_write_file_prefix))
        if best_train < train_accuracy:
            best_train = train_accuracy
            torch.save(net.state_dict(), "../{}_unified_model_best_train.pt".format(UnifiedConfig.current_configuration_write_file_prefix))
        if best_balanced_train <= train_accuracy and best_balanced_test <= test_accuracy:
            best_balanced_train = train_accuracy
            best_balanced_test = test_accuracy
            torch.save(net.state_dict(), "../{}_unified_model_best_balanced.pt".format(UnifiedConfig.current_configuration_write_file_prefix))
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
    plt.savefig("{}_chart.png".format(UnifiedConfig.current_configuration_write_file_prefix))


if __name__ == "__main__":
    main()