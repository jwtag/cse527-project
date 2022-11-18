import torch.nn as nn


# Our neural network definition
class Net(nn.Module):
    def __init__(self, acid_seq_length):
        super(Net, self).__init__()

        # this code is currently derived from
        # https://towardsdatascience.com/modeling-dna-sequences-with-pytorch-de28b0a05036

        # params
        kernel_size = 5  # size of the kernel to look with at the data.  kernel is used to look at multiple datapoints at once.  arbitrarily choosing 5, may change in the future.
        num_filters = 32  # number of out_channels from conv1 layer.  This number is chosen based off of intuition.

        # define the layers

        # in_channels should be equal to # of channels that represent an element.  This value should be 1 since we're only looking at 1 seq at a time.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=num_filters * (acid_seq_length - kernel_size + 1), out_features=acid_seq_length)

    def forward(self, x):
        # TODO:  Maybe add more layers in the future, but this works for now.

        # apply the layers
        x = self.conv1(x)

        x = self.relu(x)

        # make the tensor 1D so it can be processed by the linear layer.
        # (this is safe since the flattened dimension is only size 1)
        x = self.flatten(x)

        x = self.linear1(x)

        return x
