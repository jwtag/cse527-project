import torch.nn as nn

from config import GranularConfig


# Our neural network definition
class Net(nn.Module):
    def __init__(self, acid_seq_length):
        super(Net, self).__init__()

        # this code is currently derived from
        # https://towardsdatascience.com/modeling-dna-sequences-with-pytorch-de28b0a05036

        # define the layers

        # in_channels should be equal to # of channels that represent an element.  This value should be 1 since we're only looking at 1 seq at a time.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=GranularConfig.num_filters, kernel_size=GranularConfig.kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=GranularConfig.num_filters * (acid_seq_length - GranularConfig.kernel_size + 1), out_features=1000)

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
