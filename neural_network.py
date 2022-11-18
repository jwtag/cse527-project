import torch.nn as nn

from datasets.dataset_helper import DataCategory


# Our neural network definition
class Net(nn.Module):
    def __init__(self, acid_seq_length):
        super(Net, self).__init__()

        # this code is currently derived from
        # https://towardsdatascience.com/modeling-dna-sequences-with-pytorch-de28b0a05036

        # params
        kernel_size = 5  # size of the kernel to look with at the data.  kernel is used to look at multiple datapoints at once.  arbitrarily choosing 5, may change in the future.
        num_filters = 32  # number of out_channels from conv1 layer.  This number is chosen based off of intuition.

        # define the networks used on each drug type individually.  we do this so that they're computed separately.
        # TODO:  Maybe add more layers in the future, but this works for now.
        self.drug_type_ini_net = nn.Sequential(
            # in_channels should be equal to # of channels that represent an element.  This value should be 1 since we're only looking at 1 seq at a time.
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=num_filters * (acid_seq_length - kernel_size + 1), out_features=acid_seq_length)
        )
        self.drug_type_pi_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=num_filters * (acid_seq_length - kernel_size + 1), out_features=acid_seq_length)
        )
        self.drug_type_rti_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=num_filters * (acid_seq_length - kernel_size + 1), out_features=acid_seq_length)
        )

    def forward(self, x):
        return {
            DataCategory.INI: self.drug_type_ini_net(x),
            DataCategory.PI: self.drug_type_pi_net(x),
            DataCategory.RTI: self.drug_type_rti_net(x)
        }