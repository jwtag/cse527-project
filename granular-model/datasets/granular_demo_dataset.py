# this constructor is used by the demo to create a dataset containing one seq to feed into PyTorch.
import torch
import numpy as np

from torch.utils.data import Dataset

from dataset_helper import LabelEncoder, get_acid_mutation_value


class GranularDemoDataset(Dataset):
    def __init__(self, mutation_seq):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # first col = treatment, rest of cols = mutations

        # turn row into array of ints so it can be processed by pytorch
        mutation_csv_row_as_ints = []
        for acid_mutation in mutation_seq:
            mutation_csv_row_as_ints.append(get_acid_mutation_value(acid_mutation))

        # reformat the python data list as a tensor.
        # This code line has some quirks, here's some explanations about what's going on:
        # - we're converting the ints to float32s since the macOS ARM/"Apple Silicon" GPU-based PyTorch code
        #   doesn't support float64 (int) processing.  float32 works across all platforms, so this is safe for
        #   other machines as well.
        # - "from_numpy()" creates a PyTorch tensor from the Numpy array.
        mutation_csv_row_tensor = torch.from_numpy(np.asarray(mutation_csv_row_as_ints, dtype=np.float32))

        mutation_csv_row_tensor = torch.unsqueeze(mutation_csv_row_tensor, 0)

        # store the mutation information with a dummy label value in the "mutations" dict.
        self.mutations.append((mutation_csv_row_tensor, -1))

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    def get_num_acids_in_seq(self):
        return len(self.mutations[0][0][0])