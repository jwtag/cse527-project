# Represents a dataset containing mutation data for one of the drug types.
import csv
import torch
import numpy as np

from torch.utils.data import Dataset

from dataset_helper import LabelEncoder, get_acid_mutation_value, DataCategory


class GranularModelMutationDataset(Dataset):
    def __init__(self, mutation_csv_file, use_binary_labels):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # read in the csv
        csv_file = open(mutation_csv_file)
        mutation_csv_reader = csv.reader(csv_file)

        for mutation_csv_row in mutation_csv_reader:
            # python has each CSV row be a list of strings.

            # first col = treatment, rest of cols = mutations

            # get the list w/o the label data
            mutation_csv_row_without_label = mutation_csv_row[1:]

            # get the label for the data.
            mutation_label = self.get_label(mutation_csv_row, use_binary_labels)

            # turn row into array of ints so it can be processed by pytorch
            mutation_csv_row_as_ints = []
            for acid_mutation in mutation_csv_row_without_label:
                mutation_csv_row_as_ints.append(get_acid_mutation_value(acid_mutation))

            # reformat the python data list as a tensor.
            # This code line has some quirks, here's some explanations about what's going on:
            # - we're converting the ints to float32s since the macOS ARM/"Apple Silicon" GPU-based PyTorch code
            #   doesn't support float64 (int) processing.  float32 works across all platforms, so this is safe for
            #   other machines as well.
            # - "from_numpy()" creates a PyTorch tensor from the Numpy array.
            mutation_csv_row_tensor = torch.from_numpy(np.asarray(mutation_csv_row_as_ints, dtype=np.float32))

            mutation_csv_row_tensor = torch.unsqueeze(mutation_csv_row_tensor, 0)

            # store the mutation information with the encoded label in the "mutations" dict.
            self.mutations.append((mutation_csv_row_tensor, mutation_label))

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    # Obtains a drug label for the row.  The label specifies the drug the patient took.
    #
    # mutation_csv_row = row being analyzed
    # use_binary_labels = if we wish for the label(s) to contain a binary 0/1 simply representing if a drug of that type
    #                     was used.
    #                     if false, the labels are scoped to the drugs.
    def get_label(self, mutation_csv_row, use_binary_labels):
        if (use_binary_labels):
            return self.label_encoder.encode_label(0 if mutation_csv_row[0] == "None" else 1, DataCategory.NO_CATEGORY)
        else:
            # return a set containing a label solely for the combo
            return self.label_encoder.encode_label(mutation_csv_row[0], DataCategory.NO_CATEGORY)

    def decode_label(self, label):
        return self.label_encoder.decode_label(label, False)

    def get_num_acids_in_seq(self):
        return len(self.mutations[0][0][0])