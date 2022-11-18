# Represents a dataset for a specific category.
# Like `mutation_dataset`, this class takes in the combined overlapping dataset CSV file (where each patient has entries
# for all categories).
#
# This dataset then stores the mutations for a specific category in that CSV file.
import csv
import torch
import numpy as np
import random


from torch.utils.data import Dataset

from datasets.dataset_helper import DataCategory, LabelEncoder, get_acid_mutation_value


class CategoryDataset(Dataset):
    def __init__(self, mutation_csv_file, use_binary_labels, category):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # read in the csv
        csv_file = open(mutation_csv_file)
        mutation_csv_reader = csv.reader(csv_file)

        # row-by-row, obtain + store mutation data.
        for mutation_csv_row in mutation_csv_reader:
            # get a list containing the mutation data for the protein specified by category
            if category == DataCategory.INI:
                mutation_csv_row_without_label = mutation_csv_row[6:294]
            elif category == DataCategory.PI:
                mutation_csv_row_without_label = mutation_csv_row[294:393]
            elif category == DataCategory.RTI:
                mutation_csv_row_without_label = mutation_csv_row[393:]
            else:
                raise Exception("Invalid category: " + category)

            # get the label for the mutation.
            label = self.get_label(mutation_csv_row, use_binary_labels, category)

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

            # add a dimension to represent the number of rows (which there is only 1)
            mutation_csv_row_tensor = torch.unsqueeze(mutation_csv_row_tensor, 0)

            # store the label + mutation in the category sublist in "mutations".
            self.mutations.append((mutation_csv_row_tensor, label))

        # shuffle the order of the dataset to ensure that the ordering is nondeterministic.
        random.shuffle(self.mutations)

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    # Obtains a label for the row.  The labels specify the drug the patient took from that category.  The LabelEncoder
    # scopes the label to the category.
    #
    # mutation_csv_row = row being analyzed
    # use_binary_labels = if we wish for the label(s) to contain a binary 0/1 simply representing if a drug was used.
    #                     if false, the labels are scoped to the drugs.
    # category = the drug category for which we're pulling the label.  use the category constants defined in this class.
    def get_label(self, mutation_csv_row, use_binary_labels, category):
        if use_binary_labels:
            # make the labels simple T/F (0s/1s).
            return self.label_encoder.encode_label(0 if mutation_csv_row[3 + category] == "0" else 1, category)
        else:
            # get the values for the labels
            return self.label_encoder.encode_label(mutation_csv_row[category], category)

    def decode_label(self, encoded_label, include_category):
        return self.label_encoder.decode_label(encoded_label, include_category)