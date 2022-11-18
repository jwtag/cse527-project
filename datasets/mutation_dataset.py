# Represents a mutation dataset.
# Takes in a CSV of labelled mutations, then stores them.
import csv
import torch
import numpy as np

from torch.utils.data import Dataset

from datasets.dataset_helper import LabelEncoder, get_acid_mutation_value, DataCategory


class MutationDataset(Dataset):
    def __init__(self, mutation_csv_file, use_binary_labels):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # read in the csv
        csv_file = open(mutation_csv_file)
        mutation_csv_reader = csv.reader(csv_file)

        # first three cols = drug name.
        # second three cols = binary indicator of previous HIV drug usage.

        for mutation_csv_row in mutation_csv_reader:
            # python has each CSV row be a list of strings.

            # first col = treatment, rest of cols = mutations

            # get the list w/o the label data
            mutation_csv_row_without_label = mutation_csv_row[6:]

            # get the label(s) for the data.
            mutation_csv_row_labels_dict = self.get_multidrug_label_set(mutation_csv_row, use_binary_labels)

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

            # create a dict containing the tensor + the labels
            mutation_dict = {'mutation_seq': mutation_csv_row_tensor, 'labels': mutation_csv_row_labels_dict}

            # store the mutation dict in the "mutations" list.
            self.mutations.append(mutation_dict)

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    # Obtains a dict of labels for the row.  The labels specify the drug combination the patient has.
    #
    # mutation_csv_row = row being analyzed
    # use_binary_labels = if we wish for the label(s) to contain a binary 0/1 for each drug class.
    #                     if false, the labels are scoped to the drugs.
    def get_multidrug_label_set(self, mutation_csv_row, use_binary_labels):
        if use_binary_labels:
            # make the labels simple T/F (0s/1s).
            label1 = self.label_encoder.encode_label(0 if mutation_csv_row[3] == "0" else 1, 1)
            label2 = self.label_encoder.encode_label(0 if mutation_csv_row[4] == "0" else 1, 2)
            label3 = self.label_encoder.encode_label(0 if mutation_csv_row[5] == "0" else 1, 3)
        else:
            # get the values for the labels
            label1 = self.label_encoder.encode_label(mutation_csv_row[0], 1)
            label2 = self.label_encoder.encode_label(mutation_csv_row[1], 2)
            label3 = self.label_encoder.encode_label(mutation_csv_row[2], 3)

        return {
            DataCategory.INI: label1,
            DataCategory.PI: label2,
            DataCategory.RTI: label3
        }

    def decode_labels(self, encoded_labels_dict, include_category):
        return {
            DataCategory.INI: self.label_encoder.decode_label(encoded_labels_dict[DataCategory.INI], include_category),
            DataCategory.PI: self.label_encoder.decode_label(encoded_labels_dict[DataCategory.PI], include_category),
            DataCategory.RTI: self.label_encoder.decode_label(encoded_labels_dict[DataCategory.RTI], include_category)
        }

    def get_num_acids_in_seq(self):
        return len(self.mutations[0]['mutation_seq'][0])