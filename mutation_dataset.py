# Represents a mutation dataset.
# Takes in a CSV of labelled mutations, then stores them.

import csv

class MutationDataset(Dataset):
    def __init__(self, mutations_csv_file, csv_file_dir):
        # store params
        self.mutations_csv_file = mutations_csv_file  # name of file containing mutations
        self.csv_file_dir = csv_file_dir  # directory containing csv file

        # create dataset dict
        self.mutations = []

        # read in the csv
        mutation_csv_reader = csv.reader(mutations_csv_file)
        for mutation_csv_row in mutation_csv_reader:
            # python has each CSV row be a list of strings.

            # if the last column is the label (the product), we can use that information to map the label
            # to the data.  this is only if we are training.

            # get the list w/o the label + the label
            mutation_csv_row_without_label = mutation_csv_row[:-1]
            mutation_csv_row_label = mutation_csv_row[-1]

            # store the mutation information + the label in the "mutations" dict
            self.mutations.append((mutation_csv_row_without_label, mutation_csv_row_label))

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]