# Represents a mutation dataset.
# Takes in a CSV of labelled mutations, then stores them.

import csv

from torch.utils.data import Dataset, DataLoader, ConcatDataset

class MutationDataset(Dataset):
    def __init__(self, mutations_csv_files, csv_file_dir):
        # create dataset dict
        self.mutations = []

        # read in the csv
        for filename in mutations_csv_files:

            csv_file = open(csv_file_dir + "/" + filename)
            mutation_csv_reader = csv.reader(csv_file)

            # skip header row
            next(mutation_csv_reader)

            for mutation_csv_row in mutation_csv_reader:
                # python has each CSV row be a list of strings.

                # first col = treatment, rest of cols = mutations

                # get the list w/o the label + the label
                mutation_csv_row_without_label = mutation_csv_row[1:]
                mutation_csv_row_label = mutation_csv_row[1]

                # turn row into array of ints so it can be processed by pytorch
                mutation_csv_row_as_ints = []
                for acid_mutation in mutation_csv_row_without_label:
                    mutation_csv_row_as_ints.append(get_acid_mutation_value(acid_mutation))

                # store the mutation information + the label in the "mutations" dict
                self.mutations.append((mutation_csv_row_as_ints, mutation_csv_row_label))

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]


# returns a numerical value we can use to represent the acid at the position. This method follows the below pattern:
#
# ‘-‘ = nothing is different, Ignore by returning zero
# ‘.’ = no sequence present, Ignore by returning zero.
# ‘<LETTER>#’ = insertion.  Return acid letter char value.
# ‘~’ = deletion.  Return -1.  (That way we can indicate this change to the model.)
# ‘<LETTER>*’ = stop codon.  If letter is present, return acid letter char value (since that indicates a change).
#               Otherwise, ignore by returning zero.
# ‘<LETTER>’ = one acid substitution, return the acid letter char value.

def get_acid_mutation_value(acid_mutation_str):
    if len(acid_mutation_str) == 0:  # handle any data formatting errors which create empty strings
        return 0

    # this case == insertion or stop.  We are interested in the first character (the letter) in either case.
    if len(acid_mutation_str) > 1:
        return ord(acid_mutation_str[0])

    # if there's been a deletion, return -1.
    if acid_mutation_str[0] == '~':
        return -1

    # if nothing changed or no sequence is present, return 0.
    if acid_mutation_str[0] == '-' or acid_mutation_str[0] == '.':
        return 0

    # otherwise, return the char value of the acid letter.
    return ord(acid_mutation_str[0])