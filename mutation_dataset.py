# Represents a mutation dataset.
# Takes in a CSV of labelled mutations, then stores them.
import csv
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset

class MutationDataset(Dataset):
    def __init__(self, mutations_csv_files, csv_file_dir):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

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
                mutation_csv_row_label = self.label_encoder.encode_label(mutation_csv_row[1])

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

                # store the mutation information + the label in the "mutations" dict.
                self.mutations.append((mutation_csv_row_tensor, mutation_csv_row_label))

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    def decode_label(self, encoded_label):
        return self.label_encoder.decode(encoded_label)


# returns a numerical value we can use to represent the acid at the position. This method follows the below pattern:
#
# ‘-‘ = nothing is different, Ignore by returning zero
# ‘.’ = no sequence present, Ignore by returning zero.
# ‘<LETTER>#’ = insertion.  Ignore by returning zero.
# ‘~’ = deletion.  Ignore by returning zero.
# ‘<LETTER>*’ = stop codon.  If letter is present, return acid letter char value (since that indicates a change).
#               Otherwise, ignore by returning zero.
# ‘<LETTER>’ = one acid substitution, return the acid letter char value.

def get_acid_mutation_value(acid_mutation_str):
    if len(acid_mutation_str) == 0:  # handle any data formatting errors which create empty strings
        return 0

    # len > 1
    elif len(acid_mutation_str) > 1:
        # if the acid changed to a stop codon, we should return the value of the acid
        if acid_mutation_str[1] == '*':
            return ord(acid_mutation_str[0])

        # this case == insertion or stop.  We are interested in the first character (the letter) in either case.
        else:
            return 0

    # len == 1
    else:
        # we should not do anything for no-ops, no-seqs, or deletions.
        if acid_mutation_str[0] == '-' or acid_mutation_str[0] == '.' or acid_mutation_str[0] == '~':
            return 0

        # otherwise, return the char value of the acid letter.
        else:
            return ord(acid_mutation_str[0])




# class used to encode/decode labels to/from ints used by PyTorch for classification
class LabelEncoder:
    def __init__(self):
        self.label2int = {}  # used for encode
        self.int2label = {}  # used for decode

    def encode_label(self, label):
        # add the label to the dicts if necessary
        if label not in self.label2int:
            label_int = len(self.label2int)
            self.label2int[label] = label_int
            self.int2label[label_int] = label

        # return the int from the dict
        return self.label2int[label]

    def decode_label(self, label):
        return self.int2label[label]