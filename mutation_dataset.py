# Represents a mutation dataset.
# Takes in a CSV of labelled mutations, then stores them.
import csv
import torch
import numpy as np

from torch.utils.data import Dataset

class MutationDataset(Dataset):
    def __init__(self, mutation_csv_file, use_binary_labels):
        # create dataset dict + LabelEncoder
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # read in the csv
        csv_file = open(mutation_csv_file)
        mutation_csv_reader = csv.reader(csv_file)

        # first three cols = drug name.
        # last three cols = binary indicator of previous HIV drug usage.

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
            label1 = self.label_encoder.encode_label(0 if mutation_csv_row[0] is None else 1, 1)
            label2 = self.label_encoder.encode_label(0 if mutation_csv_row[1] is None else 1, 2)
            label3 = self.label_encoder.encode_label(0 if mutation_csv_row[2] is None else 1, 3)
        else:
            # get the values for the labels
            label1 = self.label_encoder.encode_label(mutation_csv_row[0], 1)
            label2 = self.label_encoder.encode_label(mutation_csv_row[1], 2)
            label3 = self.label_encoder.encode_label(mutation_csv_row[2], 3)

        return {
            'drug_type_1': label1,
            'drug_type_2': label2,
            'drug_type_3': label3
        }

    def decode_label(self, encoded_label):
        return self.label_encoder.decode_label(encoded_label)


    def get_num_acids_in_seq(self):
        return len(self.mutations[0]['mutation_seq'][0])

    def get_num_drugs_of_type(self, type_num):
        return self.label_encoder.get_num_type(type_num)


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
        self.type_count = [0, 0, 0]

    def encode_label(self, label, type_num):
        # the label should have the drug type category appended to it.  (this is in case it shows up in multiple columns)
        label_with_category = str(label) + "_" + str(type_num)

        # add the label to the dicts if necessary
        if label_with_category not in self.label2int:
            label_int = len(self.label2int)
            self.label2int[label_with_category] = label_int
            self.int2label[label_int] = label_with_category
            self.type_count[type_num - 1] += 1

            print("encoded new label")
            print(type_num)
            print(self.type_count[type_num - 1])

        # return the int from the dict
        return self.label2int[label_with_category]

    def decode_label(self, label):
        return self.int2label[label]

    # returns the number of drugs of the passed type (1, 2, or 3).
    def get_num_type(self, type_num):
        return self.type_count[type_num - 1]
