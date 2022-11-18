# Represents a mutation dataset derived from the matched dataset.  However, this dataset shuffles the mutation triplets
# instead of keeping them as one solid row.
# The dataset then stores the shuffled protein mutation triplet combinations for later iteration.
#
# This class is able to be substituted for the original mutation_dataset class w/o any changes being made to the
# surrounding code.
import torch
import random

from torch.utils.data import Dataset

from datasets.unified_model_category_dataset import DataCategory, UnifiedModelCategoryDataset
from dataset_helper import LabelEncoder


# Like `mutation_dataset`, this class takes in the combined overlapping dataset CSV file (where each patient has entries
# for all categories).
# This class then shuffles the protein sequences so that the entries are no longer interconnected.
class ShuffledMutationDataset(Dataset):

    # shuffle_ordering = should we shuffle the ordering of the proteins + labels in the string used for modelling.  that
    # way INI/PI/RTI ordering isn't consistent and accidentally creating motifs identified by the network.
    def __init__(self, mutation_csv_file, use_binary_labels, shuffle_ordering):
        self.mutations = []
        self.label_encoder = LabelEncoder()

        # Get datasets for each category.
        self.ini_dataset = UnifiedModelCategoryDataset(mutation_csv_file, use_binary_labels, DataCategory.INI)
        self.pi_dataset = UnifiedModelCategoryDataset(mutation_csv_file, use_binary_labels, DataCategory.PI)
        self.rti_dataset = UnifiedModelCategoryDataset(mutation_csv_file, use_binary_labels, DataCategory.RTI)

        # Get the number of mutation combos in the resultant dataset.
        num_mutation_combos = len(self.ini_dataset)

        # Randomly take the data from the datasets + append them a one csv row tensor.
        # Store the appended data in the "mutations" dataset for __getitem__ to use.
        for idx in range(0, num_mutation_combos):
            # We construct the csv row tensor by appending the sequences.  If shuffle_ordering == true, the tensor will
            # shuffle how the data is appended for each row.  Otherwise, it will be appended as INI+PI+RTI.
            mutation_categories = [self.ini_dataset[idx][0], self.pi_dataset[idx][0], self.rti_dataset[idx][0]]
            if shuffle_ordering:
                random.shuffle(mutation_categories)
            # now that the tensors have potentially had their ordering shuffled, concatenate them.
            mutation_set_tensor = torch.cat((mutation_categories[0],
                                                 mutation_categories[1],
                                                 mutation_categories[2]), 1)

            # construct the label dict.
            # (the labels in each dataset are already pre-scoped to the category)
            ini_label = self.ini_dataset[idx][1]
            pi_label = self.pi_dataset[idx][1]
            rti_label = self.rti_dataset[idx][1]
            mutation_set_labels_dict = {
                DataCategory.INI: ini_label,
                DataCategory.PI: pi_label,
                DataCategory.RTI: rti_label
            }

            # create a dict containing the tensor + the labels
            mutation_dict = {'mutation_seq': mutation_set_tensor, 'labels': mutation_set_labels_dict}

            # store the mutation dict in the "mutations" list.
            self.mutations.append(mutation_dict)

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, idx):
        return self.mutations[idx]

    def decode_labels(self, encoded_labels_dict, include_category):
        return {
            DataCategory.INI: self.ini_dataset.decode_label(encoded_labels_dict[DataCategory.INI], include_category),
            DataCategory.PI: self.pi_dataset.decode_label(encoded_labels_dict[DataCategory.PI], include_category),
            DataCategory.RTI: self.rti_dataset.decode_label(encoded_labels_dict[DataCategory.RTI], include_category)
        }

    def get_num_acids_in_seq(self):
        return len(self.mutations[0]['mutation_seq'][0])