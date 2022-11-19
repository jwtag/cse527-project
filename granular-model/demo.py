# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

# add the parent directory to the Python path so we can use dataset_helper.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from torch.utils.data import DataLoader

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import csv
import torch

from dataset_helper import DataCategory
from neural_network import Net
from datasets.granular_model_mutation_dataset import GranularModelMutationDataset
from datasets.granular_demo_dataset import GranularDemoDataset

ini_filename = './datasets/model-data/IN.csv'
pi_filename = './datasets/model-data/PR.csv'
rti_filename = './datasets/model-data/RT.csv'
demo_data_filename = './datasets/model-data/demo_data.csv'
device = "cpu" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    batch_size = 128  # must match the batch size used to generate the classifier model.
    use_binary_labels = True  # if the labels should not be drug-specific, but scoped to drug-type-specific instead.
                              # (ex: <drugname>101 = <drugname><drug><no drug><drug>)
                              # NOTE:  THIS MUST MATCH THE LABELLING SYSTEM USED TO GENERATE THE MODELS!

    # get the sequence from the demo file
    csv_file = open(demo_data_filename)
    mutation_csv_reader = csv.reader(csv_file)
    mutation_csv_sequence = next(mutation_csv_reader)

    # decode the labels
    ini_label = identify_drug(mutation_csv_sequence, DataCategory.INI, batch_size, use_binary_labels)
    pi_label = identify_drug(mutation_csv_sequence, DataCategory.PI, batch_size, use_binary_labels)
    rti_label = identify_drug(mutation_csv_sequence, DataCategory.RTI, batch_size, use_binary_labels)

    # print classified value
    print("ini " + str(ini_label))
    print("pi " + str(pi_label))
    print("rti " + str(rti_label))


def identify_drug(multi_protein_sequence, category, batch_size, use_binary_labels):

    # extract the specific protein mutation sequence from the full multi-protein sequence.
    if category == DataCategory.INI:
        model_dataset = GranularModelMutationDataset(ini_filename, use_binary_labels)
        mutation_seq = multi_protein_sequence[6:294]
    elif category == DataCategory.PI:
        model_dataset = GranularModelMutationDataset(pi_filename, use_binary_labels)
        mutation_seq = multi_protein_sequence[294:393]
    elif category == DataCategory.RTI:
        model_dataset = GranularModelMutationDataset(rti_filename, use_binary_labels)
        mutation_seq = multi_protein_sequence[393:]
    else:
        raise Exception("Invalid category: " + category)

    # get a dataset solely containing the mutation_seq + a DataLoader to feed it into PyTorch.
    mutation_seq_dataset = GranularDemoDataset(mutation_seq)
    seqLoader = DataLoader(mutation_seq_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load models, setup net
    acid_seq_length = mutation_seq_dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(device)
    net.load_state_dict(torch.load('../{}_model_best_train.pt'.format(str(category)), map_location=device))

    # get mutation_seq_tensor
    for _, data in enumerate(seqLoader, 0):
        output = net(data[0].to(device))
        _, predicted_drug_tensor = torch.max(output, 1)
        predicted_drug = predicted_drug_tensor[0].item()
        return model_dataset.decode_label(predicted_drug)


if __name__ == "__main__":
    main()