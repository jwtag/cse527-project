# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

# add the parent directory to the Python path so we can use dataset_helper.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# other imports.
from torch.utils.data import DataLoader
import csv
import torch
from dataset_helper import DataCategory
from config import GranularConfig
from neural_network import Net
from datasets.granular_model_mutation_dataset import GranularModelMutationDataset
from datasets.granular_demo_dataset import GranularDemoDataset

def main():

    # get the sequence from the demo file
    csv_file = open(GranularConfig.demo_data_file)
    mutation_csv_reader = csv.reader(csv_file)
    mutation_csv_sequence = next(mutation_csv_reader)

    # decode the labels
    ini_label = identify_drug(mutation_csv_sequence, DataCategory.INI)
    pi_label = identify_drug(mutation_csv_sequence, DataCategory.PI)
    rti_label = identify_drug(mutation_csv_sequence, DataCategory.RTI)

    # print classified value
    print("ini " + str(ini_label))
    print("pi " + str(pi_label))
    print("rti " + str(rti_label))


def identify_drug(multi_protein_sequence, category):

    # extract the specific protein mutation sequence from the full multi-protein sequence.
    if category == DataCategory.INI:
        model_dataset = GranularModelMutationDataset(GranularConfig.ini_data_file, GranularConfig.use_binary_labels)
        mutation_seq = multi_protein_sequence[6:294]
    elif category == DataCategory.PI:
        model_dataset = GranularModelMutationDataset(GranularConfig.pi_data_file, GranularConfig.use_binary_labels)
        mutation_seq = multi_protein_sequence[294:393]
    elif category == DataCategory.RTI:
        model_dataset = GranularModelMutationDataset(GranularConfig.rti_data_file, GranularConfig.use_binary_labels)
        mutation_seq = multi_protein_sequence[393:]
    else:
        raise Exception("Invalid category: " + category)

    # get a dataset solely containing the mutation_seq + a DataLoader to feed it into PyTorch.
    mutation_seq_dataset = GranularDemoDataset(mutation_seq)
    seqLoader = DataLoader(mutation_seq_dataset, batch_size=GranularConfig.batch_size, shuffle=True, num_workers=4)

    # load models, setup net
    acid_seq_length = mutation_seq_dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(GranularConfig.device)
    net.load_state_dict(torch.load('../{}_model_best_train.pt'.format(GranularConfig.current_configuration_write_file_prefix + '_' + str(category)), map_location=GranularConfig.device))

    # get mutation_seq_tensor
    for _, data in enumerate(seqLoader, 0):
        output = net(data[0].to(GranularConfig.device))
        _, predicted_drug_tensor = torch.max(output, 1)
        predicted_drug = predicted_drug_tensor[0].item()
        return model_dataset.decode_label(predicted_drug)


if __name__ == "__main__":
    main()