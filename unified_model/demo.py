# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

# add the dataset_helper to the Python path so we can use it.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# do other imports.
import torch
from dataset_helper import DataCategory
from neural_network import Net
from config import UnifiedConfig
from unified_model.datasets.unified_model_mutation_dataset import UnifiedModelMutationDataset
from torch.utils.data import DataLoader

def main():
    # get the dataset used to generate the model.
    original_model_dataset = UnifiedConfig.model_dataset_class(UnifiedConfig.model_data_file, UnifiedConfig.use_binary_labels)

    # get a DataLoader for the sequence.
    demo_dataset = UnifiedModelMutationDataset(UnifiedConfig.demo_data_file, UnifiedConfig.use_binary_labels)
    seqLoader = DataLoader(demo_dataset, batch_size=UnifiedConfig.batch_size, shuffle=True, num_workers=4)

    # load model, setup net
    acid_seq_length = seqLoader.dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(UnifiedConfig.device)
    # if the computer has cuda, load with cuda
    net.load_state_dict(torch.load('../{}_unified_model_best_train.pt'.format(UnifiedConfig.current_configuration_write_file_prefix), map_location=UnifiedConfig.device))

    # get output
    for i, data in enumerate(seqLoader, 0):
        # run sequences thru nn
        output = net(data['mutation_seq'].to(UnifiedConfig.device))

        # get predicted vals
        # get the predicted drug for each drug type
        _, predicted_type_ini = torch.max(output[DataCategory.INI], 1)
        _, predicted_type_pi = torch.max(output[DataCategory.PI], 1)
        _, predicted_type_rti = torch.max(output[DataCategory.RTI], 1)

        # get the label for each val
        predicted_val_type_ini = predicted_type_ini[0].item()
        predicted_val_type_pi = predicted_type_pi[0].item()
        predicted_val_type_rti = predicted_type_rti[0].item()

        # decode the labels
        decoded_labels = original_model_dataset.decode_labels({
            DataCategory.INI: predicted_val_type_ini,
            DataCategory.PI: predicted_val_type_pi,
            DataCategory.RTI: predicted_val_type_rti
        }, False)

        # print classified value
        print(decoded_labels)

if __name__ == "__main__":
    main()