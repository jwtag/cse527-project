# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

import torch

from torch.utils.data import DataLoader

from dataset_helper import DataCategory
from neural_network import Net
from datasets.granular_model_mutation_dataset import GranularModelMutationDataset

model_data_filename = 'model-data/cse527_unified_model_data.csv'
demo_data_filename = 'model-data/demo_data.csv'
device = "cpu" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    batch_size = 128  # must match the batch size used to generate the classifier model.
    use_binary_labels = False  # if the labels should not be drug-specific, but scoped to drug-type-specific instead.
                              # (ex: <drugname>101 = <drugname><drug><no drug><drug>)
                              # NOTE:  THIS MUST MATCH THE LABELLING SYSTEM USED TO GENERATE THE MODELS!

    # get the datasets used to generate the models.
    # (this will be used to decode the labels during the classification process and should be identical to what was used to generate the pt file)

    # get a DataLoader for the sequence.
    demo_dataset = UnifiedModelMutationDataset(demo_data_filename, use_binary_labels=use_binary_labels)
    seqLoader = DataLoader(demo_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load model, setup net
    acid_seq_length = seqLoader.dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(device)
    # if the computer has cuda, load with cuda
    net.load_state_dict(torch.load('../model_best_train.pt', map_location=device))

    # get output
    for i, data in enumerate(seqLoader, 0):
        # run sequences thru nn
        output = net(data['mutation_seq'].to(device))

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