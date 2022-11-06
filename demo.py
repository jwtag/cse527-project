# This script can be used to demonstrate the classifier by taking in input data + outputting a classification in the CLI.

import torch

from torch.utils.data import DataLoader
from mutation_dataset import MutationDataset
from neural_network import Net

model_data_filename = './model-data/cse527_proj_data.csv'
demo_data_filename = './model-data/demo_data.csv'
device = "cpu" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    batch_size = 128  # must match the batch size used to generate the classifier model.
    use_binary_labels = True  # if the labels should not be drug-specific, but scoped to drug-type-specific instead.
                              # (ex: <drugname>101 = <drugname><drug><no drug><drug>)
                              # NOTE:  THIS MUST MATCH THE LABELLING SYSTEM USED TO GENERATE THE MODELS!

    # get the dataset used to generate the model.
    # (this will be used to decode the labels during the classification process)
    original_model_dataset = MutationDataset(model_data_filename, use_binary_labels)

    # get a DataLoader for the sequence.
    demo_dataset = MutationDataset(demo_data_filename, use_binary_labels=use_binary_labels)
    seqLoader = DataLoader(demo_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load model, setup net
    acid_seq_length = seqLoader.dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(device)
    # if the computer has cuda, load with cuda
    net.load_state_dict(torch.load('./model_best_train.pt', map_location=device))

    # get output
    for i, data in enumerate(seqLoader, 0):
        # run sequences thru nn
        output = net(data[0].to(device))

        # get predicted val
        _, predicted_tensor = torch.max(output, 1)
        print(predicted_tensor)
        predicted_val = predicted_tensor[0].item()

        # print classified value
        print(predicted_tensor[2])
        print(original_model_dataset.decode_label(predicted_val))

if __name__ == "__main__":
    main()