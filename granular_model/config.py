# This file stores the configurations for the models.
# By storing this data here, we don't have to modify multiple files whenever we want to change a config value.

import torch


class GranularConfig:
    # model configs
    batch_size = 128
    momentum = 0.9
    learning_rate = 0.00005
    training_data_proportion = 0.8
    testing_data_proportion = 0.2
    use_binary_labels = False  # if the labels should not be drug-specific, but scoped to drug-type-specific instead.
    # (ex: <drugname>101 = <drugname><drug><no drug><drug>)
    num_training_epochs = 70

    # neural network configs
    kernel_size = 5  # size of the kernel to look with at the data.  kernel is used to look at multiple datapoints at once.  arbitrarily choosing 5, may change in the future.
    num_filters = 32  # number of out_channels from conv1 layer.  This number is chosen based off of intuition.
    final_num_out_channels = 1000  # number of out_channels for final layer.  This number is chosen based off of intuition and must be larger than the acid seq len or PyTorch will break.

    # eval configs
    num_results_to_print_per_dict = 10  # number of results to print per dict at end of evaluation.py.

    # output file configs (adjust this to write to different files for different test cases)
    # NOTE:  MAKE SURE TO ALSO SWITCH use_binary_labels IF CHANGING BINARY/NOT_BINARY!!!
    current_configuration_write_file_prefix = "not_binary"

    # input filepaths (as referenced when inside the `unified_model` directory)
    ini_data_file = './datasets/model-data/IN.csv'
    pi_data_file = './datasets/model-data/PR.csv'
    rti_data_file = './datasets/model-data/RT.csv'
    demo_data_file = './datasets/model-data/demo_data.csv'

    # specifies the device which should be used by PyTorch for computation
    # cuda = NVIDIA CUDA
    # cpu = plain ol' CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
