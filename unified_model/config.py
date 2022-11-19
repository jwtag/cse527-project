# This file stores the configurations for the Unified model.
# By storing this data here, we don't have to modify multiple files whenever we want to change a config value.
#
# NOTE:  This file references Datasets, so it should *never* be used inside a Dataset class.  If it is used in a Dataset
# class, a circular dependency issue will occur and the interpreter will fail to run the program.
import torch
from unified_model.datasets.unified_model_mutation_dataset import UnifiedModelMutationDataset
from unified_model.datasets.shuffled_mutation_dataset import ShuffledMutationDataset

class UnifiedConfig:
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

    # valid options are:
    # - UnifiedModelMutationDataset (keeps drug combination triplets intact)
    # - ShuffledMutationDataset (shuffles the drug combination triplets)
    model_dataset_class = ShuffledMutationDataset

    # eval configs
    num_results_to_print_per_dict = 10  # number of results to print per dict at end of evaluation.py.

    # output file configs (adjust this to write to different files for different test cases)
    current_configuration_write_file_prefix = "shuffled_not_binary"

    # input filepaths (as referenced when inside the `unified_model` directory)
    model_data_file = './datasets/model-data/cse527_unified_model_data.csv'
    demo_data_file = './datasets/model-data/demo_data.csv'

    # specifies the UnifiedConfig.device which should be used by PyTorch for computation
    # cuda = NVIDIA CUDA
    # cpu = plain ol' CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"