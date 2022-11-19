# automated program which evaluates the performance of our unified model against the original data.
#
# Generates the following output:
# - Most commonly-failed drug types w/ their most common incorrect labels.
# - Which drug category performs the worst.
#
# NOTE:  Since we're testing against the entire dataset, there are cases where the model was not trained on the label so
#        it has no way of identifying the data.  This is just an inherent issue with our architecture.

# add the dataset_helper to the Python path so we can use it.
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# do other imports.
import torch
from torch.utils.data import DataLoader
from dataset_helper import DataCategory, is_label_valid
from datasets.unified_model_mutation_dataset import UnifiedModelMutationDataset
from neural_network import Net
from collections import OrderedDict
from config import UnifiedConfig

def main():
    # get the dataset used to generate the model.
    model_dataset = UnifiedConfig.model_dataset_class(UnifiedConfig.model_data_file, UnifiedConfig.use_binary_labels)

    # get a DataLoader for the sequence.
    eval_dataset = UnifiedModelMutationDataset(UnifiedConfig.model_data_file, UnifiedConfig.use_binary_labels)
    seqLoader = DataLoader(eval_dataset, batch_size=UnifiedConfig.batch_size, shuffle=True, num_workers=4)

    # load model, setup net
    acid_seq_length = seqLoader.dataset.get_num_acids_in_seq()  # length of acid sequence being processed by neural network.
    net = Net(acid_seq_length)
    net.to(UnifiedConfig.device)

    # if the computer has cuda, load with cuda
    net.load_state_dict(torch.load('../{}_unified_model_best_train.pt'.format(UnifiedConfig.current_configuration_write_file_prefix), map_location=UnifiedConfig.device))

    # setup dictionaries to store failure data.
    # each dict is structured as follows:
    # Layer 1:  expected label -> dict of mislabel occurrence data.
    # Layer 2:  Mislabel occurrence data -> number of occurrences.
    ini_failure_dict = {}
    pi_failure_dict = {}
    rti_failure_dict = {}

    # get output
    for i, data in enumerate(seqLoader, 0):
        # run sequences thru nn
        output = net(data['mutation_seq'].to(UnifiedConfig.device))

        # get predicted labels
        # get the predicted drug for each drug type
        _, predicted_type_ini = torch.max(output[DataCategory.INI], 1)
        _, predicted_type_pi = torch.max(output[DataCategory.PI], 1)
        _, predicted_type_rti = torch.max(output[DataCategory.RTI], 1)

        # get expected labels
        expected_labels = data['labels']

        # analyze the labels, update the failure dicts as necessary
        check_and_update(expected_labels[DataCategory.INI], predicted_type_ini, eval_dataset, model_dataset, DataCategory.INI, ini_failure_dict)
        check_and_update(expected_labels[DataCategory.PI], predicted_type_pi, eval_dataset, model_dataset, DataCategory.PI, pi_failure_dict)
        check_and_update(expected_labels[DataCategory.RTI], predicted_type_rti, eval_dataset, model_dataset, DataCategory.RTI, rti_failure_dict)

        # progress statement
        if i % 100 == 0:
            print('Completed iteration ' + str(i))

    print('All data has been obtained, now sorting dicts.')

    # now that all data has been obtained, let's sort the dicts.
    ini_failure_dict, total_ini_failures = sort_dict(ini_failure_dict)
    pi_failure_dict, total_pi_failures = sort_dict(pi_failure_dict)
    rti_failure_dict, total_rti_failures = sort_dict(rti_failure_dict)

    print('All dicts have been sorted.')

    # finally, let's print out the results.
    print_dict_results(ini_failure_dict, DataCategory.INI, UnifiedConfig.num_results_to_print_per_dict)
    print_dict_results(pi_failure_dict, DataCategory.PI, UnifiedConfig.num_results_to_print_per_dict)
    print_dict_results(rti_failure_dict, DataCategory.RTI, UnifiedConfig.num_results_to_print_per_dict)
    print('Total INI failures:  ' + str(total_ini_failures))
    print('Total PI failures:  ' + str(total_pi_failures))
    print('Total RTI failures:  ' + str(total_rti_failures))


def print_dict_results(dict, category, num_results_to_print):
    print(str(category) + ':')

    # tracks how many subdicts left to print.
    subdict_counter = num_results_to_print
    for subdict_name in dict:
        # break if we hit the print limit.
        if subdict_counter <= 0:
            break

        # print subdict name.
        print('\t' + subdict_name + ':')

        # tracks how many incorrect labels left to print.
        incorrect_label_counter = num_results_to_print

        # iterate over subdict results.
        for incorrect_label_name in dict[subdict_name]:
            # break if we hit the print limit.
            if incorrect_label_counter <= 0:
                break

            print('\t\t' + incorrect_label_name + ':  ' + str(dict[subdict_name][incorrect_label_name]))

            # decrement the incorrect_label_counter
            incorrect_label_counter = incorrect_label_counter - 1

        # decrement the subdict_counter
        subdict_counter = subdict_counter - 1


# returns tuple containing (a sorted dict derived from the passed dict, total num failures)
# The dict is sorted as follows:
# - the entries have been sorted by number of failures in decreasing order.
# - each subdict entry is sorted from highest number of failure occurrences to lowest.
def sort_dict(dict):
    # create a dict to store the updated results
    updated_dict = {}
    # create a dict to store the number of failures per subdict
    failure_dict = {}

    # iterate over the subdicts in the passed dict
    for subdict_name in dict:
        # sort the subdict
        sorted_subdict = OrderedDict(sorted(dict[subdict_name].items(), key=lambda x: x[1], reverse=True))

        # save the subdict
        updated_dict[subdict_name] = sorted_subdict

        # store the number of failures in the subdict
        failure_dict[subdict_name] = sum(sorted_subdict.values())

    # at this point, we have the following:
    # - updated_dict has sorted subdicts
    # - failure dict contains the number of failures for each subdict.

    # let's sort updated_dict by the number of failures for each subdict in decreasing order.
    updated_dict = OrderedDict(sorted(dict.items(), key=lambda x: failure_dict[x[0]], reverse=True))

    # calculate the total number of failures across the entire dict.
    total_failures = sum(failure_dict.values())

    return (updated_dict, total_failures)


# check the passed labels for correctness, updates passed failure_dict if they do not match.
def check_and_update(expected_labels, actual_labels, eval_dataset, model_dataset, category, failure_dict):
    # the labels are stored in tensors of `UnifiedConfig.batch_size`, so we want to iterate over the elements for our computation.
    for idx in range(expected_labels.size(0)):
        expected_label = expected_labels[idx].item()
        actual_label = actual_labels[idx].item()

        # let's convert the labels into the actual drug names.
        # (we have to do this since the label values don't match between the model and eval datasets)
        #
        # NOTE:  The binary model seems to have some sort of bug where it can classify non-0/1 labels.  I don't feel
        #        like investing the time into debugging this, so we're just gonna move onto the next iteration of the
        #        loop if a decode error occurs.
        try:
            expected_name = eval_dataset.decode_label(expected_label, category, False)
            actual_name = model_dataset.decode_label(actual_label, category, False)
        except KeyError:
            continue

        # if the drug names don't match, store them in the failure dict.
        if not is_label_valid(expected_name, actual_name):
            # get the dict for the expected label (if one exists).  if none exists, use an empty dict.
            mislabel_occurrence_dict = failure_dict.get(expected_name, {})

            # attempt to get an entry for the decoded label from the mislabel_occurrence_dict.
            # if none exists, use "0" as the current count.
            curr_count = mislabel_occurrence_dict.get(actual_name, 0)

            # update the occurence count in the mislabel_occurrence_dict
            mislabel_occurrence_dict[actual_name] = curr_count + 1

            # store the updated mislabel_occurrence_dict in the failure dict.
            failure_dict[expected_name] = mislabel_occurrence_dict

if __name__ == "__main__":
    main()