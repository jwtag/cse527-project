# this file contains various values + methods + classes shared across the dataset classes
from enum import IntEnum


# This Enum should be used whenever labelling/referring to drug categories.
class DataCategory(IntEnum):
    INI = 0
    PI = 1
    RTI = 2
    NO_CATEGORY = 3

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


# returns if the provided label is "valid".
#
# a computed label is valid if it contains _any_ drugs which are in the drug combination specified in the other label.
#
# This method is not meant to be used for model generation:  it's meant to be used by the evaluator.py files.
def is_label_valid(expected_label, computed_label):
    # split-up the computed_label into a list of drugs
    computed_label_drugs = computed_label.split(',')

    # split-up the expected_label into a list of drugs
    expected_label_drugs = expected_label.split(',')

    return len(list(set(computed_label_drugs) & set(expected_label_drugs))) != 0

# class used to encode/decode labels to/from ints used by PyTorch for classification
class LabelEncoder:
    def __init__(self):
        self.label2int = {}  # used for encode
        self.int2label = {}  # used for decode

    def encode_label(self, label, category_enum):
        # convert the category enum to an int.
        category_int = int(category_enum)

        # the label should have the drug type category appended to it. (this is in case it shows up in multiple columns)
        label_with_category = str(label) + "_" + str(category_int)

        # add the label to the dicts if necessary
        if label_with_category not in self.label2int:
            label_int = len(self.label2int)
            self.label2int[label_with_category] = label_int
            self.int2label[label_int] = label_with_category

        # return the int from the dict
        return self.label2int[label_with_category]

    def decode_label(self, label, include_category):
        decoded_label = self.int2label[label]

        # if "include_category" is false, remove the drug category suffix from the label.
        if not include_category:
            decoded_label = decoded_label[:-2]

        return decoded_label