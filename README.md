# So... what is the current state of this repo?

`/model-data` contains the `.csv` which stores all the sequencing data.

There are three Python files:
- `mutation-dataset.py` - This file is the PyTorch Dataset object which processes + stores the DNA mutations.
- `main.py` - This file is the PyTorch neural-network executable that builds the NN.
- `demo.py` - This file is a demo which takes in a sample sequencing of the three proteins and attempts to classify the HIV drug usage.

Run `main.py` to startup the NN-training code.

# Multilabeled data.

This classifier supports multilabeled data (each entry has a label for each drug type).

It can run where the labels are the actual drug names, or where the labels are a simple 1/0 to indicate if that type of 
drug is present.  To turn on the 1/0 approach, flip the `use_binary_labels` variable in `main.py`.

While there _are_ fancy libraries out there for handling this, I chose to do this via having three separate sub-classifiers
which aggregate their results.  [I developed this approach by following this guide.](https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7)

The labels in each drug type are suffixed with the category to ensure no cross-categorization occurs.

# Remaining TODOs:

get demo working with multilabeled vars

get multidrug combos of one type to be split-up (ex: the drug entry is ""FTC,NRTI,RTI,TDF" for one of the three types)
