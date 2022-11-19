# So... what is the current state of this repo?

## `granular-model` vs `unified-model`
There are two main directories:  granular-model and unified-model.

The "granular" model takes in all the Stanford data and uses it to build three separate models (one for each drug type).
This model is more useful for accurately identifying drugs.

The "unified" model only takes in data for patients with data for all three drug types.  By evaluating all three data 
types at once, this model takes into account any potential cross-category drug interactions.

## What are the files in each model?

Both model folders contain the following content:

- `/datasets` contains the custom dataset objects used to storing the data for processing
- `/datasets/model-data` contains the `.csv` which stores all the protein sequencing data for training/testing/etc.
- `main.py` - This file is the PyTorch neural-network executable that builds the NN.
- `demo.py` - This file is a demo which takes in a sample sequencing of the three proteins and attempts to classify the HIV drug usage.
- `evaluator.py` - This file evaluates + prints out stats detailing how well the model classifies the full dataset(s).

To startup the NN-training code for either model, run `main.py`.

## Multilabeled data in `unified-model`.

The classifier in `unified-model` uses multilabeled data (each entry has a label for each drug type).

It can run where the labels are the actual drug names, or where the labels are a simple 1/0 to indicate if that type of 
drug is present.  To turn on the 1/0 approach, flip the `use_binary_labels` variable in `main.py`.

While there _are_ fancy libraries out there for handling this, I chose to do this via having three separate sub-classifiers
which aggregate their results.  [I developed this approach by following this guide.](https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7)

The labels in each drug type are suffixed with the category to ensure no cross-categorization occurs.

# Remaining TODOs:

get multidrug combos of one type to be split-up (ex: the drug entry is ""FTC,NRTI,RTI,TDF" for one of the three types)

Look into PyTorch positional embedding

Look at class imbalance blog from Alex

obtain performance data