# Welcome to the CSE527 HIV Drug Mutation Classifier Project Repo!
#### (wow, that's a mouthful!)

In this repo, we have multiple neural network models which can be used to link mutations in the HIV genome to drugs
potentially used upon the virus in the past.

The data we used for this project was derived from data on this website:  https://hivdb.stanford.edu/pages/geno-rx-datasets.html

## `granular-model` vs `unified-model`
There are two main directories:  granular-model and unified-model.

The "granular" model takes in all the Stanford data and uses it to build three separate models (one for each drug type).
This model is more useful for accurately identifying drugs.

The "unified" model only takes in data for patients with data for all three drug types.  By evaluating all three data 
types at once, this model takes into account any potential cross-category drug interactions.  Not all patients have data
for all three drug types, so this model only covers a subset of the Stanford data.

## What are the files in each model?

Both model folders contain the following content:

- `/datasets` contains the custom dataset objects used to storing the data for processing
- `/datasets/model-data` contains the `.csv` which stores all the protein sequencing data for training/testing/etc.
- `main.py` - This file is the PyTorch neural-network executable that builds the NN.
- `demo.py` - This file is a demo which takes in a sample sequencing of the three proteins and attempts to classify the HIV drug usage.
- `evaluator.py` - This file evaluates + prints out stats detailing how well the model classifies the full dataset(s).
- `config.py` - This file contains values used to configure the model.  You can tweak this file to change the behavior of the model with ease.

To startup the NN-training code for either model, run the model's `main.py` file.

## Multilabeled data in `unified-model`.

The classifier in `unified-model` uses multilabeled data (each entry has a label for each drug type).

It can run where the labels are the actual drug names, or where the labels are a simple 1/0 to indicate if that type of 
drug is present.  To turn on the 1/0 approach, flip the `UnifiedConfig.use_binary_labels` variable in `config.py`.

While there _are_ fancy libraries out there for handling this, for ease of development (and sanity) I chose to do this 
via having three separate sub-classifiers which aggregate their results.  [I developed this approach by following this guide.](https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7)

The labels in each drug type are suffixed with the category to ensure no cross-categorization occurs.

## `unified-model` dataset types.

There are two types of datasets that can be used for the `unified-model`:  `UnifiedModelMutationDataset` and `ShuffledMutationDataset`.

These two dataset types affect how the model operates as follows...
- `UnifiedModelMutationDataset`
  - Reads in the mutation triplet-combinations as written in the file.
  - Since the linkages between the mutations are intact, we can take advantage of any cross-protein drug interactions in our classification process (if any exist).
- `UnifiedModelMutationDataset`
  - Splits up the mutation triplet-combinations into three separate models.
  - Shuffles the triplets so that the combinations are no longer intact.
  - Shuffles the protein sequence ordering so that any accidental cross-protein motifs are eliminated from the model.

You can switch unified-model between the dataset types in its `config.py`.

## What's the neural network architecture?

Both models share the same neural network architecture:

----[input data]---> 1D Convolutional Layer -----> ReLU function -----> Flatten function -----> Linear function -----> Output

This is an incredibly simple architecture that works fairly well.  Since we've regularly obtained >95% accuracy on the 
`unified-model`'s training and test data, we haven't felt it necessary to improve upon this architecture.

# Remaining TODOs:

obtain performance data for:
- multilabel unified model
- single unified model
- granular model
- ALL THE ABOVE IN BINARY CLASSIFICATION