# So... what is the current state of this repo?

`/model-data` contains the `.csv` which stores all the sequencing data.

There are three Python files:
- `mutation-dataset.py` - This file is the PyTorch Dataset object which processes + stores the DNA mutations.
- `main.py` - This file is the PyTorch neural-network executable that builds the NN.
- `demo.py` - This file is a demo which takes in a sample sequencing of the three proteins and attempts to classify the HIV drug usage.

Run `main.py` to startup the NN-training code.

NOTE:  `demo.py` is not adapted (yet) to work with the DNA code.  It's still designed to be a cats-and-dogs image classifier.  This will be changed in the future.

# Remaining TODOs:

make multilabelled vars.
- each category = var
- each category can have multiple drugs.
- prefix each drug with category to make sure that there is no cross-categorization

maybe also just make things be Drug Presence?  T/F boolean.  Have drug?  Make multilabeled with binary.
