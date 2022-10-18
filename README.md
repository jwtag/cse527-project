# So... what is the current state of this repo?

This repo contains some relevant code from a past ML CSE project I worked on in 2019.  The models from the project can be 
found in the following files:
- `main.py`:  A basic NN-based classifier.
- `main_resnet.py`:  A NN-based classifier which takes advantage of ResNet (a popular CNN) for additional accuracy.
- `final_model.py`:  The most-optimized implementation of our classifier.  This model classified the best for our problem (classifying images of cats & dogs).

All the models get their data from the `model-data` directory.  All the source code for the models is tailored for the 
original problem, and will need to be made generic for our project.  To keep this repo small, I removed all the testing/training
data previously used for my 2019 project.

As part of the original project, my group had to have some sort of demo.  For this, we created `demo.py`, which contains
code which intakes an image and outputs if the classifier believes that it shows a cat or a dog.  This script may be
useful to repurpose if we need to have any sort of demo for our project.