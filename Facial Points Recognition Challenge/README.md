## About this project:

This is a dataset from Kaggle, in which the objective is to identify facial points on images.

Here is the challenge and the dataset: 
https://www.kaggle.com/c/facial-keypoints-detection/overview

The code is mainly inspired from a Kernel that can be found here:
https://www.kaggle.com/madhawav/basic-fully-connected-nn

### File Descriptions:

#### general_helper_functions.py
Some helper functions I wrote to help me out in data science projects.
I just started this so by the time you see this there will probably not be a lot in here.

#### project_helper_functions.py
Helper functions specific for this project.

#### Train.py
Run this script to train the model.

#### show_training_data_predictions.py
After training you can run this script from the command line and give it an index to see
what the model predicted on one of the images of the training set.
Right now it is set to show the predictions of the first model I trained.

#### save_predictions.py
A script that saves the predictions of a model in a directory as .npy file for further use.

### TODOS:
I want to try implement a convolutional neural network similar to the architecture described here:
https://paperswithcode.com/paper/facial-key-points-detection-using-deep

I also want to implement some data augmentation and create more training data.
