## About this project:

I found this data set on Kaggle, you can find it here: 
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
This script is extremely slow, I haven't cached the predictions and every time you start it
the data has to be preprocessed, the model is loaded and the predictions need to be plotted
on the training images. For now this is good enough for me, maybe in the future I will cache the 
results if I think I have found a model that is accurat enough.

### TODOS:
I want to try implement a convolutional neural network similar to the architecture described here:
https://paperswithcode.com/paper/facial-key-points-detection-using-deep

I also want to implement some data augmentation and create more training data.