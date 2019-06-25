### About this project

In this project I wanted to create a image classifier that can detect shoes on images from scratch.

I downloaded lots of pictures of shoes and photography not containing shoes from google images and other sources.
After that I trained a convolutional neural network using data augmentation.
I trained the network on 4927 pictures (with roughly half being pictures of shoes) and validated the model on 542 pictures while training (also half-half).

You can test the model on your own pictures by placing them in the directory "data/test/shoe" (pictures containing shoes) and "data/test/no_shoe" and running the "Make Predictions" Notebook.

The fully trained model is the "model.h5" file.

It is not very good yet but a work in progress.