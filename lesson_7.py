# notes
# nural networks are made of layers
# input layer takes in the raw data
# as many output nurons as the classes of the data we can have
# we also have hidden layers
# if every node connects to the hidden layer then it is densly connected
# connections are called weights
# wieghts are the values that get changed
# there are also biases
# biases have connections which remain at 1
# biases are constansts but programble values
# now we have some complecated math
# there are a few activation functions
# relu rectified linear unit takes any values less then zero and makes it zero
# tanh hyperbolic tangent squishes the values between -1 and 1
# sigmoid squishes the values between 0 and 1
# we use activation functions to move data into a higher demension
# A loss function calculates how far is our output form the actual output
# higher losser function output leads to more changes
# there are different loss functions
# commons one are mean squared error mean absolute error and hinge loss
# gradient descent is used to lead the function to the minimum
# back propogation updates weights and biases as move using gradient descent
# optemizer is the algorithm that does everything

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

train_images.shape
train_images[0,23,23]  # let's have a look at one pixel

train_labels[:10]  # let's have a look at the first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
