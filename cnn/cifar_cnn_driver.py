#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 5:41 PM
#
# -- Description --
# This file loads the CIFAR-100 data and unit tests the various 
# convolutional neural network structures from the "cifar_cnn.py"
# file to ensure everything is set up and working as it should.
#=================================================================

import time
import tensorflow as tf
import tflearn
import tflearn.datasets.cifar100 as cifar100
from cifar_cnn import *

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

# validation split: split total training data into smaller 
# training and validation data.
VAL_SPLIT = 40000

# load cifar-100 data from tensorflow datasets and shuffle
(X, Y), (testX, testY) = cifar100.load_data(one_hot=True)
testX, testY = tflearn.data_utils.shuffle(testX, testY)

# split training into validation data and shuffle
trainX, trainY = X[0:VAL_SPLIT], Y[0:VAL_SPLIT]
validX, validY = X[VAL_SPLIT:], Y[VAL_SPLIT:]
validX, validY = tflearn.data_utils.shuffle(validX, validY)

# reshape data to be handed over to Tensor for use
trainX = trainX.reshape([-1, 32, 32, 3])
testX = testX.reshape([-1, 32, 32, 3])
validX = validX.reshape([-1, 32, 32, 3])

# define the path where nets should be saved
NET_PATH = 'nets/'
MODEL_NAME = 'testing_convnet.tfl'

def main():
    create_save_convnet()
    print("sleeping . . .")
    time.sleep(2)
    load_test_convnet()
    

def create_save_convnet():
    tf.reset_default_graph()
    model_tfl_convnet = make_tfl_convnet()
    fit_tfl_model(
        model_tfl_convnet,
        trainX, 
        trainY, 
        testX, 
        testY,
        MODEL_NAME, 
        NET_PATH,
        n_epoch=1, 
        mbs=10
    )

def load_test_convnet():
    tf.reset_default_graph()
    model_path = NET_PATH + MODEL_NAME
    model_tfl_convnet = load_tfl_convnet(model_path)
    acc = test_tfl_model(
        model_tfl_convnet,
        validX,
        validY
    )
    print(f"{MODEL_NAME} accuracy -> {acc * 100}%")


if __name__ == "__main__":
    main()
