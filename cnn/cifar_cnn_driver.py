#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# -- Description --
# This file loads the CIFAR-100 data and unit tests the various 
# convolutional neural network structures from the "cifar_cnn.py"
# file to ensure everything is set up and working as it should.
#=================================================================

import os
import time
import unittest
import tensorflow as tf
import matplotlib.pylab as plt
import tflearn
import tflearn.datasets.cifar100 as cifar100
from cnn.cifar_cnn import *
import cnn.cifar_cnn_uts as cnn_unit_tests

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

# define name and path where nets should be saved
CNN_NET_PATH = 'nets/cnn/'
CNN_MODEL_NAME = 'cnn_cifar100_model.tfl'

# define how long network should train before checking accuracy
EPISODE_SIZE = 3

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

# recorded training time with Tensorboard: 02:04:46.599
# recorded traingin time w/o Tensorboard: 00:00:00.000
def train_cifar_convnet(model):
    """ 
    Train and save CIFAR-100 CNN model and return 
    training time. For each "epoch episode" which 
    is outlined in the fit_tfl_model call in this 
    function, continue if improvement else, stop 
    and save. Return the accumulated training time.
    """
    tf.reset_default_graph()
    model_path = CNN_NET_PATH + CNN_MODEL_NAME
    training = True
    start_time = int(time.time() * 1000)
    while training:
        current_best = 0.0
        fit_tfl_model(
            model,
            trainX,
            trainY,
            testX,
            testY,
            CNN_MODEL_NAME,
            n_epoch=EPISODE_SIZE)
        next_accuracy = test_tfl_model(
            model, 
            validX, 
            validY)
        if current_best < next_accuracy:
            model.save(model_path)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Accuracy improved: {round(next_accuracy * 100, 2)}%")
            print(f"Saved network: {model_path}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            current_best = next_accuracy
        else:
            training = False
            found_best = round(current_best * 100, 2)
            print("-------------------------------------------------")
            print(f"No accuracy improvement over epoch episode.")
            print(f"Training terminated wtih accuracy: {found_best}")
            print("-------------------------------------------------")  

    finish_time = int(time.time() * 1000)
    return finish_time - start_time

def train_convnet(model, n_epoch):
    tf.reset_default_graph()
    start_time = int(time.time() * 1000)
    fit_tfl_model(
        model,
        trainX,
        trainY,
        testX,
        testY,
        CNN_MODEL_NAME,
        n_epoch=n_epoch)
    finish_time = int(time.time() * 1000)
    model.save("nets/temp/cnn/test_conv.tfl")
    return finish_time - start_time

def run_cnn_tests():
    # run all unit tests from cifar_cnn_uts module
    suite = unittest.TestLoader().loadTestsFromModule(cnn_unit_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    

def milliseconds_to_timestamp(mils):
    """
    Helper function to neatly return the elapsed time
    training took in milliseconds and convert total
    into the standard timestamp format as hh:mm:ss.mmm
    with hours, minutes, seconds, and milliseconds.
    """
    mils_left = (mils % 1000)
    millis = int(mils)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    hours = int(hours)
    # add necessary zeros two each timed section
    if len(str(mils_left)) < 2:
        mils_left = f"00{mils_left}"
    elif len(str(mils_left)) < 3:
        mils_left = f"0{mils_left}"
    if len(str(seconds)) < 2:
        seconds = f"0{seconds}"
    if len(str(minutes)) < 2:
        minutes = f"0{minutes}"
    if len(str(hours)) < 2:
        hours = f"0{hours}"
    return f"{hours}:{minutes}:{seconds}.{mils_left}"

def clear_console():
    """Handy multi-platform function that clears the console."""
    os.system('cls' if os.name == 'nt' else 'clear')