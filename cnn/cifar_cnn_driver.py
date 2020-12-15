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
import tflearn
import tflearn.datasets.cifar100 as cifar100
from cifar_cnn import *
import cifar_cnn_uts

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

# define name and path where nets should be saved
NET_PATH = 'nets/cnn/'
MODEL_NAME = 'cnn_cifar100_model.tfl'

# define how long network should train before checking accuracy
EPISODE_SIZE = 5

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


def main():
    clear_console()
    
    model = make_cifar_convnet()
    total_time = train_cifar_convnet(model)
    print(total_time)
    time_stamp = milliseconds_to_timestamp(total_time)
    print(f"Training time -> {time_stamp}")

    # run all unit tests from cifar_cnn_uts module
    suite = unittest.TestLoader().loadTestsFromModule(cifar_cnn_uts)
    unittest.TextTestRunner(verbosity=2).run(suite)

# recorded training time: 00:00:00.000
def train_cifar_convnet(model):
    """ 
    Train and save CIFAR-100 CNN model and return 
    training time. For each "epoch episode" which 
    is outlined in the fit_tfl_model call in this 
    function, continue if improvement else, stop 
    and save. Return the accumulated training time.
    """
    tf.reset_default_graph()
    model_path = NET_PATH + MODEL_NAME
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
            MODEL_NAME,
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

if __name__ == "__main__":
    main()
