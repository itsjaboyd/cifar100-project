#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# -- Description --
# This file loads the CIFAR-100 data and outlines other useful
# functions that both the ANN and CNN portions of this project
# will import from.
#=================================================================

import os
import time
import unittest
import tensorflow as tf
import numpy as np
import tflearn
import tflearn.datasets.cifar100 as cifar100
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

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

def fully_train_network(model, model_path, model_name, ep_size=5):
    """ 
    Train and save CIFAR-100 CNN & ANN models, return 
    training time. For each "epoch episode" which 
    is outlined in the fit_tfl_model call in this 
    function, continue if improvement else, stop 
    and save. Return the accumulated training time.
    """
    tf.reset_default_graph()
    model_path = model_path + model_name
    training = True
    start_time = int(time.time() * 1000)
    current_best = 0.0
    while training:
        fit_tfl_model(
            model,
            trainX,
            trainY,
            testX,
            testY,
            model_name,
            n_epoch=ep_size)
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

def train_network(model, model_name, model_path, n_epoch=5):
    tf.reset_default_graph()
    start_time = int(time.time() * 1000)
    fit_tfl_model(
        model,
        trainX,
        trainY,
        testX,
        testY,
        model_name,
        n_epoch=n_epoch)
    finish_time = int(time.time() * 1000)
    model.save(model_path)
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


def fit_tfl_model(model, trainX, trainY, testX, testY,
    model_name, n_epoch=5, mbs=10):
    model.fit(
        trainX,
        trainY,
        n_epoch=n_epoch,
        shuffle=True,
        validation_set=(
            testX,
            testY),
        show_metric=True,
        batch_size=mbs,
        run_id=model_name
    )

def test_tfl_model(model, X, Y):
    computed_correctly = 0
    for i in range(len(X)):
        prediction = model.predict(X[i].reshape([-1, 32, 32, 3]))
        if np.argmax(prediction, axis=1)[0] == np.argmax(Y[i]):
            computed_correctly += 1
    return computed_correctly / len(X)


def make_simple_network():
    """
    This simple TensorFlow network design is purely meant
    for using and testing the system to make sure each
    component network type works within the utility file.
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        100,
        activation='softmax',
        name='fc_layer_1')
    network = regression(
        fc_layer_1,
        optimizer='sgd',
        loss='categorical_crossentropy',
        learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_simple_network(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        100,
        activation='softmax',
        name='fc_layer_1')
    model = tflearn.DNN(fc_layer_1)
    model.load(model_path)
    return model