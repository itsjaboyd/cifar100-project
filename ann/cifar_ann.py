#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
# 
# -- Description --
# This file creates the various artificial neural network
# designs using the CIFAR-100 dataset loaded in from TensorFlow.
#=================================================================

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

def make_cifar_artnet():
    """
    This artificial network structure has been the best
    performing architecture for CIFAR-100 in my personal
    experimentation. I brought in ideas from some other
    resources that are outlined in README. This network
    happens to be very simply due to the simplistic nature
    of aritifical neural networks.
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        512,
        activation='relu',
        name='fc_layer_2')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        256,
        activation='relu',
        name='fc_layer_3')
    output = fully_connected(
        fc_layer_2,
        100,
        activation='softmax',
        name='output')
    network = regression(
        output,
        optimizer='sgd',
        loss='categorical_crossentropy',
        learning_rate=0.01)

    model = tflearn.DNN(network)  # tensorboard_verbose=3)
    return model

def load_cifar_artnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        512,
        activation='relu',
        name='fc_layer_2')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        256,
        activation='relu',
        name='fc_layer_3')
    output = fully_connected(
        fc_layer_2,
        100,
        activation='softmax',
        name='output')

    model = tflearn.DNN(output)
    model.load(model_path)
    return model

def make_smaller_artnet():
    """
    This artificial neural network is purely used for using
    and testing the system to make sure each component type
    works within the main driver. This network may perform
    rather horribly on the dataset due to lack of nodes!
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        256,
        activation='relu',
        name='fc_layer_1')
    output = fully_connected(
        fc_layer_1,
        100,
        activation='softmax',
        name='output')
    network = regression(
        output,
        optimizer='sgd',
        loss='categorical_crossentropy',
        learning_rate=0.1)

    model = tflearn.DNN(network)  # tensorboard_verbose=3)
    return model

def load_smaller_artnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        256,
        activation='relu',
        name='fc_layer_1')
    output = fully_connected(
        fc_layer_1,
        100,
        activation='softmax',
        name='output')
    
    model = tflearn.DNN(output)
    model.load(model_path)
    return model

def make_larger_artnet():
    """
    Just a larger ANN architecture for testing purposes
    as with the smaller artificial network architecture.
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_layer,
        1024,
        activation='relu',
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        1536,
        activation='relu',
        name='fc_layer_2')
    fc_layer_3 = fully_connected(
        fc_layer_2,
        512,
        activation='relu',
        name='fc_layer_3')
    fc_layer_4 = fully_connected(
        fc_layer_3,
        256,
        activation='relu',
        name='fc_layer_4')
    fc_layer_5 = fully_connected(
        fc_layer_4,
        128,
        activation='relu',
        name='fc_layer_5')
    output = fully_connected(
        fc_layer_5,
        100,
        activation='softmax',
        name='output')
    network = regression(
        output,
        optimizer='sgd',
        loss='categorical_crossentropy',
        learning_rate=0.1)

    model = tflearn.DNN(network)  # tensorboard_verbose=3)
    return model

def load_larger_artnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(
        input_data,
        1024,
        activation='relu',
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        1536,
        activation='relu',
        name='fc_layer_2')
    fc_layer_3 = fully_connected(
        fc_layer_2,
        512,
        activation='relu',
        name='fc_layer_3')
    fc_layer_4 = fully_connected(
        fc_layer_3,
        256,
        activation='relu',
        name='fc_layer_4')
    fc_layer_5 = fully_connected(
        fc_layer_4,
        128,
        activation='relu',
        name='fc_layer_5')
    output = fully_connected(
        fc_layer_5,
        100,
        activation='softmax',
        name='output')

    model = tflearn.DNN(output)
    model.load(model_path)
    return model
