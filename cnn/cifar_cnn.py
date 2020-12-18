#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
# 
# -- Description --
# This file creates the various convolutional neural network
# designs using the CIFAR-100 dataset loaded in from TensorFlow.
# It also includes testing and fitting functions for training
# and testing the accuracy on those networks. The fit and test
# functions were taken from a previous homework assignment that
# used convolutional networks in TensorFlow.
#=================================================================

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


def make_cifar_convnet():
    """
    This network structure has been the best performing
    architecture for CIFAR-100 in my experimentation.
    I brought in ideas from some other sources that are
    outlined in the README, but this network is rather
    deep with three convolutional layers connected to
    three pool layers, and finally three fully
    connected layers to complete the network.
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer_1 = conv_2d(
        input_layer,
        nb_filter=32,
        filter_size=3,
        activation='relu',
        name='conv_layer_1')
    pool_layer_1 = max_pool_2d(
        conv_layer_1,
        2,
        name='pool_layer_1')
    conv_layer_2 = conv_2d(
        pool_layer_1,
        nb_filter=64,
        filter_size=3,
        activation='relu',
        name='conv_layer_2')
    pool_layer_2 = max_pool_2d(
        conv_layer_2,
        2,
        name='pool_layer_2')
    conv_layer_3 = conv_2d(
        pool_layer_2,
        nb_filter=128,
        filter_size=3,
        activation='relu',
        name='conv_layer_2')
    pool_layer_3 = max_pool_2d(
        conv_layer_3,
        2,
        name='pool_layer_3')
    fc_layer_1 = fully_connected(
        pool_layer_3,
        256,
        activation='relu',
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        128,
        activation='relu',
        name='fc_layer_2')
    fc_layer_3 = fully_connected(
        fc_layer_2,
        100,
        activation='softmax',
        name='fc_layer_3')
    network = regression(
        fc_layer_3, 
        optimizer='sgd',
        loss='categorical_crossentropy', 
        learning_rate=0.01)

    model = tflearn.DNN(network) # tensorboard_verbose=3)
    return model

def load_cifar_convnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer_1 = conv_2d(
        input_layer,
        nb_filter=32,
        filter_size=3,
        activation='relu',
        name='conv_layer_1')
    pool_layer_1 = max_pool_2d(
        conv_layer_1,
        2,
        name='pool_layer_1')
    conv_layer_2 = conv_2d(
        pool_layer_1,
        nb_filter=64,
        filter_size=3,
        activation='relu',
        name='conv_layer_2')
    pool_layer_2 = max_pool_2d(
        conv_layer_2,
        2,
        name='pool_layer_2')
    conv_layer_3 = conv_2d(
        pool_layer_2,
        nb_filter=128,
        filter_size=3,
        activation='relu',
        name='conv_layer_2')
    pool_layer_3 = max_pool_2d(
        conv_layer_3,
        2,
        name='pool_layer_3')
    fc_layer_1 = fully_connected(
        pool_layer_3,
        256,
        activation='relu',
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1,
        128,
        activation='relu',
        name='fc_layer_2')
    fc_layer_3 = fully_connected(
        fc_layer_2,
        100,
        activation='softmax',
        name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

def make_shallower_convnet():
    """
    This convolutaional network design is purely meant
    for using and testing the system to make sure each
    component network type works within the main driver.
    Obviously this network will perform rather horribly
    on the dataset due to how shallow its design is.
    """
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer_1 = conv_2d(
        input_layer,
        nb_filter=20,
        filter_size=5,
        activation='relu',
        name='conv_layer_1')
    pool_layer_1 = max_pool_2d(
        conv_layer_1,
        2,
        name='pool_layer_1')
    fc_layer_1 = fully_connected(
        pool_layer_1,
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

def load_shallower_convnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer_1 = conv_2d(
        input_layer,
        nb_filter=20,
        filter_size=5,
        activation='relu',
        name='conv_layer_1')
    pool_layer_1 = max_pool_2d(
        conv_layer_1,
        2,
        name='pool_layer_1')
    fc_layer_1 = fully_connected(
        pool_layer_1,
        100,
        activation='softmax',
        name='fc_layer_1')
    model = tflearn.DNN(fc_layer_1)
    model.load(model_path)
    return model

def make_example_convnet():
    """
    This convolutional network design came straight
    from homework five and slide 22 from the lecture
    notes in the convolutional section. Again this
    architecture was used for testing purposes.
    """
    input_layer = input_data(shape=[None, 32, 32, 3]) 
    conv_layer = conv_2d(
        input_layer, 
        nb_filter=20, 
        filter_size=5, 
        activation='sigmoid', 
        name='conv_layer_1')
    pool_layer = max_pool_2d(
        conv_layer,
         2, 
         name='pool_layer_1') 
    fc_layer_1 = fully_connected(
        pool_layer, 
        200, 
        activation='sigmoid', 
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1, 
        100, 
        activation='softmax', 
        name='fc_layer_2')
    network = regression(
        fc_layer_2, 
        optimizer='sgd', 
        loss='categorical_crossentropy', 
        learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_example_convnet(model_path):
    input_layer = input_data(shape=[None, 32, 32, 3])
    conv_layer = conv_2d(
        input_layer, 
        nb_filter=20, 
        filter_size=5, 
        activation='sigmoid', 
        name='conv_layer_1')
    pool_layer = max_pool_2d(
        conv_layer,
         2, 
         name='pool_layer_1') 
    fc_layer_1 = fully_connected(
        pool_layer, 
        200, 
        activation='sigmoid', 
        name='fc_layer_1')
    fc_layer_2 = fully_connected(
        fc_layer_1, 
        100, 
        activation='softmax', 
        name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model