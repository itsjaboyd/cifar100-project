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
        conv_layer_2,
        2,
        name='pool_layer_3')
    fc_layer_1 = fully_connected(
        pool_layer_2, 
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
    model = tflearn.DNN(network)
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
        conv_layer_2,
        2,
        name='pool_layer_3')
    fc_layer_1 = fully_connected(
        pool_layer_2,
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