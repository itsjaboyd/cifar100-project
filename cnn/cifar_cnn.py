#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 5:41 PM
# 
# -- Description --
# This file creates the various convolutional neural network
# designs using the CIFAR-100 dataset loaded in from TensorFlow.
# It also includes testing and fitting functions for training
# and testing the accuracy on those networks.
#=================================================================

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.cifar100 as cifar100

# define build and load convolutional network structures


# make_tfl_convnet provides a simple testing network 
def make_tfl_convnet():
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


def load_tfl_convnet(model_path):
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

def fit_tfl_model(model, trainX, trainY, testX, testY, 
    model_name, net_path, n_epoch=5, mbs=10):
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
    model.save(net_path + model_name)


def test_tfl_model(model, X, Y):
    computed_correctly = 0
    for i in range(len(X)):
        prediction = model.predict(X[i].reshape([-1, 32, 32, 3]))
        if np.argmax(prediction, axis=1)[0] == np.argmax(Y[i]):
            computed_correctly += 1
    return computed_correctly / len(X)