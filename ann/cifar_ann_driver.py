#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 6:00 PM
# 
# -- Description --
# This file holds the driving code for artificial neural network
# designs using the CIFAR-100 dataset from cifar_ann. This file
# will look very similiar to the cnn structure because both use
# TFLearn to create networks. This file also uses functions
# defined in the cnn structure to avoid rewriting code.
#=================================================================

import os
import time
import unittest
import tensorflow as tf
import tflearn
import tflearn.datasets.cifar100 as cifar100

# use functions already defined in CNN files
import cnn.cifar_cnn_driver as cdr
import ann.cifar_ann_uts as ann_unit_tests

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

# define name and path where nets should be saved
ANN_NET_PATH = 'nets/ann/'
ANN_MODEL_NAME = 'ann_cifar100_model.tfl'

def main():
    pass

def train_cifar_artnet(model):
    """
    Train and save CIFAR-100 ANN model and return
    its training time. This function is similar to
    the CNN version of training the convnet, but
    there are some differences needed to adjust for
    training ANNs.
    """
    tf.reset_default_graph()
    model_path = ANN_NET_PATH + ANN_MODEL_NAME
    training = True
    start_time = int(time.time() * 1000)
    while training:
        current_best = 0.0
        cdr.fit_tfl_model(
            model,
            cdr.trainX,
            cdr.trainY,
            cdr.testX,
            cdr.testY,
            ANN_MODEL_NAME,
            n_epoch=cdr.EPISODE_SIZE)
        next_accuracy = cdr.test_tfl_model(
            model, 
            cdr.validX, 
            cdr.validY)
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

def train_artnet(model, n_epoch):
    tf.reset_default_graph()
    start_time = int(time.time() * 1000)
    cdr.fit_tfl_model(
        model,
        cdr.trainX,
        cdr.trainY,
        cdr.testX,
        cdr.testY,
        ANN_MODEL_NAME,
        n_epoch=n_epoch)
    finish_time = int(time.time() * 1000)
    model.save("nets/temp/ann/test_conv.tfl")
    return finish_time - start_time

def run_ann_tests():
    # run all unit tests from cifar_ann_uts module
    suite = unittest.TestLoader().loadTestsFromModule(ann_unit_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)



if __name__ == "__main__":
    main()