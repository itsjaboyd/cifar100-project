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
import ann.cifar_ann_uts as ann_unit_tests

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

# define name and path where nets should be saved
ANN_NET_PATH = 'nets/ann/'
ANN_MODEL_NAME = 'ann_cifar100_model.tfl'

def run_ann_uts():
    # run all unit tests from cifar_ann_uts module
    suite = unittest.TestLoader().loadTestsFromModule(ann_unit_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)