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
from cnn.cifar_cnn import *
from utils import *
import cnn.cifar_cnn_uts as cnn_unit_tests

# define name and path where nets should be saved
CNN_NET_PATH = 'nets/cnn/'
CNN_MODEL_NAME = 'cnn_cifar100_model.tfl'

# define how long network should train before checking accuracy
EPISODE_SIZE = 5

# recorded training time with Tensorboard: 04:19:31.953
def run_cnn_uts():
    # run all unit tests from cifar_cnn_uts module
    suite = unittest.TestLoader().loadTestsFromModule(cnn_unit_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)