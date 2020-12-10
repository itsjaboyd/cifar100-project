#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 5:41 PM
# 
# -- Description --
# This file creates the various convolutional neural network
# designs using the CIFAR-100 dataset loaded in from TensorFlow.
#=================================================================

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.cifar100 as cifar100

### Silence annoying tensorflow logging statements
tf.logging.set_verbosity(tf.logging.ERROR)

(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

print("X_train.shape =", X_train.shape, "Y_train.shape =", Y_train.shape)
print("X_test.shape =", X_test.shape, "Y_test.shape =", Y_test.shape)
