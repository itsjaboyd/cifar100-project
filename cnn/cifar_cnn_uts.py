





import json
import random
import sys
import numpy as np
import unittest
import tensorflow as tf
import tflearn
#from tflearn.layers.core import input_data, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.estimator import regression
import tflearn.datasets.cifar100 as cifar
from cifar_cnn import *

### Let's load MNIST and reshape train, test, and validation sets.
X, Y, testX, testY = mnist.load_data(one_hot=True)
testX, testY = tflearn.data_utils.shuffle(testX, testY)
trainX, trainY = X[0:50000], Y[0:50000]
validX, validY = X[50000:], Y[50000:]
validX, validY = tflearn.data_utils.shuffle(validX, validY)
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
validX = validX.reshape([-1, 28, 28, 1])

## change this directory accordingly.
NET_PATH = 'nets/'