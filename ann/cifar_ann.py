#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 6:00 PM
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
    pass

def load_cifar_artnet():
    pass

def make_smaller_artnet():
    """
    This artificial neural network is purely used for using
    and testing the system to make sure each component type
    works within the main driver. This network may perform
    rather horribly on the dataset due to lack of nodes!
    """
    pass

def load_smaller_artnet():
    pass

def make_larger_artnet():
    """
    Just a larger ANN architecture for testing purposes
    as with the smaller artificial network architecture.
    """
    pass

def load_larger_artnet():
    pass

