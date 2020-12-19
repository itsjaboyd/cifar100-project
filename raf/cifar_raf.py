#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
# 
# -- Description --
# This file creates the various random forest network
# designs using the CIFAR-100 dataset loaded in from TensorFlow.
#=================================================================

from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def make_cifar_raf():
    # 1. determine best number of decision trees
    # 2. create the random forest with DTs
    # 3. return the random forest model
    pass

def make_larger_raf():
    # define larger number of decision trees
    # return larger model random forest
    pass

def make_smaller_raf():
    # define smaller number of decrision trees
    # return smaller model random forest
    pass
