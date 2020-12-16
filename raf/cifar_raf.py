#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd (A02258798)
# Date: December 9, 2020 5:41 PM
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

