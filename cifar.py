#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
# 
# -- Description --
# This file serves as the executive python script to run and
# return statistics on every saved net within the project. For
# more information on the project itself, refer to the README.md. 
# Questions or concerns can be emailed to [jasonboyd99@gmail.com].
#=================================================================

import sys
from cnn.cifar_cnn_driver import *

CNN_MODEL = 'nets/cnn/cnn_cifar100_model.tfl'
ANN_MODEL = 'nets/ann/ann_cifar100_model.tfl'
RAF_MODEL = 'nets/raf/raf_cifar100_model.tfl'

def main():    
    arg = sys.argv.pop(1) if len(sys.argv) > 1 else 'standard'
    if arg == 'standard':
        load_test_trained_networks()
    elif arg == 'create':
        create_train_mini_networks()
    else:
        print_useful_message(arg)

    
    

def load_test_trained_networks():
    pass

def create_train_mini_networks():
    pass

def test_saved_cnn():
    pass

def test_saved_ann():
    pass

def test_saved_raf():
    pass

def print_useful_message(arg):
    print("-------------------- PROJECT RUN UNSUCCESSFUL --------------------")
    print(f"User supplied unknown command line argument -> {arg}")
    print("Please supply file with following argument options:\n")
    print("1. 'standard' or no argument")
    print("Load and test supplied best performing neural networks.\n")
    print("2. 'create' argument")
    print("Create and breifly train small versions of neural networks.")

if __name__ == "__main__":
    main()