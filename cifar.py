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
import unittest
import tensorflow as tf
from cnn.cifar_cnn import *
from cnn.cifar_cnn_driver import run_cnn_uts
from utils.utils import *
import utils.utils_uts as utils_uts

CNN_MODEL = 'nets/cnn/cnn_cifar100_model.tfl'
ANN_MODEL = 'nets/ann/ann_cifar100_model.tfl'
RAF_MODEL = 'nets/raf/raf_cifar100_model.tfl'

def main(): 
    clear_console()   
    arg = sys.argv.pop(1) if len(sys.argv) > 1 else 'standard'
    if arg == 'standard':
        load_test_trained_networks()
    elif arg == 'create':
        create_train_mini_networks()
    elif arg == 'testing':
        run_all_uts()
    elif arg == 'pass':
        pass
    else:
        print_useful_message(arg)

def load_test_trained_networks():
    print("Loading and testing convolutional neural network . . .")
    test_saved_cnn(
        CNN_MODEL,
        "01:04:41.953",
        cnn_type=0)
    print("\nLoading and testing artificial neural network . . .")
    test_saved_ann()
    print("\nLoading and testing random forest network . . .")
    test_saved_raf()
    print("\nLoading and testing of networks completed.")
    
def create_train_mini_networks():
    # create miniature convolutional network and train
    print("Creating and training convolutional network at 3 epochs . . .")
    mini_cnn = make_shallower_convnet()
    total_time = train_network(
        mini_cnn, 
        "temp_net", 
        "testing/temp/temp_net.tfl", 
        n_epoch=3)
    total_time = milliseconds_to_timestamp(total_time)
    test_saved_cnn(
        "testing/temp/temp_net.tfl", 
        total_time, 
        cnn_type=1)
    
    print("Creating and training artificial network at 3 epochs . . .")
    print("Creating and training random forest at 3 epochs . . .")

def test_saved_cnn(model_path, training_time, cnn_type=0):
    tf.reset_default_graph()
    trained_cnn_model = None
    if cnn_type == 0:
        trained_cnn_model = load_cifar_convnet(model_path)
    elif cnn_type == 1:
        trained_cnn_model = load_shallower_convnet(model_path)
    else:
        trained_cnn_model = load_example_convnet(model_path)
    print("============= CIFAR-100 CONVOLUTIONAL NETWORK STATS =============")
    cnn_valid_acc = test_tfl_model(
        trained_cnn_model, 
        validX, 
        validY)
    cnn_valid_acc = f"{round(cnn_valid_acc * 100, 2)}%"
    print(f"{'Validation Accuracy': <25} -> {cnn_valid_acc: <15}")
    cnn_test_acc = test_tfl_model(
        trained_cnn_model,
        testX,
        testY)
    cnn_test_acc = f"{round(cnn_test_acc * 100, 2)}%"
    print(f"{'Testing Accuracy': <25} -> {cnn_test_acc: <15}")
    print(f"{'Network Training Time': <25} -> {training_time: <15}")
    print("Model Architecture & Summary:")
    for element in tf.all_variables():
        print(element)

def test_saved_ann():
    tf.reset_default_graph()

    print("=============== CIFAR-100 ARTIFICIAL NETWORK STATS ===============")

def test_saved_raf():
    tf.reset_default_graph()

    print("================= CIFAR-100 RANDOM FOREST STATS =================")

def run_all_uts():
    print("> > > > > > > > > > > > UTILS UNIT TESTS < < < < < < < < < < < <")
    run_cnn_uts()
    print("> > > > > > > > > > > > > CNN UNIT TESTS < < < < < < < < < < < < <")
    run_utils_uts()
    print("> > > > > > > > > > > > > ANN UNIT TESTS < < < < < < < < < < < < <")
    print("> > > > > > > > > > > > > RAF UNIT TESTS < < < < < < < < < < < < <")

def run_utils_uts():
    # run all unit tests from utils_uts module
    suite = unittest.TestLoader().loadTestsFromModule(utils_uts)
    unittest.TextTestRunner(verbosity=2).run(suite)

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