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
import cnn.cifar_cnn as cnn
import ann.cifar_ann as ann
from cnn.cifar_cnn_driver import run_cnn_uts
from ann.cifar_ann_driver import run_ann_uts
from raf.cifar_raf_driver import run_raf_uts
import utils.utils as utilities
import utils.utils_uts as utils_uts

CNN_MODEL = 'nets/cnn/cnn_cifar100_model.tfl'
ANN_MODEL = 'nets/ann/ann_cifar100_model.tfl'
RAF_MODEL = 'nets/raf/raf_cifar100_model.tfl'

def main(): 
    utilities.clear_console()   
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
        "06:19:31.953",
        cnn_type=0)
    print("\nLoading and testing artificial neural network . . .")
    test_saved_ann(
        ANN_MODEL,
        "02:52:23.717",
        ann_type=0)
    print("\nLoading and testing random forest network . . .")
    test_saved_raf()
    print("\nLoading and testing of networks completed.")

    # run all unit tests after testing of saved networks is complete.
    print("Running associated unit tests . . .")
    run_all_uts()
    
def create_train_mini_networks():
    # create miniature convolutional network and train
    print("Creating and training convolutional network at 3 epochs . . .")
    mini_cnn = cnn.make_shallower_convnet()
    total_time_cnn = utilities.train_network(
        mini_cnn, 
        "temp_net", 
        "testing/temp/cnn/temp_net.tfl", 
        n_epoch=3)
    total_time_cnn = utilities.milliseconds_to_timestamp(total_time_cnn)
    test_saved_cnn(
        "testing/temp/cnn/temp_net.tfl", 
        total_time_cnn, 
        cnn_type=1)
    
    tf.reset_default_graph()
    print("\nCreating and training artificial network at 3 epochs . . .")
    mini_ann = ann.make_smaller_artnet()
    total_time_ann = utilities.train_network(
        mini_ann,
        "temp_net",
        "testing/temp/ann/temp_net.tfl",
        n_epoch=3)
    total_time_ann = utilities.milliseconds_to_timestamp(total_time_ann)
    test_saved_ann(
        "testing/temp/ann/temp_net.tfl",
        total_time_ann,
        ann_type=1)

    print("\nCreating and training random forest . . .")
    print("Could not create and train random forest.")

def test_saved_cnn(model_path, training_time, cnn_type=0):
    tf.reset_default_graph()
    trained_cnn_model = None
    if cnn_type == 0:
        trained_cnn_model = cnn.load_cifar_convnet(model_path)
    elif cnn_type == 1:
        trained_cnn_model = cnn.load_shallower_convnet(model_path)
    else:
        trained_cnn_model = cnn.load_example_convnet(model_path)
    print("============= CIFAR-100 CONVOLUTIONAL NETWORK STATS =============")
    cnn_valid_acc = utilities.test_tfl_model(
        trained_cnn_model, 
        utilities.validX, 
        utilities.validY)
    cnn_valid_acc = f"{round(cnn_valid_acc * 100, 2)}%"
    print(f"{'Validation Accuracy': <25} -> {cnn_valid_acc: <15}")
    cnn_test_acc = utilities.test_tfl_model(
        trained_cnn_model,
        utilities.testX,
        utilities.testY)
    cnn_test_acc = f"{round(cnn_test_acc * 100, 2)}%"
    print(f"{'Testing Accuracy': <25} -> {cnn_test_acc: <15}")
    print(f"{'Network Training Time': <25} -> {training_time: <15}")
    print("Model Architecture & Summary:")
    for element in tf.all_variables():
        print(element)

def test_saved_ann(model_path, training_time, ann_type=0):
    tf.reset_default_graph()
    trained_ann_model = None
    if ann_type == 0:
        trained_ann_model = ann.load_cifar_artnet(model_path)
    elif ann_type == 1:
        trained_ann_model = ann.load_smaller_artnet(model_path)
    else:
        trained_ann_model = ann.load_larger_artnet(model_path)
    print("=============== CIFAR-100 ARTIFICIAL NETWORK STATS ===============")
    ann_valid_acc = utilities.test_tfl_model(
        trained_ann_model,
        utilities.validX,
        utilities.validY)
    ann_valid_acc = f"{round(ann_valid_acc * 100, 2)}%"
    print(f"{'Validation Accuracy': <25} -> {ann_valid_acc: <15}")
    ann_test_acc = utilities.test_tfl_model(
        trained_ann_model,
        utilities.testX,
        utilities.testY)
    ann_test_acc = f"{round(ann_test_acc * 100, 2)}%"
    print(f"{'Testing Accuracy': <25} -> {ann_test_acc: <15}")
    print(f"{'Network Training Time': <25} -> {training_time: <15}")
    print("Model Architecture & Summary:")
    for element in tf.all_variables():
        print(element)

def test_saved_raf():
    tf.reset_default_graph()
    print("================= CIFAR-100 RANDOM FOREST STATS =================")
    print("Could not test any saved random forest networks.")

def run_all_uts():
    print("> > > > > > > > > > > > UTILS UNIT TESTS < < < < < < < < < < < <")
    run_utils_uts()
    print("> > > > > > > > > > > > > CNN UNIT TESTS < < < < < < < < < < < < <")
    run_cnn_uts()
    print("> > > > > > > > > > > > > ANN UNIT TESTS < < < < < < < < < < < < <")
    run_ann_uts()
    print("> > > > > > > > > > > > > RAF UNIT TESTS < < < < < < < < < < < < <")
    run_raf_uts()

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
    print("Create and breifly train small versions of neural networks.\n")
    print("3. 'testing' argument")
    print("Run all unit tests associated with every type of network.")

if __name__ == "__main__":
    main()
