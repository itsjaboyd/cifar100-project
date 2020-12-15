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
    clear_console()   
    arg = sys.argv.pop(1) if len(sys.argv) > 1 else 'standard'
    if arg == 'standard':
        load_test_trained_networks()
    elif arg == 'create':
        create_train_mini_networks()
    elif arg == 'testing':
        run_cnn_tests()
    elif arg == 'pass':
        pass
    else:
        print_useful_message(arg)

def load_test_trained_networks():
    print("Loading and testing convolutional neural network . . .")
    test_saved_cnn(
        "nets/cnn/test_conv.tfl",
        "02:04:46.599",
        cnn_type=0)
    print("\nLoading and testing artificial neural network . . .")
    test_saved_ann()
    print("\nLoading and testing random forest network . . .")
    test_saved_raf()
    print("\nLoading and testing of networks completed.")
    

def create_train_mini_networks():
    # create miniature convolutional network and train
    print("Creating and training network at 3 epochs . . .")
    mini_cnn = make_shallower_convnet()
    total_time = train_convnet(mini_cnn, 3)
    total_time = milliseconds_to_timestamp(total_time)
    test_saved_cnn(
        "nets/temp/cnn/test_conv.tfl", 
        total_time, 
        cnn_type=1)

def test_saved_cnn(model_path, training_time, cnn_type=0):
    tf.reset_default_graph()
    trained_cnn_model = None
    if cnn_type == 0:
        trained_cnn_model = load_cifar_convnet(model_path)
    elif cnn_type == 1:
        trained_cnn_model = load_shallower_convnet(model_path)
    else:
        trained_cnn_model = load_example_convnet(model_path)
    cnn_acc = test_tfl_model(
        trained_cnn_model, 
        validX, 
        validY)
    cnn_acc = f"{round(cnn_acc * 100, 2)}%"
    print("============= CIFAR-100 CONVOLUTIONAL NETWORK STATS =============")
    print(f"{'Validation Accuracy': <25} -> {cnn_acc: <15}")
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