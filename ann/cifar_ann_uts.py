#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# This file contains unit tests for the cifar ann files.
#=================================================================

import unittest
import tensorflow as tf
from ann.cifar_ann import *


class cifar_ann_uts(unittest.TestCase):

    # tests for returning newly created networks
    def test_create_cifar_ann(self):
        test_net = make_cifar_artnet()
        self.assertIsNotNone(test_net)

    def test_create_shallow_cnn(self):
        test_net = make_smaller_artnet()
        self.assertIsNotNone(test_net)

    def test_create_example_cnn(self):
        test_net = make_larger_artnet()
        self.assertIsNotNone(test_net)

    # tests for loading already created networks
    def test_loading_cifar_ann(self):
        tf.reset_default_graph()
        new_net = load_cifar_convnet("testing/ann/cc/utc_art.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_shallow_cnn(self):
        tf.reset_default_graph()
        new_net = load_shallower_convnet("testing/ann/sc/uts_art.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_example_cnn(self):
        tf.reset_default_graph()
        new_net = load_example_convnet("testing/ann/ec/utl_art.tfl")
        self.assertIsNotNone(new_net)


if __name__ == '__main__':
    unittest.main()
