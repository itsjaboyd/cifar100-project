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

    def test_create_smaller_ann(self):
        test_net = make_smaller_artnet()
        self.assertIsNotNone(test_net)

    def test_create_larger_ann(self):
        test_net = make_larger_artnet()
        self.assertIsNotNone(test_net)

    # tests for loading already created networks
    def test_loading_cifar_ann(self):
        tf.reset_default_graph()
        new_net = load_cifar_artnet("testing/ann/ca/utc_art.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_smaller_ann(self):
        tf.reset_default_graph()
        new_net = load_smaller_artnet("testing/ann/sa/uts_art.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_larger_ann(self):
        tf.reset_default_graph()
        new_net = load_larger_artnet("testing/ann/la/utl_art.tfl")
        self.assertIsNotNone(new_net)


if __name__ == '__main__':
    unittest.main()