#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# This file contains unit tests for the cifar cnn files.
#=================================================================

import unittest
import tensorflow as tf
import cnn.cifar_cnn_driver as cdr

CNN_TESTING_RANGE = 1000

class cifar_cnn_uts(unittest.TestCase):

    # tests for milliseconds to timestamp function
    def test_milliseconds_to_timestamp(self):
        mils = cdr.milliseconds_to_timestamp(999)
        self.assertEqual(mils, "00:00:00.999")
        secs = cdr.milliseconds_to_timestamp(1001)
        self.assertEqual(secs, "00:00:01.001")
        mins = cdr.milliseconds_to_timestamp(61123)
        self.assertEqual(mins, "00:01:01.123")
        hour = cdr.milliseconds_to_timestamp(3612345)
        self.assertEqual(hour, "01:00:12.345")
    
    # tests for returning newly created networks
    def test_create_cifar_cnn(self):
        test_net = cdr.make_cifar_convnet()
        self.assertIsNotNone(test_net)

    def test_create_shallow_cnn(self):
        test_net = cdr.make_shallower_convnet()
        self.assertIsNotNone(test_net)
    
    def test_create_example_cnn(self):
        test_net = cdr.make_example_convnet()
        self.assertIsNotNone(test_net)

    # tests for loading already created networks
    def test_loading_cifar_cnn(self):
        tf.reset_default_graph()
        new_net = cdr.load_cifar_convnet("testing/cnn/cc/utc_conv.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_shallow_cnn(self):
        tf.reset_default_graph()
        new_net = cdr.load_shallower_convnet("testing/cnn/sc/uts_conv.tfl")
        self.assertIsNotNone(new_net)

    def test_loading_example_cnn(self):
        tf.reset_default_graph()
        new_net = cdr.load_example_convnet("testing/cnn/ec/ute_conv.tfl")
        self.assertIsNotNone(new_net)
    
if __name__ == '__main__':
    unittest.main()