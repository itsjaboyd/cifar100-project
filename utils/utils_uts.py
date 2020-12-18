#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# This file contains unit tests for the cifar cnn files.
#=================================================================

import unittest
import tensorflow as tf
from utils.utils import *

TESTING_RANGE = 1000

class utils_uts(unittest.TestCase):

    # tests for milliseconds to timestamp function
    def test_milliseconds_to_timestamp(self):
        mils = milliseconds_to_timestamp(999)
        self.assertEqual(mils, "00:00:00.999")
        secs = milliseconds_to_timestamp(1001)
        self.assertEqual(secs, "00:00:01.001")
        mins = milliseconds_to_timestamp(61123)
        self.assertEqual(mins, "00:01:01.123")
        hour = milliseconds_to_timestamp(3612345)
        self.assertEqual(hour, "01:00:12.345")

    def test_create_simple_network(self):
        test_net = make_simple_network()
        self.assertIsNotNone(test_net)

    def test_load_simple_network(self):
        tf.reset_default_graph()
        new_net = load_simple_network("testing/utils/simple_network.tfl")
        self.assertIsNotNone(new_net)

    def test_train_simple_network(self):
        tf.reset_default_graph()
        test_net = make_simple_network()
        time_stamp = train_network(
            test_net, 
            "test",
            "testing/temp/temp_net.tfl",
            n_epoch=1)
        self.assertIsNotNone(time_stamp)

    def test_test_tfl_model(self):
        tf.reset_default_graph()
        test_net = load_simple_network("testing/utils/simple_network.tfl")
        test_acc = test_tfl_model(test_net, testX, testY)
        self.assertIsNotNone(test_acc)

if __name__ == '__main__':
    unittest.main()
