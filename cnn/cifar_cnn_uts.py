#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# This file contains unit tests for the cifar cnn files.
#=================================================================

import unittest
from cnn.cifar_cnn_driver import *

class cifar_cnn_uts(unittest.TestCase):
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
    
    # tests for returning newly created cifar cnn
    def test_create_cifar_cnn(self):
        test_net = make_cifar_convnet()
        self.assertIsNotNone(test_net)

if __name__ == '__main__':
    unittest.main()