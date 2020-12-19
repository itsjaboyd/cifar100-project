#!/usr/bin/env python

#=================================================================
# Author: Jason Boyd
# USU A#: A02258798
#
# -- Description --
# This file serves as the orchestrator for bringing the unit tests
# and random forest network structures together for using within
# cifar.py and defines other useful functions.
#=================================================================

import unittest
import raf.cifar_raf_uts as raf_unit_tests

def run_raf_uts():
    # run all unit tests from cifar_cnn_uts module
    suite = unittest.TestLoader().loadTestsFromModule(raf_unit_tests)
    unittest.TextTestRunner(verbosity=2).run(suite)