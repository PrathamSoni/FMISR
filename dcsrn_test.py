from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import logging

import tensorflow as tf

from tf_dcsrn import dcsrn, image_util

#Testing stream

#checkpoint/tfboard file path
output_path = "./snapshots/random_e4"
# setup
net = dcsrn.DCSRN(channels=3)

print("\nGraph set over.\n")

# testing step
test_provider = image_util.TestDataProvider()
test_x, test_y = test_provider(30)
result = net.predict("./snapshots/fixed_e4/model.cpkt", test_x)
