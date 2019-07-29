from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import logging

import tensorflow as tf

from tf_dcsrn import dcsrn, image_util

output_path = "./snapshots/random_e4"
# path of dataset, here is the HCP dataset
dataset_HCP = "Train"

#preparing data loading, you may want to explicitly note the glob search path on you data 
data_provider = image_util.MedicalImageDataProvider()

print("\ndata_provider initialization over.\n")

# setup & training
net = dcsrn.DCSRN(channels=3)

print("\nGraph set over.\n")

trainer = dcsrn.Trainer(net)

print("\nBegin to train.\n")

#path = trainer.train(data_provider, output_path, restore = True)

print("\nTraining process is over.\n")

# verification, randomly test 4 images
test_provider = image_util.TestDataProvider()
test_x, test_y = test_provider(30)
result = net.predict("./snapshots/fixed_e4/model.cpkt", test_x)
