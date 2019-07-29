import tensorflow as tf
from glob import glob
import os
import sys
import numpy as np
import nibabel as nib



def loadModel(session, path):
    saver = tf.train.Saver()
    print("1111111111111111111111111111")
    ckpt = tf.train.get_checkpoint_state(path)
    print("22222222222222222222222222222")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Checkpoint restored')
    else:
        print('No checkpoint found')
        exit()

with tf.Session() as sess:
	loadModel(sess, "./model.cpkt")

