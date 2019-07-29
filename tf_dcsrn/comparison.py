from __future__ import print_function, division, absolute_import, unicode_literals

import os
import sys
from glob import glob
import nibabel as nib
import numpy as np
sys.path.append("../")
import math

from tf_dcsrn.tfSSIM import(tf_ssim)
import random
import tensorflow as tf
import scipy.ndimage
import scipy
hr_list=glob("/home/psoni/Desktop/project/psoni/MRISR/Test/*.nii.gz")
outdir="/home/psoni/Desktop/project/psoni/MRISR/test_outs/"
init = tf.global_variables_initializer()

    
def interpolate(hr_data, order):
    scale=2-random.random()
    '''if random.random()>2./3.:
        scale=2.
    elif random.random()>1./3.:
        scale=1.6
    else:
        scale=1.2'''
    print(scale)
    img_out=np.zeros(hr_data.shape)
    
    image0=hr_data[:,:,:,0]  
    image1=hr_data[:,:,:,1]
    image2=hr_data[:,:,:,2]
    
    image0=scipy.ndimage.zoom(image0, 1./scale, order=order)
    image1=scipy.ndimage.zoom(image1, 1./scale, order=order)
    image2=scipy.ndimage.zoom(image2, 1./scale, order=order)
    
    image0=scipy.ndimage.zoom(image0, (float(img_out.shape[0])/image0.shape[0],float(img_out.shape[1])/image0.shape[1],float(img_out.shape[2])/image0.shape[2]), order=order)
    image1=scipy.ndimage.zoom(image1, (float(img_out.shape[0])/image1.shape[0],float(img_out.shape[1])/image1.shape[1],float(img_out.shape[2])/image1.shape[2]), order=order)
    image2=scipy.ndimage.zoom(image2, (float(img_out.shape[0])/image2.shape[0],float(img_out.shape[1])/image2.shape[1],float(img_out.shape[2])/image2.shape[2]), order=order)
    
    img_out[:,:,:,0]=image0
    img_out[:,:,:,1]=image1
    img_out[:,:,:,2]=image2
    
    return img_out
    
def PSNR(low,high):
    mse=np.mean(np.square(high-low))
    return 20*math.log10(1./math.sqrt(mse))
    
with tf.Session() as sess:
    sess.run(init)
    for order_ in [0,3]:
        index=0
        print(order_)
        for GT in hr_list:
            image=np.array(nib.load(GT).dataobj)
            out_image=interpolate(image, order_)
            PSNR_=PSNR(out_image, image)
            ssim_=tf_ssim(tf.cast(tf.reshape(out_image, [1, 145, -1, 1]), tf.float32), tf.cast(tf.reshape(image, [1, 145, -1, 1]), tf.float32))
            print(PSNR_, ssim_.eval())
            save_image=nib.nifti1.Nifti1Image(out_image, np.eye(4))
            if order_==0:
                nib.save(save_image, outdir+"NN_random/"+str(index)+"_NN.nii.gz")
            else:
                nib.save(save_image, outdir+"bicubic_random/"+str(index)+"_bicubic.nii.gz")
    
            index+=1
    
