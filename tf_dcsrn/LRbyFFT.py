import numpy as np
import os
import glob
from PIL import Image
import sys
sys.path.append('../')
import random
import scipy.ndimage

def get_all_images():
    filepaths = glob.glob('../../HCP_NPY/*.npy')
    images = []
    for filename in filepaths:
        img_npy = np.load(filename)
        img = np.array(img_npy, dtype = np.float32)
        images.append(img)
    print("All images are saved in images, shape " + str(np.shape(images)))
    return images

# Get all images from the NPY files
images = get_all_images()
#LR sample generator
def getLR(hr_data):
    #define a random scale
    scale=2-random.random()
    #breakdown to discrete level
    '''if random.random()>2./3.:
        scale=2.
    elif random.random()>1./3.:
        scale=1.6
    else:
        scale=1.2'''
    print(scale)
    #dummy array
    img_out=np.zeros(hr_data.shape)
    #slice out of HR
    image0=hr_data[:,:,:,0]  
    image1=hr_data[:,:,:,1]
    image2=hr_data[:,:,:,2]
    #interpolate down
    image0=scipy.ndimage.zoom(image0, 1./scale)
    image1=scipy.ndimage.zoom(image1, 1./scale)
    image2=scipy.ndimage.zoom(image2, 1./scale)
    #interpolate up
    image0=scipy.ndimage.zoom(image0, (float(img_out.shape[0])/image0.shape[0],float(img_out.shape[1])/image0.shape[1],float(img_out.shape[2])/image0.shape[2]))
    image1=scipy.ndimage.zoom(image1, (float(img_out.shape[0])/image1.shape[0],float(img_out.shape[1])/image1.shape[1],float(img_out.shape[2])/image1.shape[2]))
    image2=scipy.ndimage.zoom(image2, (float(img_out.shape[0])/image2.shape[0],float(img_out.shape[1])/image2.shape[1],float(img_out.shape[2])/image2.shape[2]))
    #cop slices
    img_out[:,:,:,0]=image0
    img_out[:,:,:,1]=image1
    img_out[:,:,:,2]=image2
    
    return img_out
