import numpy as np
import os
import glob
from PIL import Image
import sys
sys.path.append('../')
#Load all images from files  
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

def getLR(hr_data):
    #LR function based on FFT decomposition
    imgfft = np.fft.fftn(hr_data)
    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2
    imgfft[x_center-20 : x_center+20, y_center-20 : y_center+20, z_center-20 : z_center+20] = 0
    imgifft = np.fft.ifftn(imgfft)
    img_out = abs(imgifft)

    return img_out
