#script to generate composite image patches
import numpy as np
import os
import glob
import nibabel as nib
import sys

outdir = '/home/au869/Desktop/psoni/psoni/MRISR/Train'
if not os.path.exists(outdir):
    os.makedirs(outdir)

files = glob.glob("/home/au869/Desktop/psoni/psoni/MRISR/extra_train/*.nii.gz")
for filepath in files:
    file = np.array(nib.load(filepath).dataobj)
    print('  Data shape is ' + str(file.shape) + ' .')
    for i in range(0, 100):
    #generate 100 patches
        #randomly choose starting coordinate
        x = int(np.floor((file.shape[0] - 64) * np.random.rand(1))[0])
        y = int(np.floor((file.shape[1] - 64) * np.random.rand(1))[0])
        z = int(np.floor((file.shape[2] - 64) * np.random.rand(1))[0])
        
        #crop and preprocess each channel
        #Order 0
        file_aug0= file[x:x+64, y:y+64, z:z+64, 0]
        file_aug0=file_aug0/np.percentile(file_aug0,99)
        file_aug0[file_aug0>1]=1
        file_aug0[file_aug0<0]=sys.float_info.epsilon
        #Order 2
        file_aug1= file[x:x+64, y:y+64, z:z+64, 1]
        file_aug1=file_aug1/np.percentile(file_aug1,99)
        file_aug1[file_aug1>1]=1
        file_aug1[file_aug1<0]=sys.float_info.epsilon
        #Order 4
        file_aug2= file[x:x+64, y:y+64, z:z+64, 2]
        file_aug2=file_aug2/np.percentile(file_aug2,99)
        file_aug2[file_aug2>1]=1
        file_aug2[file_aug2<0]=sys.float_info.epsilon
        #create output array
        file_aug=np.zeros((64,64,64,3)) 
        file_aug[:,:,:,0]=file_aug0
        file_aug[:,:,:,1]=file_aug1
        file_aug[:,:,:,2]=file_aug2
        #save nifiti out
        filename_ = filepath.split('/')[-1].split('.')[0]
        filename_ = filename_ + '_' + str(i) + '.nii.gz'
        filename = os.path.join(outdir, filename_)
        
        
        new_image = nib.Nifti1Image(file_aug, affine=np.eye(4))
        print(filename)
        nib.save(new_image, filename)
        print(str(i))
    print('All sliced files of ' + filepath + ' are saved.')
