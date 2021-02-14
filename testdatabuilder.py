import glob
from PIL import Image
import random
import parser
import os
import scipy.ndimage as ndimage

import numpy as np
import pickle

from skimage.measure import compare_ssim
from skimage import io


def noisy(noise_typ,image,opts=[0,0.005]):
    if noise_typ == "gauss":
        mean = opts[0]
        var = opts[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = (image + 50*gauss)
        #noisy is float64
        noisy = np.clip(noisy,0,255)
        #clip to avoid black speckling
        noisy = noisy.astype(np.uint8)
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = opts[1]
        out = np.copy(image)
        # Salt mode NB 255 is used as out is a uint8
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in (row, col)]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in (row, col)]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(opts[0]))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy/np.max(noisy)*np.max(image)
    




def partitionTestset(imgs,imgoutdir,gtoutdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(imgoutdir)
        os.makedirs(gtoutdir)
    except OSError:
        pass

    for i in range(0,len(imgs)):
        
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)
                 
        # get rid of gba channels and invert
        # src_gt_img = src_gt_img[:,:,0]
  
        # normalize and add dimension
        # src_img = (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img)) 
        # src_gt_img = src_gt_img/255

   
        h,w = src_img.shape[0:2]

        j = 0
        while j < nreps:
            # random cropping 
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            img = src_img[r_rand:r_rand+dim,c_rand:c_rand+dim,0:]
            #img is a uint8
            gt_img = img.copy()


            # adding gaussian noise
            
            gauss_param = np.max([0, 0.1*np.random.randn() + 0.5])
            img = noisy('gauss',img,[0,gauss_param])
            
            
            filename = '%s/%d-%d_testimg.png' % (imgoutdir,i,j)
            gtfilename = '%s/%d-%d.png' % (gtoutdir,i,j)

            print(i,j,r_rand,c_rand,img.shape,gt_img.shape)

            img = Image.fromarray(img.astype('uint8'))
            gt_img = Image.fromarray(gt_img.astype('uint8'))
            pickle.dump((img,gt_img), open(filename,'wb'))

            io.imsave(filename,np.array(img))
            io.imsave(gtfilename,np.array(gt_img))


            j += 1

        print('[%d/%d]' % (i+1,len(imgs)))


# --------------------------------------------

nreps = 3
#dim = 128
dim = 256

allimgs = [
    "DIV2K_train_HR/0031.png",
    "DIV2K_train_HR/0032.png",
    "DIV2K_train_HR/0033.png",
    "DIV2K_train_HR/0034.png",
    "DIV2K_train_HR/0035.png",
    "DIV2K_train_HR/0036.png",
    "DIV2K_train_HR/0037.png",
    "DIV2K_train_HR/0038.png",
    "DIV2K_train_HR/0039.png",
    "DIV2K_train_HR/0040.png",
    "DIV2K_train_HR/0041.png",
    "DIV2K_train_HR/0042.png",
    "DIV2K_train_HR/0043.png",
    "DIV2K_train_HR/0044.png",
    "DIV2K_train_HR/0045.png",
    "DIV2K_train_HR/0046.png",
    "DIV2K_train_HR/0047.png",
    "DIV2K_train_HR/0048.png",
    "DIV2K_train_HR/0049.png",
    "DIV2K_train_HR/0050.png",
    "DIV2K_train_HR/0051.png",
    "DIV2K_train_HR/0052.png",
    "DIV2K_train_HR/0053.png",
    "DIV2K_train_HR/0054.png",
    "DIV2K_train_HR/0055.png",

]

imgoutdir = 'testdata/input/noisy_' + str(dim)
gtoutdir = 'testdata/ground_truth/noisy_' +str(dim)
print('Testing data')

partitionTestset(allimgs,imgoutdir,gtoutdir,nreps,dim,False)