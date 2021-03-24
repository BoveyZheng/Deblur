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

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(opts[0]))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy/np.max(noisy)*np.max(image)
    

def blur(src,OTF):
    #current code assumes uint8 synthetic one channel OTF
    otf_img = Image.open(OTF)
    otf = np.array(otf_img)[:,:,0] 
    #Attach OTF to all channels
    #otf = np.stack((otf,otf,otf),2)
    print(otf.shape)
    #otf_img = np.sum(otf_img,2)/765
   
    img_spec1 = np.fft.fft2(src[:,:,0])
    img_spec1 = np.fft.fftshift(img_spec1)
    img_spec2 = np.fft.fft2(src[:,:,1])
    img_spec2 = np.fft.fftshift(img_spec2)
    img_spec3 = np.fft.fft2(src[:,:,2])
    img_spec3 = np.fft.fftshift(img_spec3)

    clipped_spectrum1 = img_spec1 * (otf/255)
    blurred1 = np.fft.ifftshift(clipped_spectrum1)
    clipped_spectrum2 = img_spec2 * (otf/255)
    blurred2 = np.fft.ifftshift(clipped_spectrum2)
    clipped_spectrum3 = img_spec3 * (otf/255)
    blurred3 = np.fft.ifftshift(clipped_spectrum3)

    layer1 = np.fft.ifft2(blurred1)
    layer2 = np.fft.ifft2(blurred2)
    layer3 = np.fft.ifft2(blurred3)

    output = np.stack((layer1,layer2,layer3),2)
    return abs(output)
    


def partitionDataset(imgs,outdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    for i in range(0,len(imgs)):
        
        #open image 
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)
        #gs_img = np.sum(src_img, 2)/3
     

        h,w = src_img.shape[0:2]

        j = 0
        while j < nreps:
            # random cropping 
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            img = src_img[r_rand:r_rand+dim,c_rand:c_rand+dim,0:]
            #img is a uint8, make a copy
            gt_img = img.copy()
            
            #Blur addition
            img = blur(img,'OTF_fiji.png')
            
            #Noise addition
            if 0 < np.random.rand() < 0.5:
                poisson_param = np.max([0,0.06*np.random.randn() + 0.5])
                img = noisy('poisson',img,[poisson_param])
            elif 0.5 < np.random.rand()< 1: 
                gauss_param = np.max([0, 0.1*np.random.randn() + 0.3])
                img = noisy('gauss',img,[0,gauss_param])

            # adding blur (random selection)
            # allotfs = sorted(glob.glob('OTFrange/*.png'))
            # rand_otf = random.choice(allotfs)
            # img = blur(img,rand_otf)

            
            filename = '%s/%d-%d.pkl' % (outdir,i,j)

            print(i,j,r_rand,c_rand,img.shape,gt_img.shape)

            img = Image.fromarray(img.astype('uint8'))
            gt_img = Image.fromarray(gt_img.astype('uint8'))
            pickle.dump((img,gt_img), open(filename,'wb'))

            combined = np.concatenate((np.array(img),np.array(gt_img)),axis=1)
            io.imsave(filename.replace(".pkl",".png"),combined)

            j += 1

        print('[%d/%d]' % (i+1,len(imgs)))


# --------------------------------------------

nreps = 5
dim = 128

allimgs = sorted(glob.glob('DIV2K_train_HR/*.png'))[200:800]
outdir = 'trainingdata/noisy_' + str(dim)
print('Training data')

partitionDataset(allimgs,outdir,nreps,dim,False)
