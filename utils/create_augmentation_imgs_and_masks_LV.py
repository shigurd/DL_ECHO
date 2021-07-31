import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, rescale, resize
from skimage import exposure
from skimage.util import random_noise, crop, img_as_ubyte
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import random
import os
import os.path as path
import sys


def Rotate(img, mask):
    
    rotRange = [[10, 21], [350, 361]]
    rangeChoice = random.choice(rotRange)
    degree = random.randrange(rangeChoice[0], rangeChoice[1])
    
    imgrot = rotate(img, angle=degree, mode='constant')
    maskrot = rotate(mask, angle=degree, mode='constant')
    
    return imgrot, maskrot

def Shift(img, mask):

    switch = [-1, 1] #venstre og høyre
    #y = random.randrange(10) * -1 #bare nedover
    y = 0
    x = random.randrange(20) * random.choice(switch)
    
    shift = AffineTransform(translation=(x, y))
    imgshi = warp(img, shift, mode='constant')    
    maskshi = warp(mask, shift, mode='constant')
  
    return imgshi, maskshi

def Zoom(img, mask):
    
    scale = 0.9 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgzoo = warp(img, shiftscale, mode='constant')    
    maskzoo = warp(mask, shiftscale, mode='constant')
    
    return imgzoo, maskzoo
    
def ZoomOut(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgshi = warp(img, shiftscale, mode='constant')    
    maskshi = warp(mask, shiftscale, mode='constant')
    
    return imgshi, maskshi

def Xwarp(img, mask):
    
    size = img.shape[0]
    maxScale = 1.1
    xWarp = int(size * maxScale)

    lCrop = int((xWarp - size) / 2) #cropper venstre
    rCrop = xWarp - size - lCrop #cropper høyre

    imgwarx = resize(img, (size, xWarp), anti_aliasing=True)
    maskwarx = resize(mask, (size, xWarp), anti_aliasing=True)
    
    return imgwarx, maskwarx
    
def XwarpOut(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    center = int(((scale - 1) / 2) * size) #senter av bildet ved å dele skalering på 2
    
    shiftwarp = AffineTransform(scale=(scale, 1), translation=(-center, 0))
    
    imgshi = warp(img, shiftwarp, mode='constant')
    maskshi = warp(mask, shiftwarp, mode='constant')
  
    return imgshi, maskshi
    
def Ywarp(img, mask):
    
    size = img.shape[0]
    maxscale = 1.1
    yWarp = int(size * maxscale)

    yCrop = yWarp - size #cropper nedre del

    imgwary = resize(img, (yWarp, size), anti_aliasing=True)
    maskwary = resize(mask, (yWarp, size), anti_aliasing=True)
    
    return imgwary, maskwary

def Noise(img, masknoi):

    sigma = 0.1
    imgnoi = random_noise(img, var=sigma**2)
    
    return imgnoi, masknoi

def Blur(img, maskblu):
        
    sigma = 1
    imgblu = gaussian(img, sigma=sigma, multichannel=True)
    
    return imgblu, maskblu

def GammaUp(img, maskgau):

    imggau = exposure.adjust_gamma(img, gamma=0.75,gain=1)
    
    return imggau, maskgau

def GammaDown(img, maskgad):

    imggad = exposure.adjust_gamma(img, gamma=1.5,gain=1)
    
    return imggad, maskgad


def Exposure(img, maskexp):

    v_min, v_max = np.percentile(img, (0, 95))
    imgexp = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    
    return imgexp, maskexp

def Base(imgname, imgpath, maskname, maskpath):

    img = io.imread(f'{imgpath}/{imgname}')
    mask = io.imread(f'{maskpath}/{maskname}')
    
    return img, mask
    

def create_augmentations(dataset, imgs_path, masks_path, n_augmention_copies, augmentations_dir_output):
    
    augmentations_dir_output_name = path.basename(augmentations_dir_output).rsplit('_', 1)[-1]
    
    imgsAug = path.join(augmentations_dir_output, f'imgs_{augmentations_dir_output_name}_{dataset}_MA{n_augmention_copies}')
    masksAug = path.join(augmentations_dir_output, f'masks_{augmentations_dir_output_name}_{dataset}_MA{n_augmention_copies}')
    os.mkdir(imgsAug)
    os.mkdir(masksAug)
    
    imgsFiles = os.listdir(imgs_path)
    
    min_aug = 5 #min aug er -1 så det er egt 4
    max_aug = 10 #max aug -1 så det er egt 9
    
    #husk img_as_ubyte
    
    for i in imgsFiles:
        
        imgName = i.rsplit('.', 1)[0]
        imgExt = '.' + i.rsplit('.', 1)[1]
        maskExt = '_mask.png'
        m = imgName + maskExt
        
        for x in range(1, n_augmention_copies + 1):
            
            if x > 9:
                img_save_path = f'{imgsAug}/{imgName}_MAX{x - 10}{imgExt}'
                mask_save_path = f'{masksAug}/{imgName}_MAX{x - 10}{maskExt}'
            else:
                img_save_path = f'{imgsAug}/{imgName}_MA{x}{imgExt}'
                mask_save_path = f'{masksAug}/{imgName}_MA{x}{maskExt}'
            
            temp = Base(i, imgs_path, m, masks_path) #åpner bilde. output er 2 bilder, img=0 og mask=1
            
            choice_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            aug_times = random.randrange(min_aug - 1, max_aug) #antall ganger hvert bilde skal augumenteres
            
            
            for y in range(aug_times):
                random_aug = random.randrange(0, len(choice_list))
                while choice_list[random_aug] == 1:
                    random_aug = random.randrange(0, len(choice_list))
                choice_list[random_aug] = 1
                
                
            if choice_list[0] == 1:
                temp = ZoomOut(temp[0], temp[1])
                                    
            if choice_list[1] == 1:
                temp = Xwarp(temp[0], temp[1])
                
            if choice_list[2] == 1:
                temp = XwarpOut(temp[0], temp[1])
              
            if choice_list[3] == 1:
                temp = Blur(temp[0], temp[1])
                
            if choice_list[4] == 1:
                temp = Rotate(temp[0], temp[1])
            
            if choice_list[5] == 1:
                temp = Shift(temp[0], temp[1])
                
            if choice_list[6] == 1:
                temp = Zoom(temp[0], temp[1])
            
            if choice_list[7] == 1:
                temp = GammaDown(temp[0], temp[1])
                
            if choice_list[8] == 1:
                temp = GammaUp(temp[0], temp[1])
            
            if choice_list[9] == 1:
                temp = Noise(temp[0], temp[1])
            
            size = 256
            width = temp[0].shape[0]
            height = temp[0].shape[1]
            
            cropl = int((width - size) / 2) #cropper likt begge sider
            cropr = width - size - cropl
            cropu = int((height - size) / 2) #cropper likt begge sider
            cropd = height - size - cropu

            tempimg = crop(temp[0], ((cropl, cropr), (cropu, cropd), (0,0)), copy=False)
            tempmask = crop(temp[1], ((cropl, cropr), (cropu, cropd)), copy=False)
            
            io.imsave(img_save_path, img_as_ubyte(tempimg))
            io.imsave(mask_save_path, img_as_ubyte(tempmask))


if __name__ == ' __main__':
    
    dataset_name = 'CAMUS1800'
    n_augmention_copies = 4

    datasets_imgs_dir = f'complete_datasets\imgs_{data_name}'
    datasets_masks_dir = f'complete_datasets\masks_{data_name}'
    
    create_augmentations(dataset_name, datasets_imgs_dir, datasets_masks_dir, n_augmention_copies)
    
