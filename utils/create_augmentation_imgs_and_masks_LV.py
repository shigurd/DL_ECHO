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


def rotate(img, mask):
    
    rotRange = [[10, 21], [350, 361]]
    rangeChoice = random.choice(rotRange)
    degree = random.randrange(rangeChoice[0], rangeChoice[1])
    
    imgrot = rotate(img, angle=degree, mode='constant')
    maskrot = rotate(mask, angle=degree, mode='constant')
    
    return imgrot, maskrot


def shift(img, mask):

    switch = [-1, 1] #venstre og høyre
    #y = random.randrange(10) * -1 #bare nedover
    y = 0
    x = random.randrange(20) * random.choice(switch)
    
    shift = AffineTransform(translation=(x, y))
    imgshi = warp(img, shift, mode='constant')    
    maskshi = warp(mask, shift, mode='constant')
  
    return imgshi, maskshi


def zoom_in(img, mask):
    
    scale = 0.9 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgzoo = warp(img, shiftscale, mode='constant')    
    maskzoo = warp(mask, shiftscale, mode='constant')
    
    return imgzoo, maskzoo


def zoom_out(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgshi = warp(img, shiftscale, mode='constant')    
    maskshi = warp(mask, shiftscale, mode='constant')
    
    return imgshi, maskshi


def x_warp_in(img, mask):
    
    size = img.shape[0]
    maxScale = 1.1
    xWarp = int(size * maxScale)

    lCrop = int((xWarp - size) / 2) #cropper venstre
    rCrop = xWarp - size - lCrop #cropper høyre

    imgwarx = resize(img, (size, xWarp), anti_aliasing=True)
    maskwarx = resize(mask, (size, xWarp), anti_aliasing=True)
    
    return imgwarx, maskwarx


def x_warp_out(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    center = int(((scale - 1) / 2) * size) #senter av bildet ved å dele skalering på 2
    
    shiftwarp = AffineTransform(scale=(scale, 1), translation=(-center, 0))
    
    imgshi = warp(img, shiftwarp, mode='constant')
    maskshi = warp(mask, shiftwarp, mode='constant')
  
    return imgshi, maskshi


def y_warp_in(img, mask):
    
    size = img.shape[0]
    maxscale = 1.1
    yWarp = int(size * maxscale)

    yCrop = yWarp - size #cropper nedre del

    imgwary = resize(img, (yWarp, size), anti_aliasing=True)
    maskwary = resize(mask, (yWarp, size), anti_aliasing=True)
    
    return imgwary, maskwary


def noise(img, masknoi):

    sigma = 0.1
    imgnoi = random_noise(img, var=sigma**2)
    
    return imgnoi, masknoi


def blur(img, maskblu):
        
    sigma = 1
    imgblu = gaussian(img, sigma=sigma, multichannel=True)
    
    return imgblu, maskblu


def gamma_up(img, maskgau):

    imggau = exposure.adjust_gamma(img, gamma=0.75,gain=1)
    
    return imggau, maskgau


def gamma_down(img, maskgad):

    imggad = exposure.adjust_gamma(img, gamma=1.5,gain=1)
    
    return imggad, maskgad


def exposure(img, maskexp):

    v_min, v_max = np.percentile(img, (0, 95))
    imgexp = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    
    return imgexp, maskexp


def base(imgname, imgpath, maskname, maskpath):

    img = io.imread(path.join(imgpath, imgname))
    mask = io.imread(path.join(maskpath, maskname))
    
    return img, mask
    

def create_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output):

    imgs_path = path.join(datasets_dir, 'imgs', dataset_name)
    masks_path = path.join(datasets_dir, 'masks', dataset_name)
    
    imgs_aug = path.join(augmentations_dir_output, 'imgs', f'{dataset_name}_MA{n_augmention_copies}')
    masks_aug = path.join(augmentations_dir_output, 'masks', f'{dataset_name}_MA{n_augmention_copies}')
    os.mkdir(imgs_aug)
    os.mkdir(masks_aug)
    
    imgs_files = os.listdir(imgs_path)
    
    min_aug = 5 #min aug er -1 så det er egt 4
    max_aug = 10 #max aug -1 så det er egt 9
    
    #husk img_as_ubyte
    
    for i in imgs_files:
        
        img_name = i.rsplit('.', 1)[0]
        img_ext = '.' + i.rsplit('.', 1)[1]
        mask_ext = '_mask.png'
        m = img_name + mask_ext
        
        for x in range(1, n_augmention_copies + 1):
            
            if x > 9:
                img_save_path = path.join(imgs_aug, f'{img_name}_MAX{x - 10}{img_ext}')
                mask_save_path = path.join(masks_aug, f'{img_name}_MAX{x - 10}{mask_ext}')
            else:
                img_save_path = path.join(imgs_aug, f'{img_name}_MA{x}{img_ext}')
                mask_save_path = path.join(masks_aug, f'{img_name}_MA{x}{mask_ext}')
            
            temp = base(i, imgs_path, m, masks_path) #åpner bilde. output er 2 bilder, img=0 og mask=1
            
            choice_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            aug_times = random.randrange(min_aug - 1, max_aug) #antall ganger hvert bilde skal augumenteres

            for y in range(aug_times):
                random_aug = random.randrange(0, len(choice_list))
                while choice_list[random_aug] == 1:
                    random_aug = random.randrange(0, len(choice_list))
                choice_list[random_aug] = 1
                
            if choice_list[0] == 1:
                temp = zoom_out(temp[0], temp[1])
                                    
            if choice_list[1] == 1:
                temp = x_warp_in(temp[0], temp[1])
                
            if choice_list[2] == 1:
                temp = x_warp_out(temp[0], temp[1])
              
            if choice_list[3] == 1:
                temp = blur(temp[0], temp[1])
                
            if choice_list[4] == 1:
                temp = rotate(temp[0], temp[1])
            
            if choice_list[5] == 1:
                temp = shift(temp[0], temp[1])
                
            if choice_list[6] == 1:
                temp = zoom_in(temp[0], temp[1])
            
            if choice_list[7] == 1:
                temp = gamma_down(temp[0], temp[1])
                
            if choice_list[8] == 1:
                temp = gamma_up(temp[0], temp[1])
            
            if choice_list[9] == 1:
                temp = noise(temp[0], temp[1])
            
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

    dataset_name = 'CAMUS1800_HML'
    n_augmention_copies = 4

    datasets_dir = 'datasets'
    augmentations_dir_output = 'augmentations'

    create_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output)
    
