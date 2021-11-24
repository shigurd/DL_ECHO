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
from glob import glob
from tqdm import tqdm

class MyAugmentations:
    def __init__(self, img_path, masks_paths):
        self.img = io.imread(img_path)
        self.masks = [io.imread(mask_path) for mask_path in masks_paths]
        self.y_size = self.img.shape[0]
        self.x_size = self.img.shape[1]
        if len(self.img.shape) == 3:
            self.is_rgb = True
        else:
            self.is_rgb = False

    def my_rotate(self):
        ''' angles for rotation '''
        rot_range = [[10, 21], [350, 361]]
        range_choice = random.choice(rot_range)
        degree = random.randrange(range_choice[0], range_choice[1])

        img_rot = rotate(self.img, angle=degree, mode='constant')
        masks_rot = [rotate(mask, angle=degree, mode='constant') for mask in self.masks]

        self.img = img_rot
        self.masks = masks_rot

    def my_shift(self):
        ''' only left and right shifting '''
        switch = [-1, 1]
        ''' only downwards shifting '''
        #y = random.randrange(10) * -1
        y = 0
        x = random.randrange(20) * random.choice(switch)

        shift = AffineTransform(translation=(x, y))
        img_shi = warp(self.img, shift, mode='constant')
        masks_shi = [warp(mask, shift, mode='constant') for mask in self.masks]

        self.img = img_shi
        self.masks = masks_shi

    def my_zoom_in(self):
        ''' downscales the canvas '''
        scale = 0.9
        ''' only if width and height are the same '''
        size = self.y_size
        ''' canvas upscaling increases range, image then has to be centered'''
        move_range = int(((scale - 1) * size) / 2)

        ''' all movement is towards the right since the image is not centered '''
        y = move_range * -1
        x = move_range * -1

        shift_scale = AffineTransform(scale=(scale, scale), translation=(x, y))

        img_zin = warp(self.img, shift_scale, mode='constant')
        masks_zin = [warp(mask, shift_scale, mode='constant') for mask in self.masks]

        self.img = img_zin
        self.masks = masks_zin

    def my_zoom_out(self):
        ''' upscales the canvas '''
        scale = 1.1
        ''' only if width and height are the same '''
        size = self.y_size
        ''' canvas upscaling increases range, image then has to be centered'''
        move_range = int(((scale - 1) * size) / 2)

        ''' all movement is towards the right since the image is not centered '''
        y = move_range * -1
        x = move_range * -1

        shift_scale = AffineTransform(scale=(scale, scale), translation=(x, y))

        img_zou = warp(self.img, shift_scale, mode='constant')
        masks_zou = [warp(mask, shift_scale, mode='constant') for mask in self.masks]

        self.img = img_zou
        self.masks = masks_zou

    def my_x_warp_in(self):
        ''' only if width and height are the same '''
        size = self.y_size
        max_scale = 1.1
        x_warp = int(size * max_scale)

        img_wxo = resize(self.img, (size, x_warp), anti_aliasing=True)
        masks_wxo = [resize(mask, (size, x_warp), anti_aliasing=True) for mask in self.masks]

        self.img = img_wxo
        self.masks = masks_wxo

    def my_x_warp_out(self):
        ''' scales up canvas '''
        scale = 1.1
        ''' only if width and height are the same '''
        size = self.y_size
        ''' centers the image by dividing the upscale by 2 '''
        center = int(((scale - 1) / 2) * size)

        shift_warp = AffineTransform(scale=(scale, 1), translation=(-center, 0))

        img_wxi = warp(self.img, shift_warp, mode='constant')
        masks_wxi = [warp(mask, shift_warp, mode='constant') for mask in self.masks]

        self.img = img_wxi
        self.masks = masks_wxi

    def my_y_warp_in(self):
        ''' only if width and height are the same '''
        size = self.y_size
        max_scale = 1.1
        y_warp = int(size * max_scale)

        img_wyi = resize(self.img, (y_warp, size), anti_aliasing=True)
        masks_wyi = [resize(mask, (y_warp, size), anti_aliasing=True) for mask in self.masks]

        self.img = img_wyi
        self.masks = masks_wyi

    def my_noise(self):
        sigma = 0.1
        img_noi = random_noise(self.img, var=sigma ** 2)

        self.img = img_noi

    def my_blur(self):
        sigma = 1
        img_blu = gaussian(self.img, sigma=sigma, multichannel=True)

        self.img = img_blu

    def my_gamma_up(self):
        img_gau = exposure.adjust_gamma(self.img, gamma=0.75, gain=1)

        self.img = img_gau

    def my_gamma_down(self):
        img_gad = exposure.adjust_gamma(self.img, gamma=1.5, gain=1)

        self.img = img_gad

    '''
    def my_exposure(self):
        v_min, v_max = np.percentile(self.img, (0, 95))
        img_exp = exposure.rescale_intensity(self.img, in_range=(v_min, v_max))
    
        self.img = img_exp
    '''

    def crop_img_and_masks_for_output(self):
        width_original = self.y_size
        height_original = self.x_size

        width_current = self.img.shape[0]
        height_current = self.img.shape[1]

        ''' crops the same on both sides y dim '''
        crop_l = int((width_current - width_original) / 2)
        crop_r = width_current - width_original - crop_l

        ''' crops the same on both sides x dim '''
        crop_u = int((height_current - height_original) / 2)
        crop_d = height_current - height_original - crop_u

        if self.is_rgb == True:
            img_trimmed = img_as_ubyte(crop(self.img, ((crop_l, crop_r), (crop_u, crop_d), (0,0)), copy=False))
        else:
            img_trimmed = img_as_ubyte(crop(self.img, ((crop_l, crop_r), (crop_u, crop_d)), copy=False))
        masks_trimmed = [img_as_ubyte(crop(mask, ((crop_l, crop_r), (crop_u, crop_d)), copy=False)) for mask in self.masks]

        self.img = img_trimmed
        self.masks = masks_trimmed

    def show_current_img_and_masks(self):
        io.imshow(self.img)
        plt.show()
        for mask in self.masks:
            io.imshow(mask)
            plt.show()

    def get_current_img_and_masks(self):
        img = img_as_ubyte(self.img)
        masks = img_as_ubyte(self.masks)

        return img, masks


class MyLiveAugmentations:
    def __init__(self, img_np, masks_np):
        self.img = img_as_ubyte(img_np) # convert from normalised float
        self.masks = [img_as_ubyte(mask_np) for mask_np in masks_np] # convert from normalised float
        self.y_size = self.img.shape[0]
        self.x_size = self.img.shape[1]
        if len(self.img.shape) == 3:
            self.is_rgb = True
        else:
            self.is_rgb = False

    def my_rotate(self):
        ''' angles for rotation '''
        rot_range = [[10, 21], [350, 361]]
        range_choice = random.choice(rot_range)
        degree = random.randrange(range_choice[0], range_choice[1])

        img_rot = rotate(self.img, angle=degree, mode='constant')
        masks_rot = [rotate(mask, angle=degree, mode='constant') for mask in self.masks]

        self.img = img_rot
        self.masks = masks_rot

    def my_shift(self):
        ''' only left and right shifting '''
        switch = [-1, 1]
        ''' only downwards shifting '''
        # y = random.randrange(10) * -1
        y = 0
        x = random.randrange(20) * random.choice(switch)

        shift = AffineTransform(translation=(x, y))
        img_shi = warp(self.img, shift, mode='constant')
        masks_shi = [warp(mask, shift, mode='constant') for mask in self.masks]

        self.img = img_shi
        self.masks = masks_shi

    def my_zoom_in(self):
        ''' downscales the canvas '''
        scale = 0.9
        ''' only if width and height are the same '''
        size = self.y_size
        ''' canvas upscaling increases range, image then has to be centered'''
        move_range = int(((scale - 1) * size) / 2)

        ''' all movement is towards the right since the image is not centered '''
        y = move_range * -1
        x = move_range * -1

        shift_scale = AffineTransform(scale=(scale, scale), translation=(x, y))

        img_zin = warp(self.img, shift_scale, mode='constant')
        masks_zin = [warp(mask, shift_scale, mode='constant') for mask in self.masks]

        self.img = img_zin
        self.masks = masks_zin

    def my_zoom_out(self):
        ''' upscales the canvas '''
        scale = 1.1
        ''' only if width and height are the same '''
        size = self.y_size
        ''' canvas upscaling increases range, image then has to be centered'''
        move_range = int(((scale - 1) * size) / 2)

        ''' all movement is towards the right since the image is not centered '''
        y = move_range * -1
        x = move_range * -1

        shift_scale = AffineTransform(scale=(scale, scale), translation=(x, y))

        img_zou = warp(self.img, shift_scale, mode='constant')
        masks_zou = [warp(mask, shift_scale, mode='constant') for mask in self.masks]

        self.img = img_zou
        self.masks = masks_zou

    def my_x_warp_in(self):
        ''' only if width and height are the same '''
        size = self.y_size
        max_scale = 1.1
        x_warp = int(size * max_scale)

        img_wxo = resize(self.img, (size, x_warp), anti_aliasing=True)
        masks_wxo = [resize(mask, (size, x_warp), anti_aliasing=True) for mask in self.masks]

        self.img = img_wxo
        self.masks = masks_wxo

    def my_x_warp_out(self):
        ''' scales up canvas '''
        scale = 1.1
        ''' only if width and height are the same '''
        size = self.y_size
        ''' centers the image by dividing the upscale by 2 '''
        center = int(((scale - 1) / 2) * size)

        shift_warp = AffineTransform(scale=(scale, 1), translation=(-center, 0))

        img_wxi = warp(self.img, shift_warp, mode='constant')
        masks_wxi = [warp(mask, shift_warp, mode='constant') for mask in self.masks]

        self.img = img_wxi
        self.masks = masks_wxi

    def my_y_warp_in(self):
        ''' only if width and height are the same '''
        size = self.y_size
        max_scale = 1.1
        y_warp = int(size * max_scale)

        img_wyi = resize(self.img, (y_warp, size), anti_aliasing=True)
        masks_wyi = [resize(mask, (y_warp, size), anti_aliasing=True) for mask in self.masks]

        self.img = img_wyi
        self.masks = masks_wyi

    def my_noise(self):
        sigma = 0.1
        img_noi = random_noise(self.img, var=sigma ** 2)

        self.img = img_noi

    def my_blur(self):
        sigma = 1
        img_blu = gaussian(self.img, sigma=sigma, multichannel=True)

        self.img = img_blu

    def my_gamma_up(self):
        img_gau = exposure.adjust_gamma(self.img, gamma=0.75, gain=1)

        self.img = img_gau

    def my_gamma_down(self):
        img_gad = exposure.adjust_gamma(self.img, gamma=1.5, gain=1)

        self.img = img_gad

    '''
    def my_exposure(self):
        v_min, v_max = np.percentile(self.img, (0, 95))
        img_exp = exposure.rescale_intensity(self.img, in_range=(v_min, v_max))

        self.img = img_exp
    '''

    def crop_img_and_masks_for_output(self):
        width_original = self.y_size
        height_original = self.x_size

        width_current = self.img.shape[0]
        height_current = self.img.shape[1]

        ''' crops the same on both sides y dim '''
        crop_l = int((width_current - width_original) / 2)
        crop_r = width_current - width_original - crop_l

        ''' crops the same on both sides x dim '''
        crop_u = int((height_current - height_original) / 2)
        crop_d = height_current - height_original - crop_u

        if self.is_rgb == True:
            img_trimmed = img_as_ubyte(crop(self.img, ((crop_l, crop_r), (crop_u, crop_d), (0, 0)), copy=False))
        else:
            img_trimmed = img_as_ubyte(crop(self.img, ((crop_l, crop_r), (crop_u, crop_d)), copy=False))
        masks_trimmed = [img_as_ubyte(crop(mask, ((crop_l, crop_r), (crop_u, crop_d)), copy=False)) for mask in
                         self.masks]

        self.img = img_trimmed
        self.masks = masks_trimmed

    def show_current_img_and_masks(self):
        io.imshow(self.img)
        plt.show()
        for mask in self.masks:
            io.imshow(mask)
            plt.show()

    def get_current_img_and_masks(self):
        img = img_as_ubyte(self.img)
        masks = img_as_ubyte(self.masks)

        return img, masks


def create_single_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output):

    imgs_dir_path = path.join(datasets_dir, 'imgs', dataset_name)
    masks_dir_path = path.join(datasets_dir, 'masks', dataset_name)

    imgs_aug = path.join(augmentations_dir_output, 'imgs', f'{dataset_name}_A{n_augmention_copies}')
    masks_aug = path.join(augmentations_dir_output, 'masks', f'{dataset_name}_A{n_augmention_copies}')
    os.mkdir(imgs_aug)
    os.mkdir(masks_aug)

    input_files = os.listdir(imgs_dir_path)

    with tqdm(total=len(input_files)*n_augmention_copies, desc='Total augmentations', unit='imgs and masks', leave=False) as pbar:

        for i in input_files:
            file_id = i.rsplit('.', 1)[0]
            img_path = path.join(imgs_dir_path, i)
            masks_paths = glob(path.join(masks_dir_path, file_id) + '*')
            masks_exts = [path.basename(mask_path).rsplit('_', 1)[-1] for mask_path in masks_paths]

            used_augmentations = []

            for x in range(1, n_augmention_copies + 1):
                ''' generate which augmentations should be used '''

                random_aug = random.randrange(0, 10, 1)
                while random_aug in used_augmentations:
                    random_aug = random.randrange(0, 10, 1)
                used_augmentations.append(random_aug)

                current_augmentations = MyAugmentations(img_path, masks_paths)

                if random_aug == 0:
                    current_augmentations.my_zoom_out()

                if random_aug == 1:
                    current_augmentations.my_x_warp_in()

                if random_aug == 2:
                    current_augmentations.my_x_warp_out()

                if random_aug == 3:
                    current_augmentations.my_blur()

                if random_aug == 4:
                    current_augmentations.my_rotate()

                if random_aug == 5:
                    current_augmentations.my_shift()

                if random_aug == 6:
                    current_augmentations.my_zoom_in()

                if random_aug == 7:
                    current_augmentations.my_gamma_down()

                if random_aug == 8:
                    current_augmentations.my_gamma_up()

                if random_aug == 9:
                    current_augmentations.my_noise()

                ''' crops images and masks to same size after transforms '''
                current_augmentations.crop_img_and_masks_for_output()

                ''' shows augmented images without saving '''
                #current_augmentations.show_current_img_and_masks()

                img_augmented, masks_augmented = current_augmentations.get_current_img_and_masks()

                if x < 10:
                    img_save_path = path.join(imgs_aug, f'{file_id}_A0{x}.png')
                    masks_save_paths = [path.join(masks_aug, f'{file_id}_A0{x}_{mask_ext}') for mask_ext in masks_exts]
                else:
                    img_save_path = path.join(imgs_aug, f'{file_id}_A{x}.png')
                    masks_save_paths = [path.join(masks_aug, f'{file_id}_A{x}_{mask_ext}') for mask_ext in masks_exts]

                io.imsave(img_save_path, img_augmented)
                for pm in zip(masks_save_paths, masks_augmented):
                    io.imsave(pm[0], pm[1])

                pbar.update()


def create_combined_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output):

    imgs_dir_path = path.join(datasets_dir, 'imgs', dataset_name)
    masks_dir_path = path.join(datasets_dir, 'masks', dataset_name)

    imgs_aug = path.join(augmentations_dir_output, 'imgs', f'{dataset_name}_MA{n_augmention_copies}')
    masks_aug = path.join(augmentations_dir_output, 'masks', f'{dataset_name}_MA{n_augmention_copies}')
    os.mkdir(imgs_aug)
    os.mkdir(masks_aug)

    min_aug = 4
    max_aug = 9

    input_files = os.listdir(imgs_dir_path)

    with tqdm(total=len(input_files)*n_augmention_copies, desc='Total augmentations', unit='imgs and masks', leave=False) as pbar:

        for i in input_files:
            file_id = i.rsplit('.', 1)[0]
            img_path = path.join(imgs_dir_path, i)
            masks_paths = glob(path.join(masks_dir_path, file_id) + '*')
            masks_exts = [path.basename(mask_path).rsplit('_', 1)[-1] for mask_path in masks_paths]

            for x in range(1, n_augmention_copies + 1):

                aug_times = random.randrange(min_aug, max_aug + 1)  # augmentations per image augmentation

                ''' generate which augmentations should be used '''
                choice_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for y in range(aug_times):
                    random_aug = random.randrange(0, len(choice_list))
                    while choice_list[random_aug] == 1:
                        random_aug = random.randrange(0, len(choice_list))
                    choice_list[random_aug] = 1

                current_augmentations = MyAugmentations(img_path, masks_paths)

                if choice_list[0] == 1:
                    current_augmentations.my_zoom_out()

                if choice_list[1] == 1:
                    current_augmentations.my_x_warp_in()

                if choice_list[2] == 1:
                    current_augmentations.my_x_warp_out()

                if choice_list[3] == 1:
                    current_augmentations.my_blur()

                if choice_list[4] == 1:
                    current_augmentations.my_rotate()

                if choice_list[5] == 1:
                    current_augmentations.my_shift()

                if choice_list[6] == 1:
                    current_augmentations.my_zoom_in()

                if choice_list[7] == 1:
                    current_augmentations.my_gamma_down()

                if choice_list[8] == 1:
                    current_augmentations.my_gamma_up()

                if choice_list[9] == 1:
                    current_augmentations.my_noise()

                ''' crops images and masks to same size after transforms '''
                current_augmentations.crop_img_and_masks_for_output()

                ''' shows augmented images without saving '''
                #current_augmentations.show_current_img_and_masks()

                img_augmented, masks_augmented = current_augmentations.get_current_img_and_masks()

                if x < 10:
                    img_save_path = path.join(imgs_aug, f'{file_id}_MA0{x}.png')
                    masks_save_paths = [path.join(masks_aug, f'{file_id}_MA0{x}_{mask_ext}') for mask_ext in masks_exts]
                else:
                    img_save_path = path.join(imgs_aug, f'{file_id}_MA{x}.png')
                    masks_save_paths = [path.join(masks_aug, f'{file_id}_MA{x}_{mask_ext}') for mask_ext in masks_exts]

                io.imsave(img_save_path, img_augmented)
                for pm in zip(masks_save_paths, masks_augmented):
                    io.imsave(pm[0], pm[1])

                pbar.update()


''' code below is deprecated '''








def my_rotate(img, mask):
    
    rotRange = [[10, 21], [350, 361]]
    rangeChoice = random.choice(rotRange)
    degree = random.randrange(rangeChoice[0], rangeChoice[1])
    
    imgrot = rotate(img, angle=degree, mode='constant')
    maskrot = rotate(mask, angle=degree, mode='constant')
    
    return imgrot, maskrot


def my_shift(img, mask):

    switch = [-1, 1] #venstre og høyre
    #y = random.randrange(10) * -1 #bare nedover
    y = 0
    x = random.randrange(20) * random.choice(switch)
    
    shift = AffineTransform(translation=(x, y))
    imgshi = warp(img, shift, mode='constant')    
    maskshi = warp(mask, shift, mode='constant')
  
    return imgshi, maskshi


def my_zoom_in(img, mask):
    
    scale = 0.9 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgzoo = warp(img, shiftscale, mode='constant')    
    maskzoo = warp(mask, shiftscale, mode='constant')
    
    return imgzoo, maskzoo


def my_zoom_out(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    moveRange = int(((scale - 1) * size) / 2) #økt range av forstørret canvas. senterer bildet
    
    y = moveRange * -1 #all bevegelse er mot høyre fordi bildet ikke er sentrert 
    x = moveRange * -1
    
    shiftscale = AffineTransform(scale=(scale, scale), translation=(x, y))
    
    imgshi = warp(img, shiftscale, mode='constant')    
    maskshi = warp(mask, shiftscale, mode='constant')
    
    return imgshi, maskshi


def my_x_warp_in(img, mask):
    
    size = img.shape[0]
    maxScale = 1.1
    xWarp = int(size * maxScale)

    lCrop = int((xWarp - size) / 2) #cropper venstre
    rCrop = xWarp - size - lCrop #cropper høyre

    imgwarx = resize(img, (size, xWarp), anti_aliasing=True)
    maskwarx = resize(mask, (size, xWarp), anti_aliasing=True)
    
    return imgwarx, maskwarx


def my_x_warp_out(img, mask):
    
    scale = 1.1 #skalerer opp canvas
    size = img.shape[0]
    center = int(((scale - 1) / 2) * size) #senter av bildet ved å dele skalering på 2
    
    shiftwarp = AffineTransform(scale=(scale, 1), translation=(-center, 0))
    
    imgshi = warp(img, shiftwarp, mode='constant')
    maskshi = warp(mask, shiftwarp, mode='constant')
  
    return imgshi, maskshi


def my_y_warp_in(img, mask):
    
    size = img.shape[0]
    maxscale = 1.1
    yWarp = int(size * maxscale)

    yCrop = yWarp - size #cropper nedre del

    imgwary = resize(img, (yWarp, size), anti_aliasing=True)
    maskwary = resize(mask, (yWarp, size), anti_aliasing=True)
    
    return imgwary, maskwary


def my_noise(img, masknoi):

    sigma = 0.1
    imgnoi = random_noise(img, var=sigma**2)
    
    return imgnoi, masknoi


def my_blur(img, maskblu):
        
    sigma = 1
    imgblu = gaussian(img, sigma=sigma, multichannel=True)
    
    return imgblu, maskblu


def my_gamma_up(img, maskgau):

    imggau = exposure.adjust_gamma(img, gamma=0.75,gain=1)
    
    return imggau, maskgau


def my_gamma_down(img, maskgad):

    imggad = exposure.adjust_gamma(img, gamma=1.5,gain=1)
    
    return imggad, maskgad


def my_exposure(img, maskexp):

    v_min, v_max = np.percentile(img, (0, 95))
    imgexp = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    
    return imgexp, maskexp


def my_base(imgname, imgpath, maskname, maskpath):

    img = io.imread(path.join(imgpath, imgname))
    mask = io.imread(path.join(maskpath, maskname))
    
    return img, mask
    
''' deprecated '''
def create_augmentations_deprecated(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output):

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

            temp = my_base(i, imgs_path, m, masks_path) #åpner bilde. output er 2 bilder, img=0 og mask=1

            choice_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            aug_times = random.randrange(min_aug - 1, max_aug) #antall ganger hvert bilde skal augumenteres

            for y in range(aug_times):
                random_aug = random.randrange(0, len(choice_list))
                while choice_list[random_aug] == 1:
                    random_aug = random.randrange(0, len(choice_list))
                choice_list[random_aug] = 1

            if choice_list[0] == 1:
                temp = my_zoom_out(temp[0], temp[1])

            if choice_list[1] == 1:
                temp = my_x_warp_in(temp[0], temp[1])

            if choice_list[2] == 1:
                temp = my_x_warp_out(temp[0], temp[1])

            if choice_list[3] == 1:
                temp = my_blur(temp[0], temp[1])

            if choice_list[4] == 1:
                temp = my_rotate(temp[0], temp[1])

            if choice_list[5] == 1:
                temp = my_shift(temp[0], temp[1])

            if choice_list[6] == 1:
                temp = my_zoom_in(temp[0], temp[1])

            if choice_list[7] == 1:
                temp = my_gamma_down(temp[0], temp[1])

            if choice_list[8] == 1:
                temp = my_gamma_up(temp[0], temp[1])

            if choice_list[9] == 1:
                temp = my_noise(temp[0], temp[1])

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

    create_combined_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output)
    
