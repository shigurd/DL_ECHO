from os.path import splitext
import os.path as path
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = img_scale
        assert 0 < img_scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        ''' expand dim if image is 1 channel '''
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        ''' HWC to CHW '''
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    ''' used with hardcoded 3 in_channel torchvision resnet50 '''
    @classmethod
    def convert_to_3ch(cls, np_img, mid_systole):

        if np_img.shape[0] == 1:
            np_img = np.stack([np_img, np_img, np_img], axis=0)
        else:
            if mid_systole == True:
                np_img = np_img[1, :, :] # 0 frame before, 1 midsystole, 2 frame after
                np_img = np.stack([np_img, np_img, np_img], axis=0)

        return np_img

    ''' for coordinate convolution '''
    @classmethod
    def add_cc_channel(cls, np_img):

        x_size = np_img.shape[-1]
        y_size = np_img.shape[-2]

        cc_x = np.zeros((y_size, x_size))
        cc_y = np.zeros((y_size, x_size))

        for p in range(y_size):
            cc_y[p, :] = (p + 1) / y_size

        for j in range(x_size):
            cc_x[:, j] = (j + 1) / x_size

        cc_y = np.expand_dims(cc_y, axis=0)
        cc_x = np.expand_dims(cc_x, axis=0)
        img = np.concatenate((np_img, cc_y, cc_x), axis=0)
        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(path.join(self.masks_dir, idx) + '*')
        img_file = glob(path.join(self.imgs_dir, idx) + '*')

        assert len(mask_file) == 2, \
            f'Either only one mask found or more than 2 masks found {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask_i = Image.open(mask_file[0])
        mask_i = mask_i.convert('L')
        mask_s = Image.open(mask_file[1])
        mask_s = mask_s.convert('L')
        img = Image.open(img_file[0])
        #img = img.convert('RGB')

        assert img.size == mask_i.size and img.size == mask_s.size, \
            f'Image {idx} and mask_i and mask_s should be the same size, but are img: {img.size} and mask_i: {mask_i.size} and mask_s: {mask_s.size}'

        img = self.preprocess(img, self.scale)
        mask_i = self.preprocess(mask_i, self.scale)
        mask_s = self.preprocess(mask_s, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask_i': torch.from_numpy(mask_i).type(torch.FloatTensor),
            'mask_s': torch.from_numpy(mask_s).type(torch.FloatTensor)
        }

