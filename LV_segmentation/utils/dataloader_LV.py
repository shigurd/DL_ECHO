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
    def __init__(self, imgs_dir, masks_dir, img_scale=1, mid_systole_only=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = img_scale
        self.mid_systole = mid_systole_only
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

    ''' used with hardcoded 3 in_channel torchvision resnet50. more or less deprecated as a result of the edited resnet50 '''
    @classmethod
    def convert_to_3ch(cls, np_img, mid_systole):

        if np_img.shape[0] == 1:
            np_img = np.stack([np_img, np_img, np_img], axis=0)
        else:
            if mid_systole == True:
                np_img = np_img[1, :, :] # 0 frame before, 1 midsystole, 2 frame after
                np_img = np.stack([np_img, np_img, np_img], axis=0)

        return np_img

    @classmethod
    def extract_midsystole(cls, np_img, mid_systole):
        if np_img.shape[0] != 1:
            if mid_systole == True:
                np_img = np_img[1, :, :]  # 0 frame before, 1 midsystole, 2 frame after
                np_img = np.expand_dims(np_img, axis=0)

        return np_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(path.join(self.masks_dir, idx) + '*')
        img_file = glob(path.join(self.imgs_dir, idx) + '*')

        assert len(mask_file) == 1, \
            f'Either no image or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        mask = mask.convert('L')
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image {idx} and mask should be the same size, but are img: {img.size} and mask: {mask.size}'

        img = self.preprocess(img, self.scale)
        img = self.extract_midsystole(img, self.mid_systole)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

