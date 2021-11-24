import numpy as np
import torch
import random
from PIL import Image

from data_partition_utils.create_augmentation_imgs_and_masks import MyLiveAugmentations


def preprocess(pil_img, scale=1):
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


def augment_imgs_masks_batch(imgs_torch_batch, masks_torch_batch):
    ''' augments images and masks randomly for every batch '''
    imgs_temp= []
    masks_temp = []

    for img_mask_torch in zip(imgs_torch_batch, masks_torch_batch): # zips files according to order
        img_np = img_mask_torch[0].numpy()
        img_np = img_np.transpose((1, 2, 0)).squeeze()  # reconvert to np img format from dataloader
        masks_np = [mask.numpy() for mask in img_mask_torch[1]] # in case of multiple masks, masks are always greyscale

        temp = MyLiveAugmentations(img_np, masks_np)
        augmentation_choice = random.randrange(0, 11)

        if augmentation_choice == 0:
            temp.my_zoom_out()

        if augmentation_choice == 1:
            temp.my_x_warp_in()

        if augmentation_choice == 2:
            temp.my_x_warp_out()

        if augmentation_choice == 3:
            temp.my_blur()

        if augmentation_choice == 4:
            temp.my_rotate()

        if augmentation_choice == 5:
            temp.my_shift()

        if augmentation_choice == 6:
            temp.my_zoom_in()

        if augmentation_choice == 7:
            temp.my_gamma_down()

        if augmentation_choice == 8:
            temp.my_gamma_up()

        if augmentation_choice == 9:
            temp.my_noise()

        if augmentation_choice == 10:
            ''' no augmentation '''
            pass

        ''' crops images and masks to same size after transforms '''
        temp.crop_img_and_masks_for_output()

        img_augmented_np, masks_augmented_np = temp.get_current_img_and_masks()
        img_augmented_np = preprocess(Image.fromarray(img_augmented_np), scale=1) # reprocessing for dataloader
        masks_augmented_np = [preprocess(Image.fromarray(mask_augmented_np), scale=1) for mask_augmented_np in masks_augmented_np] # reprocessing for dataloader

        imgs_temp.append(img_augmented_np)
        masks_temp.append(np.concatenate(masks_augmented_np, axis=0))

    imgs_augmented_stack = np.stack(imgs_temp, axis=0)
    masks_augmented_stack = np.stack(masks_temp, axis=0)

    return torch.from_numpy(imgs_augmented_stack).type(torch.FloatTensor), torch.from_numpy(masks_augmented_stack).type(torch.FloatTensor)