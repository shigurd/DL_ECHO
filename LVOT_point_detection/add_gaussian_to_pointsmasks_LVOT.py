import numpy as np
from utils.dataloader_LVOT import calculate_angle, create_gaussian_on_mask
from PIL import Image
import os
import shutil
from glob import glob


def add_gaussian_to_masks(imask_pil, smask_pil):
    imask_np = np.array(imask_pil)
    smask_np = np.array(smask_pil)

    i_y, i_x = np.where(imask_np == 255)
    s_y, s_x = np.where(smask_np == 255)

    i_angle, s_angle = calculate_angle(i_x[0], i_y[0], s_x[0], s_y[0])

    imask_np = create_gaussian_on_mask(imask_np, i_x, i_y, i_angle, x_warp=5, y_warp=1, sigma=3)
    smask_np = create_gaussian_on_mask(smask_np, s_x, s_y, s_angle, x_warp=5, y_warp=1, sigma=3)

    temp_i = Image.fromarray(imask_np.astype(np.uint8)).convert('L')
    temp_s = Image.fromarray(smask_np.astype(np.uint8)).convert('L')

    return temp_i, temp_s

if __name__ == '__main__':
    data_names = ['GE1408_HMHMAVA_K1', 'GE1408_HMHMAVA_K2', 'GE1408_HMHMAVA_K3', 'GE1408_HMHMAVA_K4', 'GE1408_HMHMAVA_K5']
    #data_names = ['GE1423_HMLHMLAVA']
    is_kfold = True
    data_name_new = 'HMHMAVAGAUS3'

    imgs_dir = 'data/train/imgs'
    masks_dir = 'data/train/masks'

    for n in data_names:
        if is_kfold == True:
            dataset_name, data_quality, n_k = n.rsplit('_', 2)
            new_name = f'{dataset_name}_{data_name_new}_{n_k}'
        else:
            dataset_name, data_quality = n.rsplit('_', 1)
            new_name = f'{dataset_name}_{data_name_new}'

        imgs_out_dir = os.path.join(imgs_dir, new_name)
        masks_out_dir = os.path.join(masks_dir, new_name)
        os.mkdir(imgs_out_dir)
        os.mkdir(masks_out_dir)

        org_dir_img = os.path.join(imgs_dir, n)
        org_dir_mask = os.path.join(masks_dir, n)

        for i in os.listdir(org_dir_img):
            file_name = i.rsplit('.', 1)[0]
            mask_files = glob(os.path.join(org_dir_mask, file_name) + '*')
            assert len(mask_files) == 2

            shutil.copyfile(os.path.join(org_dir_img, i), os.path.join(imgs_out_dir, i))
            imask = Image.open(mask_files[0])
            smask = Image.open(mask_files[1])

            imask_mod, smask_mod = add_gaussian_to_masks(imask, smask)
            imask_mod.save(os.path.join(masks_out_dir, os.path.basename(mask_files[0])))
            smask_mod.save(os.path.join(masks_out_dir, os.path.basename(mask_files[1])))
