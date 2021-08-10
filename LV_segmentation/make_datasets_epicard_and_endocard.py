from utils.convert_myomask_to_endomask_and_epimask_LV import get_endocard_epicard
import os
import os.path as path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':

    dataset = 'CAMUS1800_HML'
    imgs_dir = path.join('datasets', 'imgs', dataset)
    masks_dir = path.join('datasets', 'masks', dataset)

    epicard_masks_output = path.join(path.dirname(masks_dir), f'{dataset}EPI')
    endocard_masks_output = path.join(path.dirname(masks_dir), f'{dataset}END')
    epicard_imgs_output = path.join(path.dirname(imgs_dir), f'{dataset}EPI')
    endocard_imgs_output = path.join(path.dirname(imgs_dir), f'{dataset}END')

    os.mkdir(epicard_masks_output)
    os.mkdir(endocard_masks_output)
    os.mkdir(epicard_imgs_output)
    os.mkdir(endocard_imgs_output)

    with tqdm(total=len(os.listdir(masks_dir)), desc='Creating epicard and endocard datasets', unit='imgs', leave=False) as pbar:

        for m in os.listdir(masks_dir):
            mask_path = path.join(masks_dir, m)
            i = f'{m.rsplit("_", 1)[0]}.png'
            img_path = path.join(imgs_dir, i)

            int_roi_filled, ext_roi_filled = get_endocard_epicard(mask_path)

            ext_roi_filled_pil = Image.fromarray(ext_roi_filled.astype(np.uint8))
            int_roi_filled_pil = Image.fromarray(int_roi_filled.astype(np.uint8))

            ext_roi_filled_pil.save(path.join(epicard_masks_output, m))
            int_roi_filled_pil.save(path.join(endocard_masks_output, m))

            shutil.copyfile(img_path, path.join(epicard_imgs_output, i))
            shutil.copyfile(img_path, path.join(endocard_imgs_output, i))

            pbar.update()