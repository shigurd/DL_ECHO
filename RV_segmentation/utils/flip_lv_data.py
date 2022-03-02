from PIL import Image
import numpy as np
import os
import sys
sys.path.insert(0, '..')
np.set_printoptions(threshold=sys.maxsize)


def flip_image_horizontal_np(img_pth):
    img_pil = Image.open(img_pth)
    img_np = np.array(img_pil)

    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)

    img_np_flip = img_np[:, ::-1, :]

    if img_np_flip.shape[-1] == 1:
        img_np_flip = np.squeeze(img_np_flip, axis=2)
        img_np_flip = img_np_flip.astype(np.float32) * 255

    return img_np_flip.astype(np.uint8)


def make_flipped_files(img_dir):
    output_dir = os.path.join(os.path.dirname(img_dir), f'{os.path.basename(img_dir)}_flipped')
    os.mkdir(output_dir)

    for img in os.listdir(img_dir):
        img_pth = os.path.join(img_dir, img)

        img_flipped_np = flip_image_horizontal_np(img_pth)

        img_flipped_pil = Image.fromarray(img_flipped_np)
        img_flipped_pil.save(os.path.join(output_dir, img))



if __name__ == '__main__':
    imgs_dir = 'D:\DL_ECHO\RV_segmentation\datasets\imgs\LV715'
    masks_dir = 'D:\DL_ECHO\RV_segmentation\datasets\masks\LV715'

    make_flipped_files(imgs_dir)
    make_flipped_files(masks_dir)


