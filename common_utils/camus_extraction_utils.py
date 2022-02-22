import os
import os.path as path
from PIL import Image
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def extract_configs(txt_file):
    file_name = os.path.basename(txt_file).rsplit('.')[0]
    projection = file_name.split('_')[-1]

    dict = {}
    dict['Projection'] = projection

    with open(txt_file) as txt:
        lines = txt.readlines()
        for l in lines:
            l = l.replace(' ', '').strip('\n')
            key, value = l.rsplit(':')
            dict[key] = value
    txt.close()

    return dict


def convert_raw_mhd_to_pil(raw_pth):
    itkimage_img = sitk.ReadImage(raw_pth)
    img = sitk.GetArrayFromImage(itkimage_img)
    img = np.squeeze(img)
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((256, 256))

    return img


def get_img_quality(loaded_cfg_dict):
    tag = 'NONE'
    for key in loaded_cfg_dict:
        if key == 'ImageQuality':
            quality = loaded_cfg_dict[key]
            if quality == 'Good':
                tag = 'HIG'
            if quality == 'Medium':
                tag = 'MED'
            if quality == 'Poor':
                tag = 'LOW'
            break

    return tag


def extract_img_from_camus_mhd(input_dir, imgs_output, masks_output):

    patient_folders = os.listdir(input_dir)
    with tqdm(total=len(patient_folders), desc='patients extracted', unit='patients', leave=False) as pbar:

        for patient_folder in patient_folders:
            img_info = []
            info_files = glob(path.join(input_dir, patient_folder) + '\*.cfg')
            ''' extract cfg info '''
            for cfg in info_files:
                img_info.append(extract_configs(cfg))

            ''' find all .raw files '''
            raw_files = glob(path.join(input_dir, patient_folder) + '\*.mhd')

            for raw_path in raw_files:
                file_name, ext = os.path.basename(raw_path).rsplit(".", 1)
                file_tags = file_name.split('_')
                patient_tag = file_tags[0]
                view_tag = file_tags[1]
                timing_tag = file_tags[2]

                if file_tags[-1] != 'sequence':
                    img_pil = convert_raw_mhd_to_pil(raw_path)

                    ''' get view quality tag '''
                    for d in img_info:
                        if d['Projection'] == view_tag:
                            quality_tag = get_img_quality(d)

                    ''' process gt mask '''
                    if file_tags[-1] == 'gt':
                        mask_np = np.array(img_pil)
                        mask_np = mask_np / 3 * 255 #greyscale tricolor mask: 1 endocard, 2 myocard, 3 atrium
                        mask_pil = Image.fromarray(mask_np.astype(np.uint8))

                        mask_name = f'{patient_tag}_{view_tag}_{timing_tag}_{quality_tag}_mask.png'
                        mask_pil.save(os.path.join(masks_output, mask_name))
                    else:
                        img_name = f'{patient_tag}_{view_tag}_{timing_tag}_{quality_tag}.png'
                        img_pil.save(os.path.join(imgs_output, img_name))

            pbar.update()

if __name__ == '__main__':
    input_dir = r'C:\Users\Brekke\Downloads\training'
    output_dir_imgs = r'C:\Users\Brekke\Downloads\imgs'
    output_dir_masks = r'C:\Users\Brekke\Downloads\masks'

    os.mkdir(output_dir_imgs)
    os.mkdir(output_dir_masks)

    extract_img_from_camus_mhd(input_dir, output_dir_imgs, output_dir_masks)