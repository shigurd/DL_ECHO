"""This scripts generate Masks from AFIRoi Images """
import os
import os.path as path
import sys
import csv

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LVOT_coords import iterate_folders

if __name__ == "__main__":
    ''' extract LVOT images and LVOT coordinates and LVOTd data into a keyfile '''
    keyfile_name = 'keyfile_GE1424.csv'

    '''define input dir and parameters for mask '''
    input_dcm_dir = r'H:\ML_LVOT\dcm_lvot_backup\GE1424_lvot_dcm'
    height = 256
    width = 256
    with_gaussian = True
    x_warp = 5
    y_warp = 1

    if with_gaussian == True:
        dataset_name = f'GE1424X{x_warp}_HMLHML'
    else:
        dataset_name = f'GE1424_HMLHML'

    keyfile_output = path.join(r'H:\ML_LVOT\txt_keyfile_and_duplicate', keyfile_name)
    output_imgs_dir = path.join('datasets', 'imgs', dataset_name)
    output_masks_dir = path.join('datasets', 'masks', dataset_name)
    os.mkdir(output_imgs_dir)
    os.mkdir(output_masks_dir)

    keyfile_log = open(keyfile_output, 'w', newline='')
    writer = csv.writer(keyfile_log)
    writer.writerow(['dicom_id', 'exam_id', 'x_superior_pixel', 'y_superior_pixel', 'x_inferior_pixel',
                 'y_inferior_pixel', 'x_superior_cm', 'y_superior_cm', 'x_inferior_cm',
                 'y_inferior_cm', 'lvot_diameter_cm', 'SCparams'])

    iterate_folders(input_dcm_dir, output_imgs_dir, output_masks_dir, height, width, with_gaussian, x_warp, y_warp, writer)
    keyfile_log.close()
