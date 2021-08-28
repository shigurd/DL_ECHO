"""This scripts generate Masks from AFIRoi Images """
import sys
import os
import os.path as path
import csv

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LV_AFI_roi_to_mask import iterate_folders

if __name__ == "__main__":
    ''' extract LV images and masks and log data into a keyfile '''
    keyfile_log = 'keyfile_GE1956.csv'

    ''' define input dir '''
    input_dcm_dir = r'H:\ML_LV\backup_dcm\crawler_data\ML_LVROI_Angio18'
    output_dir_imgs = path.join('datasets', 'imgs', 'GE1956')
    output_dir_masks = path.join('datasets', 'masks', 'GE1956')
    os.mkdir(output_dir_imgs)
    os.mkdir(output_dir_masks)

    csv_file = open(keyfile_log, 'w', newline='')
    writer = csv.writer(csv_file)

    iterate_folders(input_dcm_dir, output_dir_imgs, output_dir_masks, writer)

