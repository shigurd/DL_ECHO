import sys
import os
import os.path as path

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LV_mask_to_AFI_roi_unlimited_points import generate_AFIRoi


if __name__ == "__main__":
    prediction_masks = 'Mar17_20-37-03_RES50UNETIMGN_DICEBCE_ADAM_LR5_AL_T-RV141_HMLHMLLVFHALF_V-NONE_EP30_LR0.0005_BS16_OUT'

    keyfile_csv = r'H:\RV_strain\keyfile_RV141\RV141_keyfile.csv'
    input_dcm_dir = r'H:\ML_LV\backup_dcm\GE2023_crawler_output_processing\ML_LVROI_Basedata_unfiltered\ML-ROI_data_14_02_20_ECHOPAC_ufiltrert_rv_sax_etc'
    input_masks_dir = path.join('predictions', 'abstract_to_dcm', prediction_masks)

    output_dcm_original_dir = path.join('predictions_dicom', prediction_masks, 'raw')
    output_dcm_predicted_dir = path.join('predictions_dicom', prediction_masks, 'dcm')
    os.mkdir(path.join('predictions_dicom', prediction_masks))
    os.mkdir(output_dcm_original_dir)
    os.mkdir(output_dcm_predicted_dir)

    generate_AFIRoi(input_masks_dir, input_dcm_dir, keyfile_csv, output_dcm_original_dir, output_dcm_predicted_dir)

