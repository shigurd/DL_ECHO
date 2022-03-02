import sys
import os
import os.path as path

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LV_mask_to_AFI_roi_unlimited_points import generate_AFIRoi


if __name__ == "__main__":
    prediction_masks = 'Feb05_21-52-38_EFFIB1-LR5-DICBCE_AL_ADAM_T-CAMUS1800_HML_V-NONE_EP30_LR0.001_BS20_SCL1_OUT'

    keyfile_csv = r'H:\ML_LV\backup_keyfiles\keyfile_GE1965_QC_CRIDP.csv'
    input_dcm_dir = r'H:\ML_LV\backup_dcm\GE2023_dcm'
    input_masks_dir = path.join('predictions', 'strain', prediction_masks)

    output_dcm_original_dir = path.join('predictions_dicom', prediction_masks, 'original')
    output_dcm_predicted_dir = path.join('predictions_dicom', prediction_masks, 'predicted')
    os.mkdir(path.join('predictions_dicom', prediction_masks))
    os.mkdir(output_dcm_original_dir)
    os.mkdir(output_dcm_predicted_dir)

    generate_AFIRoi(input_masks_dir, input_dcm_dir, keyfile_csv, output_dcm_original_dir, output_dcm_predicted_dir)

