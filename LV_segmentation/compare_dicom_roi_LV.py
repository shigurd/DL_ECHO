import sys
import os
import os.path as path
from PIL import Image

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LV_AFI_roi_to_mask import get_mask_from_dicom
from predict_LV import concat_img, pil_overlay_predicted_and_gt, pil_overlay

if __name__ == "__main__":

    prediction_name = 'Dec04_22-38-28_EFFIB0-DICBCE_AL_TF-GEHMLHML_ADAM_T-CAMUS1800_HM_V-NONE_TRANSFER-EP150+150_LR0.001_BS20_SCL1_OUT'
    original_dcm = path.join('predictions_dicom', prediction_name, 'original')
    predicted_dcm = path.join('predictions_dicom',prediction_name, 'predicted')

    for fn in os.listdir(original_dcm):
        pth_org = path.join(original_dcm, fn)
        pth_pred = path.join(predicted_dcm, fn)

        mask_org, img_org, exam_org = get_mask_from_dicom(pth_org)
        mask_pred, img_pred, exam_pred = get_mask_from_dicom(pth_pred)

        overlay_mask = pil_overlay_predicted_and_gt(mask_org, mask_pred)
        joint_overlay = pil_overlay(overlay_mask, img_org, alpha=0.2)

        joint_overlay.show()

        #print(exam_org, exam_pred)
        #mask_org.show()
        #mask_pred.show()
        #img_org.show()
        #img_pred.show()



