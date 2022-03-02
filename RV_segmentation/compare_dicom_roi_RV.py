import sys
import os
import os.path as path
from PIL import Image
import csv
from glob import glob

sys.path.insert(0, '..')
from dicom_extraction_utils_GE.LV_AFI_roi_to_mask import get_mask_from_dicom
from predict_RV import concat_img, pil_overlay_predicted_and_gt, pil_overlay


def get_3_masks_to_pil(keyfile_csv, predicted_masks_dir, original_dcms_dir, predicted_dcms_dir):

    ''' make output folder '''
    out_dir_pth = path.join(path.split(original_dcms_dir)[0], 'mask_comparisons')
    os.mkdir(out_dir_pth)

    ''' get final and gt masks and imgs to PIL from strain dcm with exam tag '''
    for fn in os.listdir(original_dcms_dir):
        pth_org = path.join(original_dcms_dir, fn)
        pth_pred = path.join(predicted_dcms_dir, fn)

        mask_pil_gt, img_gt, filename_gt_dcm = get_mask_from_dicom(pth_org)
        mask_pil_final, img_final, filename_final_dcm = get_mask_from_dicom(pth_pred)

        #mask_org.show()
        #mask_pred.show()
        #img_org.show()
        #img_pred.show()

        ''' get pred mask to PIL using exam tag'''
        with open(keyfile_csv, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            ''' skips header '''
            next(csv_reader, None)

            for row in csv_reader:
                dicom_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row

                if filename_gt_dcm == dicom_id:
                    mask_pth_pred = glob(path.join(predicted_masks_dir, patient_id) + '*')
                    break

        mask_pil_pred = Image.open(mask_pth_pred[0])

        ''' save individual imgs and masks '''
        img_gt.convert('RGB')
        mask_pil_gt.convert('L')
        mask_pil_pred.convert('L')
        mask_pil_final.convert('L')

        img_gt.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}.png'))
        mask_pil_gt.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}_gtmask.png'))
        mask_pil_pred.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}_predmask.png'))
        mask_pil_final.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}_finalmask.png'))

        ''' get overlays '''
        overlay_gt_vs_pred = pil_overlay_predicted_and_gt(mask_pil_gt, mask_pil_pred)
        overlay_gt_vs_final = pil_overlay_predicted_and_gt(mask_pil_gt, mask_pil_final)

        overlay_gt_vs_pred.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}_gtvspred.png'))
        overlay_gt_vs_final.save(path.join(out_dir_pth, f'{patient_id}_{projection}_{img_quality}_{mask_quality}_{data_setting}_gtvsfinal.png'))

        #joint_overlay_img_masks = pil_overlay(overlay_mask, img_org, alpha=0.2)


if __name__ == "__main__":
    prediction_name = 'Dec04_18-22-09_EFFIB0-DICBCE_AL_IMGN_ADAM_T-GE1956_HMLHML_V-NONE_EP150_LR0.001_BS20_SCL1_OUT'
    keyfile_csv = r'H:\ML_LV\backup_keyfiles\keyfile_GE1956_QC.csv'
    predicted_masks_dir = path.join('predictions', 'strain', prediction_name)

    original_dcms_dir = path.join('H:\ML_LV\predictions_dcm\strain_data_article', prediction_name, 'original')
    predicted_dcms_dir = path.join('H:\ML_LV\predictions_dcm\strain_data_article', prediction_name, 'predicted')


    get_3_masks_to_pil(keyfile_csv, predicted_masks_dir, original_dcms_dir, predicted_dcms_dir)





