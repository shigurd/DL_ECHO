import os
import os.path as path
import csv
from glob import glob

if __name__ == '__main__':

    dataset_name = 'GE1424_HMLHML'
    keyfile_QC_path = r'H:\ML_LVOT\backup_keyfile_and_duplicate\keyfile_GE1424_QC.csv'
    imgs_dir = path.join('datasets', 'imgs', dataset_name)
    masks_dir = path.join('datasets', 'masks', dataset_name)

    for i in os.listdir(imgs_dir):
        dicom_name = i.rsplit('.', 1)[0]

        with open(keyfile_QC_path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader) #skip header

            for row in csv_reader:
                dicom_id, exam_id, patient_id, measure_type, img_quality, gt_quality, img_view, x_s_pix, y_s_pix, x_i_pix, y_i_pix, x_superior_cm, y_superior_cm, x_inferior_cm, y_inferior_cm, lvot_diam_cm, scanconvert = row

                if dicom_id == dicom_name:

                    ''' rename img '''
                    os.rename(path.join(imgs_dir, i), path.join(imgs_dir, f'{patient_id}_{measure_type}_{img_view}_{img_quality}_{gt_quality}.png'))

                    ''' rename masks '''
                    masks_paths = glob(path.join(masks_dir, dicom_name) + '*')

                    os.rename(masks_paths[0], path.join(masks_dir, f'{patient_id}_{measure_type}_{img_view}_I{img_quality}_L{gt_quality}_imask.png'))
                    os.rename(masks_paths[1], path.join(masks_dir, f'{patient_id}_{measure_type}_{img_view}_I{img_quality}_L{gt_quality}_smask.png'))

                    break