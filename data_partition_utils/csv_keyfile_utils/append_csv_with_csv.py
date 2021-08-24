import os
import csv
import argparse

def append_data(old_key, quality_data, writer):
    with open(old_key, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        next(csv_reader)

        for row in csv_reader:
            dicom_id, exam_id, patient_id, x_s_pix, y_s_pix, x_i_pix, y_i_pix, x_s_cm, y_s_cm, x_i_cm, y_i_cm, lvot_diam_cm, scanconvert = row
            
            found = False
            with open(quality_data, 'r') as read_obj_append:
                csv_reader_append = csv.reader(read_obj_append)
                next(csv_reader_append)

                for row_append in csv_reader_append:
                    patient_id_append, _, _, _, _, measure_type, img_quality, gt_quality, img_view = row_append
                    
                    if patient_id_append == patient_id: 
                        found = True
                        writer.writerow([dicom_id, exam_id, patient_id, measure_type, img_quality, gt_quality, img_view, x_s_pix, y_s_pix, x_i_pix, y_i_pix, x_s_cm, y_s_cm, x_i_cm, y_i_cm, lvot_diam_cm, scanconvert])
            
            if found == False:
                print(patient_id, 'not found')
                
def main():
    parser = argparse.ArgumentParser(
        description='append file-inclusion status to keyfile')
    parser.add_argument('old_key')
    parser.add_argument('quality_data')
    args = parser.parse_args()
    
    file = open('keyfile_1424_all_data.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['file_id','exam_id','patient_id','measure_type','img_quality','gt_quality','img_view','x1','y1','x2','y2','x1_cm','y1_cm','x2_cm','y2_cm','lvot_diam_cm','scanconvert'])
    append_data(args.old_key, args.quality_data, writer)
    file.close()
    
main()