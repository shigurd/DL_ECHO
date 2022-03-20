import os
from data_partition_utils.random_testsample_kfold import patient_list_from_folder

if __name__ == '__main__':
    img_dir = 'data/train/imgs/GE1408_HMLHML'
    img_dir_files = os.listdir(img_dir)
    patient_list = patient_list_from_folder(img_dir)
    print('folder: ', img_dir)
    print('n patients: ', len(patient_list))
    print('n files: ', len(img_dir_files))