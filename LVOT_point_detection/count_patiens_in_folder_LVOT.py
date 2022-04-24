import os
from data_partition_utils.random_testsample_kfold import patient_dict_from_folder

if __name__ == '__main__':
    img_dir = 'data/test/imgs/GE1408_HMLHMLAVA'
    img_dir_files = os.listdir(img_dir)
    patient_dict = patient_dict_from_folder(img_dir)
    print('folder: ', img_dir)
    print('n patients: ', len(patient_dict))
    print('n files: ', len(img_dir_files))

    n_measurement_dict = dict()
    for key in patient_dict:
        if patient_dict[key] not in n_measurement_dict:
            n_measurement_dict[patient_dict[key]] = 1
        else:
            n_measurement_dict[patient_dict[key]] += 1

    for key2 in n_measurement_dict:
        print(key2, 'measurements has n =', n_measurement_dict[key2])