import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.add_augmentation_to_train_folder import add_aug_to_folder

if __name__ == '__main__':

    ''' create train data with augmentations '''

    data_names = ['GE1956_HMHM_K1', 'GE1956_HMHM_K2', 'GE1956_HMHM_K3', 'GE1956_HMHM_K4', 'GE1956_HMHM_K5']
    #data_names =['GE1956_HMLHML']
    is_kfold = True
    aug_name = 'GE1956_HMLHML_MA4'

    augmentation_dir = 'augmentations'
    data_dir = path.join('data', 'train')
    augmentation_data_dir_output = path.join('data', 'train')

    for data_name in data_names:
        add_aug_to_folder(data_name, data_dir, is_kfold, aug_name, augmentation_dir, augmentation_data_dir_output)