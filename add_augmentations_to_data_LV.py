from utils.create_augmentation_imgs_and_masks_LV import create_augmentations
from utils.add_augmentation_to_train_folder_LV import add_aug_to_folder

import os.path as path

if __name__ == '__main__':
    
    ''' create augmentations for the entire complete dataset '''

    dataset_name = 'CAMUS1800_HML'
    n_augmention_copies = 4

    datasets_dir = 'datasets'
    augmentations_dir_output = 'augmentations'
    
    create_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output)

    ''' create train data with augmentations '''

    #data_names = ['CAMUS1800_HML_K1', 'CAMUS1800_HML_K2', 'CAMUS1800_HML_K3', 'CAMUS1800_HML_K4', 'CAMUS1800_HML_K5']
    data_names =['CAMUS1800_HM']
    is_kfold = False
    aug_name = 'CAMUS1800_HML_MA4'

    augmentation_dir = 'augmentations'
    data_dir = path.join('data', 'train')
    augmentation_data_dir_output = path.join('data', 'train')
    
    add_aug_to_folder(data_names, data_dir, is_kfold, aug_name, augmentation_dir, augmentation_data_dir_output)