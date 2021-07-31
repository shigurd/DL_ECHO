from utils.create_augmentation_imgs_and_masks_LV import create_augmentations
from utils.add_augmentation_to_train_folder_LV import add_aug_to_folder

import os.path as path

if __name__ == '__main__':
    
    ''' create augmentations for the entire complete dataset '''
    
    dataset_name = 'CAMUS1800_HML'
    n_augmention_copies = 4

    datasets_imgs_dir = f'complete_datasets\imgs_{dataset_name}'
    datasets_masks_dir = f'complete_datasets\masks_{dataset_name}'
    augmentations_dir_output = 'data_augmentations'
    
    create_augmentations(dataset_name, datasets_imgs_dir, datasets_masks_dir, n_augmention_copies, augmentations_dir_output)
    

    ''' create train data with augmentations '''
    #data_names = ['train_CAMUS1800_HML_K1', 'train_CAMUS1800_HML_K2', 'train_CAMUS1800_HML_K3', 'train_CAMUS1800_HML_K4', 'train_CAMUS1800_HML_K5']
    data_names =['train_CAMUS1800_HM']
    is_kfold = False
    aug_name = 'augmentations_CAMUS1800_HML_MA4'
    augmentation_dir = 'data_augmentations'
    imgs_dir_output = 'data\data_train'
    masks_dir_output = 'data\data_train'
    
    
    add_aug_to_folder(data_names, aug_name, augmentation_dir, imgs_dir_output, masks_dir_output, is_kfold)