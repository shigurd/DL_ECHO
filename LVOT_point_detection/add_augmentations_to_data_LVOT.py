import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.create_augmentation_imgs_and_masks import create_augmentations
from data_partition_utils.add_augmentation_to_train_folder import add_aug_to_folder

if __name__ == '__main__':
    
    ''' create augmentations for the entire complete dataset '''

    dataset_name = 'AVA1314Y1_HML'
    n_augmention_copies = 4

    datasets_dir = 'datasets'
    augmentations_dir_output = 'augmentations'
    
    create_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output)

    ''' create train data with augmentations '''
    '''
    #data_names = ['CAMUS1800_HML_K1', 'CAMUS1800_HML_K2', 'CAMUS1800_HML_K3', 'CAMUS1800_HML_K4', 'CAMUS1800_HML_K5']
    data_names =['CAMUS1800_HM']
    is_kfold = False
    aug_name = 'CAMUS1800_HML_MA4'

    augmentation_dir = 'augmentations'
    data_dir = path.join('data', 'train')
    augmentation_data_dir_output = path.join('data', 'train')

    for data_name in data_names:
        add_aug_to_folder(data_name, data_dir, is_kfold, aug_name, augmentation_dir, augmentation_data_dir_output)
    '''