import os.path as path
from utils.random_testsample_kfold_LV import random_sample_split


if __name__ == '__main__':
    
    ''' partition of complete dataset into test and train '''
    data_name = 'CAMUS1800_HML'
    test_percent = 0.15
    kfold = 1

    datasets_imgs_dir = f'complete_datasets\imgs_{data_name}'
    datasets_masks_dir = f'complete_datasets\masks_{data_name}'
    main_dir_output = 'data\data_train'
    partition_dir_output = 'data\data_test'
    
    random_sample_split(datasets_imgs_dir, datasets_masks_dir, test_percent, kfold, main_dir_output, partition_dir_output)
    
    ''' partition of train into kfold train and validation '''
    data_name2 = 'train_CAMUS1800_HML'
    test_percent2 = 0.2
    kfold2 = 5

    datasets_imgs_dir2 = f'data\data_train\imgs_{data_name2}'
    datasets_masks_dir2 = f'data\data_train\masks_{data_name2}'
    main_dir_output2 = 'data\data_train'
    partition_dir_output2 = 'data\data_validate'

    random_sample_split(datasets_imgs_dir2, datasets_masks_dir2, test_percent2, kfold2, main_dir_output2, partition_dir_output2)
