import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.random_testsample_kfold import random_sample_split


if __name__ == '__main__':

    ''' partition of complete dataset into test and train '''

    data_name = 'AVA1314Y1_HML'
    test_percent = 0.15
    kfold = 1

    datasets_dir = 'datasets'
    main_dir_output = path.join('data', 'train')
    partition_dir_output = path.join('data', 'test')

    random_sample_split(datasets_dir, data_name, test_percent, kfold, main_dir_output, partition_dir_output)
    
    ''' partition of train into kfold train and validation '''

    data_name2 = 'AVA1314Y1_HML'
    test_percent2 = 0.2
    kfold2 = 5

    datasets_dir2 = path.join('data', 'train')
    main_dir_output2 = path.join('data', 'train')
    partition_dir_output2 = path.join('data', 'validate')

    random_sample_split(datasets_dir2, data_name2, test_percent2, kfold2, main_dir_output2, partition_dir_output2)
