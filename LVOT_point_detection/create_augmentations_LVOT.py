import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.create_augmentation_imgs_and_masks import create_augmentations

if __name__ == '__main__':
    ''' create augmentations for the entire complete dataset '''

    dataset_name = 'AVA1314X5_HMLHML'
    n_augmention_copies = 4

    datasets_dir = 'datasets'
    augmentations_dir_output = 'augmentations'

    create_augmentations(dataset_name, datasets_dir, n_augmention_copies, augmentations_dir_output)
