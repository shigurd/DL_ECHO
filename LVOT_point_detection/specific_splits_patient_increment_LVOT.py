from data_partition_utils.random_testsample_kfold import n_patient_partition
import os.path as path


if __name__ == '__main__':
    data_name = 'GE1423_HMLHMLAVA'
    img_folder = path.join('data', 'train', 'imgs', data_name)
    mask_folder = path.join('data', 'train', 'masks', data_name)

    n_patient_partition(img_folder, mask_folder, increment=100, n_masks=2)