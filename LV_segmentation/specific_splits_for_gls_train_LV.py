from data_partition_utils.random_testsample_kfold import match_patient_count, n_patient_partition
import os.path as path

def match_rea_with_cli_patients():
    small_name = 'GE1956_HMLHMLCLI'
    large_name = 'GE1956_HMLHMLRES'
    small_folder_img = path.join('data', 'train', 'imgs', small_name)
    large_folder_img = path.join('data', 'train', 'imgs', large_name)
    large_folder_mask = path.join('data', 'train', 'masks', large_name)

    new_name = 'GE1956_HMLHMLRESMatched'
    new_folder_img = path.join('data', 'train', 'imgs', new_name)
    new_folder_mask = path.join('data', 'train', 'masks', new_name)

    match_patient_count(small_folder_img, large_folder_img, large_folder_mask, new_folder_img, new_folder_mask)


def partitioned_patient_count():
    data_name = 'GE1956_HMLHML'
    img_folder = path.join('data', 'train', 'imgs', data_name)
    mask_folder = path.join('data', 'train', 'masks', data_name)
    n_patient_partition(img_folder, mask_folder, increment=100)


if __name__ == '__main__':
    #match_rea_with_cli_patients()
    partitioned_patient_count()