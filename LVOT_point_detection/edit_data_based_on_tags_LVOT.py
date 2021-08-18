import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.edit_data_based_on_tags import remove_tag_from_folder

if __name__ == '__main__':

    ''' remove tags from train folders '''

    # data_names = ['AVA1314Y1_HML_K1', 'AVA1314Y1_HML_K2', 'AVA1314Y1_HML_K3', 'AVA1314Y1_HML_K4', 'AVA1314Y1_HML_K5']
    data_names = ['AVA1314Y1_HML']
    is_kfold = False
    tags_to_remove = ['LOW']
    new_data_name = 'HM'

    input_dir = path.join('data', 'train')
    output_dir = path.join('data', 'train')
    remove_tag_from_folder(data_names, tags_to_remove, new_data_name, is_kfold, input_dir, output_dir)

    if is_kfold == True:
        ''' remove tags from validate folders '''

        data_names_validate = [data_name.split("_", 1)[-1] for data_name in data_names]

        input_dir2 = path.join('data', 'validate')
        output_dir2 = path.join('data', 'validate')
        remove_tag_from_folder(data_names_validate, tags_to_remove, new_data_name, is_kfold, input_dir2, output_dir2)