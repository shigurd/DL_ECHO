import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.edit_data_based_on_tags import remove_tag_from_folder

if __name__ == '__main__':

    ''' remove tags from train folders '''

    #data_names = ['GE1423_HMLHMLAVA_K1', 'GE1423_HMLHMLAVA_K2', 'GE1423_HMLHMLAVA_K3', 'GE1423_HMLHMLAVA_K4', 'GE1423_HMLHMLAVA_K5']
    data_names = ['GE1423_HMLHMLAVA']
    is_kfold = False
    tags_to_remove = ['ALAX', 'TRANS']
    new_data_name = '1408_HMLHMLAVA'

    input_dir = path.join('data', 'test')
    output_dir = path.join('data', 'test')
    remove_tag_from_folder(data_names, tags_to_remove, new_data_name, is_kfold, input_dir, output_dir)

    if is_kfold == True:
        ''' remove tags from validate folders '''

        data_names_validate = data_names

        input_dir2 = path.join('data', 'validate')
        output_dir2 = path.join('data', 'validate')
        remove_tag_from_folder(data_names_validate, tags_to_remove, new_data_name, is_kfold, input_dir2, output_dir2)