import os.path as path
import sys
sys.path.insert(0, '..')
from data_partition_utils.edit_data_based_on_tags import remove_tag_from_folder

if __name__ == '__main__':

    ''' remove tags from train folders '''

    #data_names = ['CAMUS1800_HML_K1', 'CAMUS1800_HML_K2', 'CAMUS1800_HML_K3', 'CAMUS1800_HML_K4', 'CAMUS1800_HML_K5']
    data_names = ['RV141_HMLHML_K1', 'RV141_HMLHML_K2', 'RV141_HMLHML_K3', 'RV141_HMLHML_K4', 'RV141_HMLHML_K5']
    #data_names = ['GE1956_HMLHML']
    is_kfold = True
    tags_to_remove = ['ILOW', 'MLOW']
    new_data_name = 'HMHM'

    input_dir = path.join('data', 'train')
    output_dir = path.join('data', 'train')
    remove_tag_from_folder(data_names, tags_to_remove, new_data_name, is_kfold, input_dir, output_dir)

    if is_kfold == True:
        ''' remove tags from validate folders '''

        data_names_validate = data_names

        input_dir2 = path.join('data', 'validate')
        output_dir2 = path.join('data', 'validate')
        remove_tag_from_folder(data_names_validate, tags_to_remove, new_data_name, is_kfold, input_dir2, output_dir2)