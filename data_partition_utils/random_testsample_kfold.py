import os
import os.path as path
import sys
import random
import shutil 
import glob
import csv

def split_text_on_first_number(input_str):
    str_numbers = [str(x) for x in range(10)]
    for i, s in enumerate(input_str):
        if s in str_numbers:
            number_idx = i
            break

    letters = input_str[:number_idx]
    numbers = input_str[number_idx:]

    return letters, numbers


def lv_stratified_split(datasets_dir, data_name, test_percent, kfold, main_dir_output, partition_dir_output, csv_keyfile):
    imgs_path = path.join(datasets_dir, 'imgs', data_name)
    masks_path = path.join(datasets_dir, 'masks', data_name)

    imgs_files = os.listdir(imgs_path)
    print(f'Splitting data: {path.basename(imgs_path)}\n'
          f'Main partition: {main_dir_output}\n'
          f'Secondary partition {partition_dir_output}')
    print('_______________________________')
    print('Total files n =', len(imgs_files))

    patients_in_folder = []

    for f in imgs_files:
        patinet_in_file = f.split('_')[0]
        if patinet_in_file not in patients_in_folder:
            patients_in_folder.append(patinet_in_file)

    ''' only counts patients if they are specified in the file. sanity check count '''
    print('Total patients n =', len(patients_in_folder))

    ''' stratification using csv file '''
    studies = dict()
    patients_in_csv = []

    with open(csv_keyfile, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)

        for row in csv_reader:
            dicom_id, patient_id, exam_id, projection, img_quality, mask_quality, data_setting = row

            exam_study, exam_study_number = split_text_on_first_number(exam_id)
            patient_id_no_n = patient_id.split('_', 1)[0]

            if patient_id_no_n not in patients_in_folder:
                pass

            else:
                if exam_study not in ('CRID', 'CRIDP'):
                    if exam_study not in studies:
                        studies[exam_study] = [patient_id_no_n]
                    else:
                        if patient_id_no_n not in studies[exam_study]:
                            studies[exam_study].append(patient_id_no_n)
                else:
                    if 'CRID_CRIDP' not in studies:
                        studies['CRID_CRIDP'] = [patient_id_no_n]
                    else:
                        if patient_id_no_n not in studies['CRID_CRIDP']:
                            studies['CRID_CRIDP'].append(patient_id_no_n)

            if patient_id_no_n not in patients_in_csv:
                patients_in_csv.append(patient_id_no_n)

    print('Total patients in csv n =', len(patients_in_csv), '\n_____________________________')

    #for kk in studies:
        #print(kk, studies[kk])

    ''' 'make kfold folders '''
    for folders in range(1, kfold + 1):
        if kfold == 1:
            add_kfold = ''
        else:
            add_kfold = f'_K{folders}'

        imgs_train_dir = path.join(main_dir_output, 'imgs', f'{data_name}{add_kfold}')
        masks_train_dir = path.join(main_dir_output, 'masks', f'{data_name}{add_kfold}')
        os.mkdir(imgs_train_dir)
        os.mkdir(masks_train_dir)

        imgs_test_dir = path.join(partition_dir_output, 'imgs', f'{data_name}{add_kfold}')
        masks_test_dir = path.join(partition_dir_output, 'masks', f'{data_name}{add_kfold}')
        os.mkdir(imgs_test_dir)
        os.mkdir(masks_test_dir)

    ''' put files into kfolds '''
    for key in studies:
        rounded_patients = round(len(studies[key]) * test_percent)
        print(f'{main_dir_output} {key} n patients =', len(studies[key]) - rounded_patients)
        print(f'{partition_dir_output} {key} n patients =', rounded_patients)
        print('______________________________')

        for k1 in range(1, kfold + 1):
            for p in range(rounded_patients):

                try:
                    rand_selection = random.choice(studies[key])
                    studies[key].remove(rand_selection)

                    img_wildcard = glob.glob(imgs_path + '/' + rand_selection + '*')
                    mask_wildcard = glob.glob(masks_path + '/' + rand_selection + '*')

                    if kfold == 1:
                        for k in img_wildcard:
                            shutil.copyfile(k, path.join(partition_dir_output, 'imgs', data_name, path.basename(k)))
                            for k2 in range(1, kfold + 1):
                                if k2 != k1:
                                    shutil.copyfile(k, path.join(main_dir_output, 'imgs', data_name, path.basename(k)))

                        for d in mask_wildcard:
                            shutil.copyfile(d, path.join(partition_dir_output, 'masks', data_name, path.basename(d)))
                            for k3 in range(1, kfold + 1):
                                if k3 != k1:
                                    shutil.copyfile(d, path.join(main_dir_output, 'masks', data_name, path.basename(d)))

                    else:
                        for k in img_wildcard:
                            shutil.copyfile(k, path.join(partition_dir_output, 'imgs', f'{data_name}_K{k1}', path.basename(k)))
                            for k2 in range(1, kfold + 1):
                                if k2 != k1:
                                    shutil.copyfile(k,
                                                    path.join(main_dir_output, 'imgs', f'{data_name}_K{k2}', path.basename(k)))

                        for d in mask_wildcard:
                            shutil.copyfile(d, path.join(partition_dir_output, 'masks', f'{data_name}_K{k1}', path.basename(d)))
                            for k3 in range(1, kfold + 1):
                                if k3 != k1:
                                    shutil.copyfile(d,
                                                    path.join(main_dir_output, 'masks', f'{data_name}_K{k3}', path.basename(d)))
                except:
                    print('skipped', key, 'for', k1, 'due to n in list =', studies[key])


    ''' for kfold it adds the remaning files into train. if no kfold all the train files will be added here '''
    for key in studies:
        for remaining in studies[key]:
            img_remain = glob.glob(imgs_path + '/' + remaining + '*')
            mask_remain = glob.glob(masks_path + '/' + remaining + '*')
            if kfold == 1:
                for n in img_remain:
                    # shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                    for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, 'imgs', f'{data_name}', path.basename(n)))

                for s in mask_remain:
                    # shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                    for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, 'masks', f'{data_name}', path.basename(s)))
            else:
                for n in img_remain:
                    # shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                    for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, 'imgs', f'{data_name}_K{k4}', path.basename(n)))

                for s in mask_remain:
                    # shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                    for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, 'masks', f'{data_name}_K{k5}', path.basename(s)))



def random_sample_split(datasets_dir, data_name, test_percent, kfold, main_dir_output, partition_dir_output):

    imgs_path = path.join(datasets_dir, 'imgs', data_name)
    masks_path = path.join(datasets_dir, 'masks', data_name)

    imgs_files = os.listdir(imgs_path)
    print(f'Splitting data: {path.basename(imgs_path)}\n'
          f'Main partition: {main_dir_output}\n'
          f'Secondary partition {partition_dir_output}')
    print('______________________________________________')
    print('Total files n =', len(imgs_files))
    
    filenames = []
    
    for f in imgs_files:
        filename = f.split('_')[0]
        if filename not in filenames:
            filenames.append(filename)
    
    ''' only counts patients if they are specified in the file. for LV it only counts exams '''
    print('Total patients n =', len(filenames))
    file_sum = int(len(filenames) * test_percent)
    print(f'{main_dir_output} n patients =', len(filenames) - file_sum)
    print(f'{partition_dir_output} n patients =', file_sum)
    
    for folders in range(1, kfold + 1):
        
        if kfold == 1:
            add_kfold = ''
        else:
            add_kfold = f'_K{folders}'

        imgs_train_dir = path.join(main_dir_output, 'imgs', f'{data_name}{add_kfold}')
        masks_train_dir = path.join(main_dir_output, 'masks', f'{data_name}{add_kfold}')
        os.mkdir(imgs_train_dir)
        os.mkdir(masks_train_dir)

        imgs_test_dir = path.join(partition_dir_output, 'imgs', f'{data_name}{add_kfold}')
        masks_test_dir = path.join(partition_dir_output, 'masks', f'{data_name}{add_kfold}')
        os.mkdir(imgs_test_dir)
        os.mkdir(masks_test_dir)
    
    for k1 in range(1, kfold + 1):
        for p in range(file_sum):
                
            rand_selection = random.choice(filenames)
            filenames.remove(rand_selection)
            
            img_wildcard = glob.glob(imgs_path + '/' + rand_selection + '*')
            mask_wildcard = glob.glob(masks_path + '/' + rand_selection + '*')
            
            if kfold == 1:
                for k in img_wildcard:
                    shutil.copyfile(k, path.join(partition_dir_output, 'imgs', data_name, path.basename(k)))
                    for k2 in range(1, kfold + 1):
                        if k2 != k1:
                            shutil.copyfile(k, path.join(main_dir_output, 'imgs', data_name, path.basename(k)))
                
                for d in mask_wildcard:
                    shutil.copyfile(d, path.join(partition_dir_output, 'masks', data_name, path.basename(d)))
                    for k3 in range(1, kfold + 1):
                        if k3 != k1:
                            shutil.copyfile(d, path.join(main_dir_output, 'masks', data_name, path.basename(d)))
    
            else:            
                for k in img_wildcard:
                    shutil.copyfile(k, path.join(partition_dir_output, 'imgs', f'{data_name}_K{k1}', path.basename(k)))
                    for k2 in range(1, kfold + 1):
                        if k2 != k1:
                            shutil.copyfile(k, path.join(main_dir_output, 'imgs', f'{data_name}_K{k2}', path.basename(k)))
                
                for d in mask_wildcard:
                    shutil.copyfile(d, path.join(partition_dir_output, 'masks', f'{data_name}_K{k1}', path.basename(d)))
                    for k3 in range(1, kfold + 1):
                        if k3 != k1:
                            shutil.copyfile(d, path.join(main_dir_output, 'masks', f'{data_name}_K{k3}', path.basename(d)))
    
    ''' for kfold it adds the remaning files into train. if no kfold all the train files will be added here '''
    if kfold != 1:
        print('Rest n =', len(filenames), '\n')
    else:
        print('\n')
        
    for r in filenames:
        
        img_remain = glob.glob(imgs_path + '/' + r + '*')
        mask_remain = glob.glob(masks_path + '/' + r + '*')
        if kfold == 1:
            for n in img_remain:
                #shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, 'imgs' ,f'{data_name}', path.basename(n)))
            
            for s in mask_remain:
                #shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, 'masks', f'{data_name}', path.basename(s)))
        else:
            for n in img_remain:
                #shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, 'imgs', f'{data_name}_K{k4}', path.basename(n)))
            
            for s in mask_remain:
                #shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, 'masks', f'{data_name}_K{k5}', path.basename(s)))


if __name__ == '__main__':

    # make test and train data
    data_name = 'CAMUS1800_HML'
    test_percent = 0.15
    kfold = 1

    datasets_dir = 'datasets'
    main_dir_output = path.join('data', 'train')
    partition_dir_output = path.join('data', 'test')

    random_sample_split(datasets_dir, data_name, test_percent, kfold, main_dir_output, partition_dir_output)
    
