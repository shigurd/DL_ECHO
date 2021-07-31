import os
import os.path as path
import sys
import random
import shutil 
import glob

def random_sample_split(path_imgs, path_masks, test_percent, kfold, main_dir_output, partition_dir_output):
    
    main_output_name = path.basename(main_dir_output).rsplit('_', 1)[-1]
    partition_output_name = path.basename(partition_dir_output).rsplit('_', 1)[-1]
    
    data_name = path.basename(path_imgs).split('_', 1)[1]
    
    imgs_files = os.listdir(path_imgs)
    print(f'Splitting data: {path.basename(path_imgs).split("_", 1)[-1]} into {main_output_name} and {partition_output_name}')
    print('______________________________________________')
    print('Total files n =', len(imgs_files))
    
    filenames = []
    img_ext = '.png'
    mask_ext = '_mask.png'
    
    for f in imgs_files:
        filename = f.split('_')[0]
        if filename not in filenames:
            filenames.append(filename)
    
    # only counts patients if they are specified in the file. for LV it only counts exams
    print('Total patients n =', len(filenames))
    file_sum = int(len(filenames) * test_percent)
    print(f'{main_output_name} n =', len(filenames) - file_sum)
    print(f'{partition_output_name} n =', file_sum)
    
    
    if kfold == 1:
        main_output_imgs = f'imgs_{main_output_name}_{data_name}'
        main_output_masks = f'masks_{main_output_name}_{data_name}'
        partition_output_imgs = f'imgs_{partition_output_name}_{data_name}'
        partition_output_masks = f'masks_{partition_output_name}_{data_name}'        
    else:
        main_output_imgs = f'imgs_{main_output_name}_{data_name.split("_", 1)[-1]}'
        main_output_masks = f'masks_{main_output_name}_{data_name.split("_", 1)[-1]}'
        partition_output_imgs = f'imgs_{partition_output_name}_{data_name.split("_", 1)[-1]}'
        partition_output_masks = f'masks_{partition_output_name}_{data_name.split("_", 1)[-1]}'
    
    for folders in range(1, kfold + 1):
        
        if kfold == 1:
            add_kfold = ''
        else:
            add_kfold = f'_K{folders}'
            
        imgs_testdir = path.join(partition_dir_output, f'{partition_output_imgs}{add_kfold}')
        masks_testdir = path.join(partition_dir_output, f'{partition_output_masks}{add_kfold}')
        os.mkdir(imgs_testdir)
        os.mkdir(masks_testdir)
        
        imgs_traindir = path.join(main_dir_output, f'{main_output_imgs}{add_kfold}')
        masks_traindir = path.join(main_dir_output, f'{main_output_masks}{add_kfold}')
        os.mkdir(imgs_traindir)
        os.mkdir(masks_traindir)
    
    for k1 in range(1, kfold + 1):
    
    
        for p in range(file_sum):
                
            rand_selection = random.choice(filenames)
            filenames.remove(rand_selection)
            
            img_wildcard = glob.glob(path_imgs + '/' + rand_selection + '*')
            mask_wildcard = glob.glob(path_masks + '/' + rand_selection + '*')
            
            if kfold == 1:
                for k in img_wildcard:
                    shutil.copyfile(k, path.join(partition_dir_output, path.join(f'{partition_output_imgs}', path.basename(k))))
                    for k2 in range(1, kfold + 1):
                        if k2 != k1:
                            shutil.copyfile(k, path.join(main_dir_output, path.join(f'{main_output_imgs}', path.basename(k))))
                
                for d in mask_wildcard:
                    shutil.copyfile(d, path.join(partition_dir_output, path.join(f'{partition_output_masks}', path.basename(d))))
                    for k3 in range(1, kfold + 1):
                        if k3 != k1:
                            shutil.copyfile(d, path.join(main_dir_output, path.join(f'{main_output_masks}', path.basename(d))))
    
            else:            
                for k in img_wildcard:
                    shutil.copyfile(k, path.join(partition_dir_output, path.join(f'{partition_output_imgs}_K{k1}', path.basename(k))))
                    for k2 in range(1, kfold + 1):
                        if k2 != k1:
                            shutil.copyfile(k, path.join(main_dir_output, path.join(f'{main_output_imgs}_K{k2}', path.basename(k))))
                
                for d in mask_wildcard:
                    shutil.copyfile(d, path.join(partition_dir_output, path.join(f'{partition_output_masks}_K{k1}', path.basename(d))))
                    for k3 in range(1, kfold + 1):
                        if k3 != k1:
                            shutil.copyfile(d, path.join(main_dir_output, path.join(f'{main_output_masks}_K{k3}', path.basename(d))))
    
    # for kfold it adds the remaning files into train. if no kfold all the train files will be added here
    if kfold != 1:
        print('Rest n =', len(filenames), '\n')
    else:
        print('\n')
        
    for r in filenames:
        
        img_remain = glob.glob(path_imgs + '/' + r + '*')
        mask_remain = glob.glob(path_masks + '/' + r + '*')
        if kfold == 1:
            for n in img_remain:
                #shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, path.join(f'{main_output_imgs}', path.basename(n))))
            
            for s in mask_remain:
                #shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, path.join(f'{main_output_masks}', path.basename(s))))
        else:
            for n in img_remain:
                #shutil.copyfile(n, path.join(f'{partition_output_imgs}_{data_name}_K{j+1}', path.basename(n)))
                for k4 in range(1, kfold + 1):
                        shutil.copyfile(n, path.join(main_dir_output, path.join(f'{main_output_imgs}_K{k4}', path.basename(n))))
            
            for s in mask_remain:
                #shutil.copyfile(s, path.join(f'{partition_output_masks}_{data_name}_K{j+1}', path.basename(s)))
                for k5 in range(1, kfold + 1):
                        shutil.copyfile(s, path.join(main_dir_output, path.join(f'{main_output_masks}_K{k5}', path.basename(s))))

if __name__ == '__main__':

    # make test and train data
    data_name = 'CAMUS1800'
    test_percent = 0.15
    kfolds = 1
        
    path_imgs = f'datasets/imgs_{data_name}'
    path_masks = f'datasets/masks_{data_name}'
    
    random_sample_split(path_imgs, path_masks, test_percent, kfolds, 'train', 'validate')
    
