import os
import os.path as path
import shutil 
import glob


def remove_tags_in_imgs_and_masks(imgs_dir, imgs_dir_output, masks_dir, masks_dir_output, tags_to_remove):
    
    for f in os.listdir(imgs_dir):
        
        filename = f.rsplit('.', 1)[0]
        tag_list = filename.rsplit('_', 4)
        #patient, measure_type, view, img_q, mask_q  = filename.rsplit('_', 4)
        
        in_tags = False
        for e in tag_list:
            if e in tags_to_remove:
                in_tags = True
                break
        
        if in_tags == False:
            shutil.copyfile(path.join(imgs_dir, f), path.join(imgs_dir_output, f))
            masks = glob.glob(f'{path.join(masks_dir, filename)}*')
            for m in masks:
                shutil.copyfile(m, path.join(masks_dir_output, path.basename(m)))
            

def remove_tag_from_folder(data_names, tags_to_remove, new_quality, is_kfold, imgs_dir, masks_dir):
    
    for data_name in data_names:
        
        imgs_dir_path = path.join(imgs_dir, f'imgs_{data_name}')
        masks_dir_path = path.join(masks_dir, f'masks_{data_name}')
        
        if is_kfold == True:
            data_type, dataset_name, data_quality, n_k = data_name.rsplit('_', 3)
            new_name = f'{data_type}_{dataset_name}_{new_quality}_{n_k}'
        else:
            data_type, dataset_name, data_quality = data_name.rsplit('_', 2)
            new_name = f'{data_type}_{dataset_name}_{new_quality}'
        
        imgs_aug_dir_output = path.join(imgs_dir, f'imgs_{new_name}')
        masks_aug_dir_output = path.join(masks_dir, f'masks_{new_name}')
        os.mkdir(imgs_aug_dir_output)
        os.mkdir(masks_aug_dir_output)

        remove_tags_in_imgs_and_masks(imgs_dir_path, imgs_aug_dir_output, masks_dir_path, masks_aug_dir_output, tags_to_remove)
    

if __name__ == '__main__':
    
    ''' remove tags from train folders '''
    
    #data_names = ['train_CAMUS1800_HML_K1', 'train_CAMUS1800_HML_K2', 'train_CAMUS1800_HML_K3', 'train_CAMUS1800_HML_K4', 'train_CAMUS1800_HML_K5']
    data_names = ['train_CAMUS1800_HML']
    is_kfold = False
    tags_to_remove = ['LOW']
    new_data_name = 'HM'
    
    imgs_train_dir = 'data\data_train'
    masks_train_dir = 'data\data_train'
    remove_tag_from_folder(data_names, tags_to_remove, new_data_name, is_kfold, imgs_train_dir, masks_train_dir)
    
    if is_kfold == True:
        ''' remove tags from validate folders '''
        
        data_names_validate = [f'validate_{data_name.split("_", 1)[-1]}' for data_name in data_names]
        
        imgs_validate_dir = 'data\data_validate'
        masks_validate_dir = 'data\data_validate'
        remove_tag_from_folder(data_names_validate, tags_to_remove, new_data_name, is_kfold, imgs_validate_dir, masks_validate_dir)
   
    
    
