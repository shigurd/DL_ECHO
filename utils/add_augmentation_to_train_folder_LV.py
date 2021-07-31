import os
import os.path as path
import shutil 
import glob


def add_aug_to_folder(data_names, aug_name, augmentation_dir, imgs_dir_output, masks_dir_output, is_kfold=False):
    
    for d in data_names:
        
        aug_name_output = aug_name.rsplit('_', 1)[-1]
        img_path = path.join(imgs_dir_output, f'imgs_{d}')
        mask_path = path.join(masks_dir_output, f'masks_{d}')

        if is_kfold == True:
            data_type, d_without_kfold, data_quality, n_k = d.rsplit('_', 3)
            aug_img_path = path.join(augmentation_dir, f'imgs_{aug_name}')
            aug_mask_path = path.join(augmentation_dir, f'masks_{aug_name}')
            imgs_dir_output_path = path.join(imgs_dir_output, f'imgs_{data_type}_{d_without_kfold}_{data_quality}_{aug_name_output}_{n_k}')
            masks_dir_output_path = path.join(masks_dir_output, f'masks_{data_type}_{d_without_kfold}_{data_quality}_{aug_name_output}_{n_k}')
        else:
            aug_img_path = path.join(augmentation_dir, f'imgs_{aug_name}')
            aug_mask_path = path.join(augmentation_dir, f'masks_{aug_name}')
            imgs_dir_output_path = path.join(imgs_dir_output, f'imgs_{d}_{aug_name_output}')
            masks_dir_output_path = path.join(masks_dir_output, f'masks_{d}_{aug_name_output}')
        
        os.mkdir(imgs_dir_output_path)
        os.mkdir(masks_dir_output_path)
        
        imgs_files = os.listdir(img_path)

        for f in imgs_files:
            ''' copy imgs '''
            f_org = f'{f.rsplit(".", 1)[0]}_ORG.png'
            shutil.copyfile(os.path.join(img_path, f), os.path.join(imgs_dir_output_path, f_org))
            
            filename = f.rsplit('.', 1)[0]
            filename = filename.rsplit("_", 1)[0]
            
            ''' copy masks '''
            masks = glob.glob(mask_path + '/' + filename + '*')
            for m in masks:
                m_org = f'{m.rsplit("_", 1)[0]}_ORG_mask.png'
                shutil.copyfile(m, os.path.join(masks_dir_output_path, os.path.basename(m_org)))
                
            ''' copy img augs '''
            img_augs = glob.glob(aug_img_path + '/' + filename + '*')
            for ia in img_augs:
                shutil.copyfile(ia, os.path.join(imgs_dir_output_path, os.path.basename(ia)))
                
            ''' copy mask augs '''
            mask_augs = glob.glob(aug_mask_path + '/' + filename + '*')
            for ma in mask_augs:
                shutil.copyfile(ma, os.path.join(masks_dir_output_path, os.path.basename(ma)))
        


if __name__ == '__main__':
    
    data_names = ['train_CAMUS1800_K1', 'train_CAMUS1800_K2', 'train_CAMUS1800_K3', 'train_CAMUS1800_K4', 'train_CAMUS1800_K5']
    aug_name = 'AUG4'
    is_kfold = True
    imgs_dir_output = 'data\data_train'
    masks_dir_output = 'data\data_train'
    augmentation_dir = 'data_augmentations'
    
    add_aug_to_folder(data_names, aug_name, is_kfold, augmentation_dir, imgs_dir_output, masks_dir_output)
