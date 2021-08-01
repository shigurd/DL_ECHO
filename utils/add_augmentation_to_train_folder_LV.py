import os
import os.path as path
import shutil 
import glob


def add_aug_to_folder(data_names, data_dir, is_kfold, aug_name, augmentation_dir, augmentation_data_dir_output):
    
    for d in data_names:
        aug_name_tag = aug_name.rsplit('_', 1)[-1]
        imgs_path = path.join(data_dir, 'imgs', d)
        masks_path = path.join(data_dir, 'masks', d)
        imgs_aug_path = path.join(augmentation_dir, 'imgs', aug_name)
        masks_aug_path = path.join(augmentation_dir, 'masks', aug_name)

        if is_kfold == True:
            d_without_kfold, data_quality, n_k = d.rsplit('_', 2)
            imgs_dir_output_path = path.join(augmentation_data_dir_output, 'imgs', f'{d_without_kfold}_{data_quality}_{aug_name_tag}_{n_k}')
            masks_dir_output_path = path.join(augmentation_data_dir_output, 'masks', f'{d_without_kfold}_{data_quality}_{aug_name_tag}_{n_k}')
        else:
            imgs_dir_output_path = path.join(augmentation_data_dir_output, 'imgs', f'{d}_{aug_name_tag}')
            masks_dir_output_path = path.join(augmentation_data_dir_output, 'masks', f'{d}_{aug_name_tag}')
        
        os.mkdir(imgs_dir_output_path)
        os.mkdir(masks_dir_output_path)
        
        imgs_files = os.listdir(imgs_path)

        for f in imgs_files:
            ''' copy imgs '''
            f_org = f'{f.rsplit(".", 1)[0]}_ORG.png'
            shutil.copyfile(os.path.join(imgs_path, f), os.path.join(imgs_dir_output_path, f_org))
            
            filename = f.rsplit('.', 1)[0]
            filename = filename.rsplit("_", 1)[0]
            
            ''' copy masks '''
            masks = glob.glob(masks_path + '/' + filename + '*')
            for m in masks:
                m_org = f'{m.rsplit("_", 1)[0]}_ORG_mask.png'
                shutil.copyfile(m, os.path.join(masks_dir_output_path, os.path.basename(m_org)))
                
            ''' copy img augs '''
            img_augs = glob.glob(imgs_aug_path + '/' + filename + '*')
            for ia in img_augs:
                shutil.copyfile(ia, os.path.join(imgs_dir_output_path, os.path.basename(ia)))
                
            ''' copy mask augs '''
            mask_augs = glob.glob(masks_aug_path + '/' + filename + '*')
            for ma in mask_augs:
                shutil.copyfile(ma, os.path.join(masks_dir_output_path, os.path.basename(ma)))
        


if __name__ == '__main__':

    # data_names = ['CAMUS1800_HML_K1', 'CAMUS1800_HML_K2', 'CAMUS1800_HML_K3', 'CAMUS1800_HML_K4', 'CAMUS1800_HML_K5']
    data_names = ['CAMUS1800_HM']
    is_kfold = False
    aug_name = 'CAMUS1800_HML_MA4'

    augmentation_dir = 'augmentations'
    data_dir = path.join('data', 'train')
    augmentation_data_dir_output = path.join('data', 'train')

    add_aug_to_folder(data_names, data_dir, is_kfold, aug_name, augmentation_dir, augmentation_data_dir_output)