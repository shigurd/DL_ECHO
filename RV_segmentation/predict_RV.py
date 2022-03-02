import logging
import os
import os.path as path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from LV_segmentation.utils.dataloader_LV import BasicDataset
from LV_segmentation.utils.convert_myomask_to_endomask_and_epimask_LV import get_endocard_epicard_from_np
from LV_segmentation.utils.segmentation_losses_LV import DiceHard

#from torchvision.models.segmentation import fcn_resnet50
sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from networks.unet import UNet
import segmentation_models_pytorch as smp
import SimpleITK as sitk

def predict_tensor(net,
                img_pil,
                device,
                scale_factor=1,
                mid_systole=False,
                coord_conv=False):
    net.eval()

    img_np = BasicDataset.preprocess(img_pil, scale_factor)

    if mid_systole:
        img_np = BasicDataset.extract_midsystole(img_np)
    if coord_conv:
        img_np = BasicDataset.add_coord_conv(img_np)
    
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_tensor)
        #output = output['out'] #torchvision syntax
    
    return output


def convert_tensor_mask_to_pil(mask_tensor_predicted):
    mask_tensor_predicted = torch.sigmoid(mask_tensor_predicted)
    
    mask_np_predicted = mask_tensor_predicted.squeeze().cpu().numpy()
    mask_np_predicted = mask_np_predicted > mask_threshold
    mask_pil_predicted = mask_to_image(mask_np_predicted)
    
    return mask_pil_predicted


def get_output_filenames(in_file):
    in_files = in_file
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], '.png'))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def concat_img(img1, img2):
    newImg = Image.new('RGB', (img1.width, img1.height + img2.height))
    newImg.paste(img1, (0, 0))
    newImg.paste(img2, (0, img1.height))
    return newImg


def pil_overlay_predicted_and_gt(mask, pred_img):
    mask = mask.convert('RGB')
    mask = np.array(mask, dtype=np.float) # float for negative values
    mask[:, :, 0] = 0 # removes R and B in RGB
    mask[:, :, 2] = 0 
    
    pred_mask = pred_img.convert('RGB')
    pred_mask = np.array(pred_mask, dtype=np.float) 
    pred_mask[:, :, 1] = 0 # removes G and B in RGB
    pred_mask[:, :, 2] = 0
    
    absolutt_diff = np.absolute(mask - pred_mask)
    absolutt_diff = np.array(absolutt_diff, dtype=np.uint8) # convert to unit8 for saving
    
    plot = Image.fromarray(absolutt_diff, 'RGB')
    #plot.show()
    return plot


def pil_overlay(foreground, background, alpha=0.1):
    img1 = foreground.convert("RGBA")
    img2 = background.convert("RGBA")
    
    overlay = Image.blend(img2, img1, alpha=alpha)
    
    return overlay


def calc_hausdorff(true_torch, pred_torch):
    true_torch = torch.sigmoid(true_torch) > 0.5
    pred_torch = torch.sigmoid(pred_torch) > 0.5

    true_np = true_torch.squeeze().cpu().numpy()
    pred_np = pred_torch.squeeze().cpu().numpy()

    true_np = (true_np * 255).astype(np.uint8)
    pred_np = (pred_np * 255).astype(np.uint8)

    img_sitk_true = sitk.GetImageFromArray(true_np)
    img_sitk_pred = sitk.GetImageFromArray(pred_np)

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(img_sitk_true, img_sitk_pred)

    hd = hausdorff_distance_filter.GetHausdorffDistance()

    return hd


def endocard_epicard_to_tensor(mask_pil):
    mask_np = np.array(mask_pil)
    endocard_np, epicard_np = get_endocard_epicard_from_np(mask_np)
    endocard_pil = Image.fromarray(endocard_np).convert('L')
    epicard_pil = Image.fromarray(epicard_np).convert('L')
    endocard_np = BasicDataset.preprocess(endocard_pil, scale=1)
    epicard_np = BasicDataset.preprocess(epicard_pil, scale=1)
    endocard_tensor = torch.from_numpy(endocard_np).cuda()
    epicard_tensor = torch.from_numpy(epicard_np).cuda()

    return endocard_tensor, epicard_tensor, endocard_pil, epicard_pil


if __name__ == "__main__":
    
    ''' define model name, prediction dataset and model parameters '''
    model_file = f'Mar01_16-07-25_EFFIB1UNET-LR5-DICBCE_AL_IMGN_ADAM_T-LV715_flipped_V-RV146_EP30_LR0.001_BS20_SCL1.pth'
    data_name = 'RV146'
    n_channels = 1
    n_classes = 1
    scaling = 1
    mask_threshold = 0.5
    mid_systole = True
    coord_conv = False
    compare_with_ground_truth = True
    convert_to_epicard_and_endocard = False

    model_path = path.join('checkpoints', model_file)
    dir_img = path.join('data', 'validate', 'imgs', data_name)
    dir_mask = path.join('data', 'validate', 'masks', data_name)

    ''' make output dir '''
    if compare_with_ground_truth == True:
        model_name = f'{data_name}___{model_file.rsplit(".", 1)[0]}_VAL'
    else:
        model_name = f'{model_file.rsplit(".", 1)[0]}_OUT'

    predictions_output = path.join('predictions', model_name)
    os.mkdir(predictions_output)
    
    ''' create filenames for output '''
    input_files = os.listdir(dir_img)
    out_files = get_output_filenames(input_files)
    
    ''' define network settings '''
    #net = fcn_resnet50(pretrained=False, progress=True, in_channels=n_channels, num_classes=n_classes, aux_loss=None)
    net = smp.Unet(encoder_name="efficientnet-b1", encoder_weights=None, in_channels=n_channels, classes=n_classes)
    #net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=n_channels, classes=n_classes)
    #net = UNet(n_channels, n_classes, bilinear=False)
    #net = smp.Unet(encoder_name="se_resnext50_32x4d", encoder_weights=None, in_channels=n_channels, classes=n_classes)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    
    ''' load checkpoint data '''
    checkpoint_data = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint_data['model_state_dict'])
    logging.info("Checkpoint loaded !")
    
    if compare_with_ground_truth == True:
        file = open(path.join(predictions_output, f'DICE_HD_DATA.txt'), 'w+')
        file.write('file_name,projection,data_setting,img_quality,gt_quality,dice_score,hd_score\n')
        file1 = open(path.join(predictions_output, 'AVG_MEDIAN.txt'), 'w+')

        median_dice_list = np.array([])
        median_hd_list = np.array([])
        total_dice = 0
        total_hd = 0

        if convert_to_epicard_and_endocard == True:
            endo_output = path.join(predictions_output, 'endocard')
            os.mkdir((endo_output))
            file_endo = open(path.join(endo_output, f'DICE_HD_DATA_ENDO.txt'), 'w+')
            file_endo.write('file_name,projection,data_setting,img_quality,gt_quality,dice_score,hd_score\n')
            file_endo1 = open(path.join(endo_output, 'AVG_MEDIAN_endo.txt'), 'w+')

            median_dice_list_endo = np.array([])
            median_hd_list_endo = np.array([])
            total_dice_endo = 0
            total_hd_endo = 0

            epi_output = path.join(predictions_output, 'epicard')
            os.mkdir(epi_output)
            file_epi = open(path.join(epi_output, f'DICE_HD_DATA_EPI.txt'), 'w+')
            file_epi.write('file_name,projection,data_setting,img_quality,gt_quality,dice_score,hd_score\n')
            file_epi1 = open(path.join(epi_output, 'AVG_MEDIAN_epi.txt'), 'w+')

            median_dice_list_epi = np.array([])
            median_hd_list_epi = np.array([])
            total_dice_epi = 0
            total_hd_epi = 0


    with tqdm(total=len(input_files), desc='Predictions', unit='imgs', leave=False) as pbar:

        for i, fn in enumerate(input_files):
            out_fn = out_files[i]
            logging.info("\nPredicting image {} ...".format(fn))
            img_pil = Image.open(path.join(dir_img, fn))
            img_pil = img_pil.convert('RGB')

            ''' predict_tensor returns logits '''
            mask_tensor_predicted = predict_tensor(net=net,
                               img_pil=img_pil,
                               scale_factor=scaling,
                               device=device,
                               mid_systole=mid_systole,
                               coord_conv=coord_conv)

            ''' converting predicted tensor to pil mask '''
            mask_pil_predicted = convert_tensor_mask_to_pil(mask_tensor_predicted)

            ''' if ground truth is available, make overlays and calculate mean and median dice '''
            if compare_with_ground_truth == True:
                mask_path_true = path.join(dir_mask, f'{fn.rsplit(".", 1)[0]}_mask.png')
                mask_pil_true = Image.open(mask_path_true)
                mask_pil_true = mask_pil_true.convert('L')
                mask_np_true = BasicDataset.preprocess(mask_pil_true, scaling)
                mask_tensor_true = torch.from_numpy(mask_np_true).cuda() # to cuda as this is loaded with cpu

                criterion = DiceHard()
                dice_score = criterion(mask_tensor_predicted, mask_tensor_true).item()
                hd_score = calc_hausdorff(mask_tensor_predicted, mask_tensor_true)

                ''' calculate mean dice and median dice and logging in txt '''
                total_dice += dice_score
                total_hd += hd_score
                median_dice_list = np.append(median_dice_list, dice_score)
                median_hd_list = np.append(median_hd_list, hd_score)

                ''' added support for CAMUS style files with n file tags '''
                fn_tags_list = fn.rsplit('.', 1)[0].rsplit('_')
                if len(fn_tags_list) < 5:
                    write_string = ''
                    for tag in fn_tags_list:
                        write_string += f'{tag},'
                    write_string = write_string[:-1] + '\n'
                    file.write(write_string)
                else:
                    patient_id, projection, data_setting, img_quality, gt_quality = fn.rsplit('.', 1)[0].rsplit('_', 4)
                    file.write(f'{fn},{projection},{data_setting},{img_quality},{gt_quality},{"{:.4f}".format(dice_score)},{"{:.4f}".format(hd_score)}\n') #rounds

                ''' plotting overlays between predicted masks and gt masks '''
                comparison_masks = pil_overlay_predicted_and_gt(mask_pil_true, mask_pil_predicted)
                ''' plotting overlays between predicted masks and input image '''
                #prediction_on_img = pil_overlay(mask_pil_true.convert('L'), img_pil)

                img_with_comparison = concat_img(img_pil, comparison_masks)
                img_with_comparison.save(path.join(predictions_output, f'{str("{:.4f}".format(dice_score).rsplit(".", 1)[1])}_{out_fn}')) #removes 0. from dice

                if convert_to_epicard_and_endocard == True:
                    endocard_tensor_predicted, epicard_tensor_predicted, endocard_pil_predicted, epicard_pil_predicted = endocard_epicard_to_tensor(mask_pil_predicted)
                    endocard_tensor_true, epicard_tensor_true, endocard_pil_true, epicard_pil_true = endocard_epicard_to_tensor(mask_pil_true)

                    dice_score_endo = criterion(endocard_tensor_predicted, endocard_tensor_true).item()
                    hd_score_endo = calc_hausdorff(endocard_tensor_predicted, endocard_tensor_true)

                    total_dice_endo += dice_score_endo
                    total_hd_endo += hd_score_endo
                    median_dice_list_endo = np.append(median_dice_list_endo, dice_score_endo)
                    median_hd_list_endo = np.append(median_hd_list_endo, hd_score_endo)

                    file_endo.write(f'{fn},{projection},{data_setting},{img_quality},{gt_quality},{"{:.4f}".format(dice_score_endo)} \n') #rounds
                    comparison_masks_endo = pil_overlay_predicted_and_gt(endocard_pil_true, endocard_pil_predicted)
                    img_with_comparison_endo = concat_img(img_pil, comparison_masks_endo)
                    img_with_comparison_endo.save(path.join(endo_output, f'{str("{:.4f}".format(dice_score_endo)).rsplit(".", 1)[1]}_{out_fn}')) #removes 0. from dice

                    dice_score_epi = criterion(epicard_tensor_predicted, epicard_tensor_true).item()
                    hd_score_epi = calc_hausdorff(epicard_tensor_predicted, epicard_tensor_true)

                    total_dice_epi += dice_score_epi
                    total_hd_epi += hd_score_epi
                    median_dice_list_epi = np.append(median_dice_list_epi, dice_score_epi)
                    median_hd_list_epi = np.append(median_hd_list_epi, hd_score_epi)

                    file_epi.write(f'{fn},{projection},{data_setting},{img_quality},{gt_quality},{"{:.4f}".format(dice_score_epi)} \n') #rounds
                    comparison_masks_epi = pil_overlay_predicted_and_gt(epicard_pil_true, epicard_pil_predicted)
                    img_with_comparison_epi = concat_img(img_pil, comparison_masks_epi)
                    img_with_comparison_epi.save(path.join(epi_output, f'{str("{:.4f}".format(dice_score_epi)).rsplit(".", 1)[1]}_{out_fn}')) #removed 0. from dice

            else:
                ''' just save predicted masks '''
                mask_pil_predicted.save(path.join(predictions_output, out_fn))

            logging.info("Mask saved to {}".format(out_files[i]))
            pbar.update()

        if compare_with_ground_truth == True:
            file.close()
            avg_dice = '{:.4f}'.format(total_dice / (i + 1)) #rounds
            median_dice = '{:.4f}'.format(np.median(median_dice_list)) #rounds
            avg_hd = '{:.4f}'.format(total_hd / (i + 1)) #rounds
            median_hd = '{:.4f}'.format(np.median(median_hd_list)) #rounds

            file1.write(f'AVG Dice: {avg_dice}\n')
            file1.write(f'MEDIAN Dice: {median_dice}\n')
            file1.write(f'AVG HD: {avg_hd}\n')
            file1.write(f'MEDIAN HD: {median_hd}\n')
            file1.close()

            if convert_to_epicard_and_endocard == True:
                file_endo.close()
                avg_dice_endo = '{:.4f}'.format(total_dice_endo / (i + 1)) #rounds
                median_dice_endo = '{:.4f}'.format(np.median(median_dice_list_endo)) #rounds
                avg_hd_endo = '{:.4f}'.format(total_hd_endo / (i + 1)) #rounds
                median_hd_endo = '{:.4f}'.format(np.median(median_hd_list_endo)) #rounds

                file_endo1.write(f'MEAN Dice: {avg_dice_endo}\n')
                file_endo1.write(f'MEDIAN Dice: {median_dice_endo}\n')
                file_endo1.write(f'MEAN HD: {avg_hd_endo}\n')
                file_endo1.write(f'MEDIAN HD: {median_hd_endo}\n')
                file_endo1.close()

                file_epi.close()
                avg_dice_epi =  '{:.4f}'.format(total_dice_epi / (i + 1)) #rounds
                median_dice_epi = '{:.4f}'.format(np.median(median_dice_list_epi)) #rounds
                avg_hd_epi = '{:.4f}'.format(total_hd_epi / (i + 1)) #rounds
                median_hd_epi = '{:.4f}'.format(np.median(median_hd_list_epi)) #rounds

                file_epi1.write(f'MEAN Dice: {avg_dice_epi}\n')
                file_epi1.write(f'MEDIAN Dice: {median_dice_epi}\n')
                file_epi1.write(f'MEAN HD: {avg_hd_epi}\n')
                file_epi1.write(f'MEDIAN HD: {median_hd_epi}\n')
                file_epi.close()
