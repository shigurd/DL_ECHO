import logging
import os
import os.path as path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.dataloader_LV import BasicDataset
from utils.convert_myomask_to_endomask_and_epimask_LV import get_endocard_epicard_from_np
from utils.segmentation_losses_LV import DiceHard

#from torchvision.models.segmentation import fcn_resnet50
sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50

def predict_tensor(net,
                img_pil,
                device,
                scale_factor=1,
                mid_systole=False):
    net.eval()

    img_np = BasicDataset.preprocess(img_pil, scale_factor)
    img_np = BasicDataset.extract_midsystole(img_np, mid_systole)
    
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_tensor)
        output = output['out']
    
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


def pil_overlay(foreground, background):
    img1 = foreground.convert("RGBA")
    img2 = background.convert("RGBA")
    
    overlay = Image.blend(img2, img1, alpha=.1)
    
    return overlay


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
    model_file = 'Aug28_14-10-36_RES50_DICBCE_ADAM_T-GE1956_HMLHML_MA4_V-_EP30_LR0.001_BS20_SCL1.pth'
    data_name = 'GE1956_HMLHML'
    n_channels = 1
    n_classes = 1
    scaling = 1
    mask_threshold = 0.5
    mid_systole = True
    compare_with_ground_truth = False
    convert_to_epicard_and_endocard = False

    model_path = path.join('checkpoints', model_file)
    dir_img = path.join('data', 'test', 'imgs', data_name)
    dir_mask = path.join('data', 'test', 'masks', data_name)

    ''' make output dir '''
    if compare_with_ground_truth == True:
        model_name = f'{model_file.rsplit(".", 1)[0]}_VAL'
    else:
        model_name = f'{model_file.rsplit(".", 1)[0]}_OUT'

    predictions_output = path.join('predictions', model_name)
    os.mkdir(predictions_output)
    
    ''' create filenames for output '''
    input_files = os.listdir(dir_img)
    out_files = get_output_filenames(input_files)
    
    ''' define network settings '''
    net = fcn_resnet50(pretrained=False, progress=True, in_channels=n_channels, num_classes=n_classes, aux_loss=None)
    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    
    ''' load checkpoint data '''
    checkpoint_data = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint_data['model_state_dict'])
    logging.info("Checkpoint loaded !")
    
    if compare_with_ground_truth == True:
        file = open(path.join(predictions_output, f'DICEDATA_{model_name}.txt'), 'w+')
        file.write('file_name,dice_score\n')
        file1 = open(path.join(predictions_output, 'temp.txt'), 'w+')
        file1.close()
        file2 = open(path.join(predictions_output, 'temp1.txt'), 'w+')
        file2.close()

        median_list = np.array([])
        total_dice = 0

        if convert_to_epicard_and_endocard == True:
            endo_output = path.join(predictions_output, 'endocard')
            os.mkdir((endo_output))
            file_endo = open(path.join(endo_output, f'DICEDATA_{model_name}_ENDO.txt'), 'w+')
            file.write('file_name_endo,dice_score\n')
            file_endo1 = open(path.join(endo_output, 'temp_endo.txt'), 'w+')
            file_endo1.close()
            file_endo2 = open(path.join(endo_output, 'temp_endo1.txt'), 'w+')
            file_endo2.close()

            median_list_endo = np.array([])
            total_dice_endo = 0

            epi_output = path.join(predictions_output, 'epicard')
            os.mkdir(epi_output)
            file_epi = open(path.join(epi_output, f'DICEDATA_{model_name}_EPI.txt'), 'w+')
            file.write('file_name_epi,dice_score\n')
            file_epi1 = open(path.join(epi_output, 'temp_epi.txt'), 'w+')
            file_epi1.close()
            file_epi2 = open(path.join(epi_output, 'temp_epi1.txt'), 'w+')
            file_epi2.close()

            median_list_epi = np.array([])
            total_dice_epi = 0


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
                               mid_systole=mid_systole)

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

                ''' calculate mean dice and median dice and logging in txt '''
                total_dice += dice_score
                median_list = np.append(median_list, dice_score)
                dice4 = '{:.4f}'.format(dice_score)
                file.write(f'{fn},{dice4} \n')

                ''' plotting overlays between predicted masks and gt masks '''
                comparison_masks = pil_overlay_predicted_and_gt(mask_pil_true, mask_pil_predicted)
                ''' plotting overlays between predicted masks and input image '''
                #prediction_on_img = pil_overlay(mask_pil_true.convert('L'), img_pil)

                img_with_comparison = concat_img(img_pil, comparison_masks)
                img_with_comparison.save(path.join(predictions_output, f'{str(dice4).rsplit(".", 1)[1]}_{out_fn}'))

                if convert_to_epicard_and_endocard == True:
                    endocard_tensor_predicted, epicard_tensor_predicted, endocard_pil_predicted, epicard_pil_predicted = endocard_epicard_to_tensor(mask_pil_predicted)
                    endocard_tensor_true, epicard_tensor_true, endocard_pil_true, epicard_pil_true = endocard_epicard_to_tensor(mask_pil_true)

                    dice_score_endo = criterion(endocard_tensor_predicted, endocard_tensor_true).item()
                    total_dice_endo += dice_score_endo
                    median_list_endo = np.append(median_list_endo, dice_score_endo)
                    dice_endo4 = '{:.4f}'.format(dice_score_endo)
                    file_endo.write(f'{fn},{dice_endo4} \n')
                    comparison_masks_endo = pil_overlay_predicted_and_gt(endocard_pil_true, endocard_pil_predicted)
                    img_with_comparison_endo = concat_img(img_pil, comparison_masks_endo)
                    img_with_comparison_endo.save(path.join(endo_output, f'{str(dice_endo4).rsplit(".", 1)[1]}_{out_fn}'))

                    dice_score_epi = criterion(epicard_tensor_predicted, epicard_tensor_true).item()
                    total_dice_epi += dice_score_epi
                    median_list_epi = np.append(median_list_epi, dice_score_epi)
                    dice_epi4 = '{:.4f}'.format(dice_score_epi)
                    file_epi.write(f'{fn},{dice_epi4} \n')
                    comparison_masks_epi = pil_overlay_predicted_and_gt(epicard_pil_true, epicard_pil_predicted)
                    img_with_comparison_epi = concat_img(img_pil, comparison_masks_epi)
                    img_with_comparison_epi.save(path.join(epi_output, f'{str(dice_epi4).rsplit(".", 1)[1]}_{out_fn}'))

            else:
                ''' just save predicted masks '''
                mask_pil_predicted.save(path.join(predictions_output, out_fn))

            logging.info("Mask saved to {}".format(out_files[i]))
            pbar.update()

        if compare_with_ground_truth == True:
            file.close()
            avg_dice = total_dice / (i + 1)
            avg_dice4 = '{:.4f}'.format(avg_dice)[2:] #runder av dice og fjerner 0.
            os.rename(path.join(predictions_output, 'temp.txt'), path.join(predictions_output, f'AVGDICE_{avg_dice4}_DICEDATA_{model_name}.txt'))
            os.rename(path.join(predictions_output, 'temp1.txt'), path.join(predictions_output, f'MEDIAN_{np.median(median_list)}_DICEDATA_{model_name}.txt'))

            if convert_to_epicard_and_endocard == True:
                file_endo.close()
                avg_dice_endo = total_dice_endo / (i + 1)
                avg_dice_endo4 = '{:.4f}'.format(avg_dice_endo)[2:]  # runder av dice og fjerner 0.
                os.rename(path.join(endo_output, 'temp_endo.txt'),
                          path.join(endo_output, f'AVGDICE_{avg_dice_endo4}_DICEDATA_{model_name}_ENDO.txt'))
                os.rename(path.join(endo_output, 'temp_endo1.txt'),
                          path.join(endo_output, f'MEDIAN_{np.median(median_list_endo)}_DICEDATA_{model_name}_ENDO.txt'))

                file_epi.close()
                avg_dice_epi = total_dice_epi / (i + 1)
                avg_dice_epi4 = '{:.4f}'.format(avg_dice_epi)[2:]  # runder av dice og fjerner 0.
                os.rename(path.join(epi_output, 'temp_epi.txt'),
                          path.join(epi_output, f'AVGDICE_{avg_dice_epi4}_DICEDATA_{model_name}_EPI.txt'))
                os.rename(path.join(epi_output, 'temp_epi1.txt'),
                          path.join(epi_output, f'MEDIAN_{np.median(median_list_epi)}_DICEDATA_{model_name}_EPI.txt'))