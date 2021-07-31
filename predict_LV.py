import logging
import os
import os.path as path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataloader_LV import BasicDataset
from torch.utils.data import DataLoader
from segmentation_losses_LV import DiceHard

import statistics

from torchvision.models.segmentation import fcn_resnet50

def predict_tensor(net,
                img_pil,
                device,
                scale_factor=1,
                out_threshold=0.5,
                mid_systole=False):
    net.eval()

    img_np = BasicDataset.preprocess(img_pil, scale_factor)
    img_np = BasicDataset.convert_to_3ch(img_np, mid_systole)
    
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_tensor)
        output = output['out']
    
    return output

def convert_tensor_mask_to_pil(net, mask_tensor_predicted):
   
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

if __name__ == "__main__":
    
    ''' define model name, prediction dataset and model parameters '''
    model_file = 'Jul30_02-44-11_T-CAMUS1800_V-CAMUS1800_EPOCH_30_LR0.001_BS20_SCL1.pth'
    data_name = 'CAMUS1800'
    scaling = 1
    mask_threshold = 0.5
    mid_systole = False
    compare_with_ground_truth = True
    
    checkpoints_dir = 'checkpoints'
    predicitions_dir = 'predictions'
    model_path = path.join(checkpoints_dir, model_file)
    dir_img = f'data\data_test\imgs_test_{data_name}'
    dir_mask = f'data\data_test\masks_test_{data_name}'
    
    ''' create filenames for output '''
    input_files = os.listdir(dir_img)
    out_files = get_output_filenames(input_files)
    
    ''' define dataloader and network settings '''
    net = fcn_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)
    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    
    ''' load checkpoint data '''
    checkpoint_data = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint_data['model_state_dict'])
    logging.info("Checkpoint loaded !")
    
    ''' make output folder '''
    model_name = model_file.rsplit('.', 1)[0]
    output_dir = path.join(predicitions_dir, model_name)
    os.mkdir(output_dir)
    
    if compare_with_ground_truth == True:
        file = open(path.join(output_dir, f'DICEDATA_{model_name}.txt'), 'w+')
        file1 = open(path.join(output_dir, 'temp.txt'), 'w+')
        file2 = open(path.join(output_dir, 'temp1.txt'), 'w+')
        file2.close()
        
        median_list = []
        total_dice = 0

    for i, fn in enumerate(input_files):
        out_fn = out_files[i]
        logging.info("\nPredicting image {} ...".format(fn))
        img_pil = Image.open(path.join(dir_img, fn))
        img_pil = img_pil.convert('RGB')
        
        ''' predict_tensor returns logits '''
        mask_tensor_predicted = predict_tensor(net=net,
                           img_pil=img_pil,
                           scale_factor=scaling,
                           out_threshold=mask_threshold,
                           device=device,
                           mid_systole=mid_systole)
        
        ''' converting predicted tensor to pil mask '''
        mask_pil_predicted = convert_tensor_mask_to_pil(net, mask_tensor_predicted)
        
        ''' if ground truth is available, make overlays and calculate mean and median dice '''
        if compare_with_ground_truth == True:
            mask_path_true = path.join(dir_mask, f'{fn.rsplit(".", 1)[0]}_mask.png')
            mask_pil_true = Image.open(mask_path_true)
            mask_pil_true = mask_pil_true.convert('L') 
            mask_np_true = BasicDataset.preprocess(mask_pil_true, scaling)
            mask_tensor_true = torch.from_numpy(mask_np_true).cuda() # to cuda as this is loaded with cpu
            
            criterion = DiceHard()
            dice_score = criterion(mask_tensor_predicted, mask_tensor_true).item()
            
            ''' caluculate mean dice and median dice and logging in txt '''
            total_dice += dice_score 
            median_list.append(dice_score) 
            dice4 = '{:.4f}'.format(dice_score) 
            file.write(f'{dice4} \n') 
            file1.write(f'{fn} \n') 
            
            ''' plotting overlays between predicted masks and gt masks '''
            comparison_masks = pil_overlay_predicted_and_gt(mask_pil_true, mask_pil_predicted) 
            ''' plotting overlays between predicted masks and input image '''
            #prediction_on_img = pil_overlay(mask_pil_true.convert('L'), img_pil) 
            
            img_with_comparison = concat_img(img_pil, comparison_masks) 
            img_with_comparison.save(path.join(output_dir, f'{str(dice4).rsplit(".", 1)[1]}_{out_fn}'))
        
        else:
            ''' just save predicted masks '''
            mask_pil_predicted.save(path.join(output_dir, out_fn))
        
        logging.info("Mask saved to {}".format(out_files[i]))

    if compare_with_ground_truth == True:
        file.close()
        file1.close()
        avg_dice = total_dice / (i + 1)
        avg_dice4 = '{:.4f}'.format(avg_dice)[2:] #runder av dice og fjerner 0.
        os.rename(path.join(output_dir, 'temp.txt'), path.join(output_dir, f'AVGDICE_{avg_dice4}_DICEDATA_{model_name}.txt'))
        os.rename(path.join(output_dir, 'temp1.txt'), path.join(output_dir, f'MEDIAN_{statistics.median(median_list)}_DICEDATA_{model_name}.txt'))
    

    