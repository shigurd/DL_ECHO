import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .segmentation_losses_LV import DiceAndIoUHardWithFPFN

def validate_mean_and_median(net, loader, device):
    ''' hard dice is used for evaluation '''
    net.eval()
    mask_type = torch.float32 #if net.output_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    
    tot_dice = 0
    tot_iou = 0
    tot_fpfn = 0
    median_list_dice_np = np.array([])
    median_list_iou_np = np.array([])
    median_list_fpfn_np = np.array([])
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                preds = net(imgs)
                #preds = preds['out'] # use if network is from torchvision

            eval = DiceAndIoUHardWithFPFN()
            dice, iou, fpfn = eval(preds, true_masks)
            
            median_list_dice_np = np.append(median_list_dice_np, dice.item())
            median_list_iou_np = np.append(median_list_iou_np, iou.item())
            median_list_fpfn_np = np.append(median_list_fpfn_np, fpfn.item())
            tot_dice += dice.item()
            tot_iou += iou.item()
            tot_fpfn += fpfn.item()
            pbar.update()

    median_dice = np.median(median_list_dice_np)
    median_iou = np.median(median_list_iou_np)
    median_fpfn = np.median(median_list_fpfn_np)
    mean_dice = tot_dice / n_val
    mean_iou = tot_iou / n_val
    mean_fpfn = tot_fpfn / n_val
    
    net.train()
    
    return mean_dice, median_dice, mean_iou, median_iou, mean_fpfn, median_fpfn