import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .segmentation_losses_LV import DiceAndIoUHardWithFPFN, DiceAndIoUHardMedianFix, DiceAndIoUHardWithHD


def validate_mean_and_median(net, loader, device):
    ''' hard dice is used for evaluation '''
    net.eval()
    mask_type = torch.float32 #if net.output_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    
    tot_dice = 0
    tot_iou = 0

    median_list_dice_np = np.array([])
    median_list_iou_np = np.array([])

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                preds = net(imgs)
                #preds = preds['out'] # use if network is from torchvision

            eval = DiceAndIoUHardMedianFix()
            dice, iou, dice_list, iou_list = eval(preds, true_masks)
            
            median_list_dice_np = np.concatenate((median_list_dice_np, dice_list), 0)
            median_list_iou_np = np.concatenate((median_list_iou_np, iou_list), 0)
            tot_dice += dice.item()
            tot_iou += iou.item()
            pbar.update()

    median_dice = np.median(median_list_dice_np)
    median_iou = np.median(median_list_iou_np)
    mean_dice = tot_dice / n_val
    mean_iou = tot_iou / n_val
    
    net.train()
    
    return mean_dice, median_dice, mean_iou, median_iou


def validate_mean_and_median_hd(net, loader, device):
    ''' hard dice is used for evaluation '''
    net.eval()
    mask_type = torch.float32  # if net.output_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    n_total_val = 0  # total single images

    tot_dice = 0
    tot_iou = 0
    tot_hd = 0

    median_list_dice_np = np.array([])
    median_list_iou_np = np.array([])
    median_list_hd_np = np.array([])

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                preds = net(imgs)
                # preds = preds['out'] # use if network is from torchvision

            eval = DiceAndIoUHardWithHD()
            dice, iou, hd, dice_list, iou_list, hd_list, n_in_batch = eval(preds, true_masks)
            n_total_val += n_in_batch

            median_list_dice_np = np.concatenate((median_list_dice_np, dice_list), 0)
            median_list_iou_np = np.concatenate((median_list_iou_np, iou_list), 0)
            median_list_hd_np = np.concatenate((median_list_hd_np, hd_list), 0)
            tot_dice += dice.item()
            tot_iou += iou.item()
            tot_hd += hd
            pbar.update()

    median_dice = np.median(median_list_dice_np)
    median_iou = np.median(median_list_iou_np)
    median_hd = np.median(median_list_hd_np)
    mean_dice = tot_dice / n_total_val
    mean_iou = tot_iou / n_total_val
    mean_hd = tot_hd / n_total_val

    net.train()

    return mean_dice, median_dice, mean_iou, median_iou, mean_hd, median_hd