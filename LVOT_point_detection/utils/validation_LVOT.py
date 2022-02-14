import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .point_losses_LVOT import PixelDSNTDistanceDoubleEval

def validate_mean_and_median_for_distance_and_diameter(net, loader, device):
    net.eval()
    mask_type = torch.float32 #if net.output_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch

    i_tot_abs = 0
    s_tot_abs = 0

    tot_abs = 0
    tot_diam_abs = 0

    median_list_tot_abs_np = np.array([])
    median_list_diam_abs_np = np.array([])

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks_i = batch['mask_i']
            true_masks_s = batch['mask_s']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks_cat = torch.cat((true_masks_i, true_masks_s), 1).to(device=device, dtype=mask_type)

            with torch.no_grad():
                preds = net(imgs)
                #preds = preds['out'] # torchvision syntax

            loss_eval = PixelDSNTDistanceDoubleEval()
            ''' outputs absolute inferior loss, superior loss, total loss and diameter difference '''
            eval_i_abs, eval_s_abs, eval_tot_abs, eval_diameter_abs, eval_tot_list, eval_diam_list = loss_eval(preds, true_masks_cat)

            i_tot_abs += eval_i_abs.item()
            s_tot_abs += eval_s_abs.item()

            tot_abs += eval_tot_abs.item()
            tot_diam_abs += eval_diameter_abs.item()
            median_list_tot_abs_np = np.concatenate((median_list_tot_abs_np, eval_tot_list), 0)
            median_list_diam_abs_np = np.concatenate((median_list_diam_abs_np, eval_diam_list), 0)
            pbar.update()

    mean_i_dist_abs = i_tot_abs / n_val
    mean_s_dist_abs = s_tot_abs / n_val
    mean_tot_dist_abs = tot_abs / n_val
    mean_tot_diam_abs = tot_diam_abs / n_val

    median_tot_dist_abs = np.median(median_list_tot_abs_np)
    median_diam_abs = np.median(median_list_diam_abs_np)

    net.train()
    return mean_i_dist_abs, mean_s_dist_abs, mean_tot_dist_abs, median_tot_dist_abs, mean_tot_diam_abs, median_diam_abs
