import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from .segmentation_losses_LV import DiceHard

def validate_mean_and_median(net, loader, device):
    # Hard dice is used for evaluation
    net.eval()
    mask_type = torch.float32 #if net.output_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    
    tot = 0
    median_list_np = np.array([])
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                preds = net(imgs)
                preds = preds['out']

            loss_eval = DiceHard()
            loss = loss_eval(preds, true_masks)
            
            median_list_np = np.append(median_list_np, loss.item())
            tot += loss.item()
            pbar.update()

    median = np.median(median_list_np)
    mean = tot / n_val
    
    net.train()
    
    return mean, median