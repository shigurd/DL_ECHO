import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

import numpy as np

from scipy.ndimage import distance_transform_edt

''' all loss functions use logits as input '''

class DiceSoftLoss(nn.Module):
    def __init___(self, smooth=1):
        super(DiceSoftLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            iflat = torch.sigmoid(c[0]).view(-1)
            tflat = c[1].view(-1)
            intersection = (iflat * tflat).sum()

            a_sum = torch.sum(iflat * iflat)
            b_sum = torch.sum(tflat * tflat)

            s += 1 - ((2. * intersection + self.smooth) / (a_sum + b_sum + self.smooth))
        
        return s / (i + 1)

class DiceHard(nn.Module):
    def __init__(self, smooth=1):
        super(DiceHard, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            i_flat = i_flat > 0.5 # hard cutoff
            t_flat = c[1].view(-1)
            intersection = (i_flat * t_flat).sum()

            a_sum = torch.sum(i_flat * i_flat)
            b_sum = torch.sum(t_flat * t_flat)

            s += ((2. * intersection + self.smooth) / (a_sum + b_sum + self.smooth))
        
        return s / (i + 1)


class IoUHard(nn.Module):
    def __init__(self, smooth=1):
        super(IoUHard, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            i_flat = i_flat > 0.5  # hard cutoff
            t_flat = c[1].view(-1)
            intersection = (i_flat * t_flat).sum()

            a_sum = torch.sum(i_flat * i_flat)
            b_sum = torch.sum(t_flat * t_flat)

            s += (intersection + self.smooth) / (a_sum + b_sum - intersection + self.smooth)

        return s / (i + 1)

class DiceSoftBCELoss(nn.Module):
    def __init__(self, smooth=1, dice_weight=1, bce_weight=1):
        super(DiceSoftBCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)

            a_sum = torch.sum(i_flat * i_flat)
            b_sum = torch.sum(t_flat * t_flat)
            
            intersection = (i_flat * t_flat).sum()                            
            dice_loss = 1 - (2. * intersection + self.smooth) / (a_sum + b_sum + self.smooth)  
            
            bce = F.binary_cross_entropy(i_flat, t_flat, reduction='mean')
            
            s += dice_loss * self.dice_weight + bce * self.bce_weight
        
        return s / (i + 1)

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)

            bce = F.binary_cross_entropy(i_flat, t_flat, reduction='mean')
            
            s += bce
        
        return s / (i + 1)

class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        self.smooth = smooth
        super(IoULoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)
            #intersection = (i_flat * t_flat).sum()
            intersection = torch.sum(i_flat * t_flat)

            a_sum = torch.sum(i_flat * i_flat)
            b_sum = torch.sum(t_flat * t_flat)
            iou_loss = 1 - ((intersection + self.smooth) / (a_sum + b_sum - intersection + self.smooth))
            
            s += iou_loss
        
        return s / (i + 1)

class TverskyLoss(nn.Module):
    def __init__(self, smooth=1, tp_weight=0.6, fn_weight=0.4):
        self.smooth = smooth
        self.tp_weight = tp_weight
        self.fn_weight = fn_weight
        super(TverskyLoss, self).__init__()

    def forward(self, input, target):    
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)
            
            #True Positives, False Positives & False Negatives
            tp = (i_flat * t_flat).sum()    
            fp = ((1 - t_flat) * i_flat).sum()
            fn = (t_flat * (1 - i_flat)).sum()
            tversky_loss = 1 -((tp + self.smooth) / (tp + self.tp_weight * fp + self.fn_weight * fn + self.smooth))
            
            s += tversky_loss
            
        return s / (i + 1)

class TverskyBCELoss(nn.Module):
    def __init__(self, smooth=1, tp_weight=0.6, fn_weight=0.4, bce_weight=1, tver_weight=1):
        self.smooth = smooth
        self.tp_weight = tp_weight
        self.fn_weight = fn_weight
        self.bce_weight = bce_weight
        self.tver_weight = tver_weight
        super(TverskyBCELoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)
            
            #True Positives, False Positives & False Negatives
            tp = torch.sum(i_flat * t_flat)
            fp = torch.sum((1 - t_flat) * i_flat)
            fn = torch.sum(t_flat * (1 - i_flat))
            tversky_loss = 1 - ((tp + smooth) / (tp + self.tp_weight * fp + self.fn_weight * fn + self.smooth))
            
            bce = F.binary_cross_entropy(i_flat, t_flat, reduction='mean')
            
            s += tversky_loss * self.tver_weight + bce * self.bce_weight
        
        return s / (i + 1)
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        if inputs.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(inputs, targets)):    
            bce_loss = F.binary_cross_entropy(torch.sigmoid(c[0]), c[1], reduce=False)
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

            if self.reduce:
                s += torch.mean(focal_loss)
            else:
                s += focal_loss
        return s / (i + 1)

class WeightedDiceSoftBCELoss(nn.Module):
    def __init__(self, smooth=1):
        self.smooth = smooth
        super(WeightedDiceSoftBCELoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            
            #weights
            pool = F.avg_pool2d(c[1], kernel_size=5, padding=2, stride=1) #settings contour
            #pool = F.avg_pool2d(c[1], kernel_size=9, padding=4, stride=1) #settings core
            outline = (pool.ge(0.01) * pool.le(0.99)).float() #outline contour
            #outline = pool.le(0.99) * -1 + 1 #outline core
            weights = Variable(torch.ones(pool.size()).cuda())
            w0 = weights.sum()
            weights = weights + outline * 2
            w1 = weights.sum()
            weights = weights / w1 * w0
            
            #bce
            wflat = weights.view(-1)
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)
            bce = wflat * i_flat.clamp(min=0) - wflat * i_flat * t_flat + wflat * torch.log(1 + torch.exp(-i_flat.abs()))
            bce = bce.sum() / wflat.sum()
            
            #dice
            w_sqr = wflat * wflat
            d_i_flat = torch.sigmoid(c[0]).view(-1)
            d_t_flat = c[1].view(-1)
            intersection = d_i_flat * d_t_flat
            dice_loss = 1 - (2. * ((w_sqr * intersection).sum() + self.smooth) / ((w_sqr * d_i_flat).sum() + (w_sqr * d_t_flat).sum() + self.smooth))

            s += dice_loss + bce
            
        return s / (i + 1)


def compute_edts_forhdloss(mask):
    res = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        posmask = mask[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res


class HDDTBinaryLoss(nn.Module):
    def __init__(self):
        super(HDDTBinaryLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            input = torch.sigmoid(c[0]).type(torch.float32)
            target = c[1].type(torch.float32)
            
            with torch.no_grad():
                input_dist = compute_edts_forhdloss(input.cpu().numpy()>0.5)
                target_dist = compute_edts_forhdloss(target.cpu().numpy()>0.5)
    
            pred_error = (target - input)**2
            dist = input_dist**2 + target_dist**2 # \alpha=2 in eq(8)

            dist = torch.from_numpy(dist)
            if dist.device != pred_error.device:
                dist = dist.to(pred_error.device).type(torch.float32)

            multipled = torch.einsum("bxy,bxy->bxy", pred_error, dist)
            hd_loss = multipled.mean()
            
            s += hd_loss

        return s / (i + 1)
        
class HDDiceSoftBCELoss(nn.Module):
    def __init__(self, smooth=1, hd_weight=0.5, dice_weight=1, bce_weight=1):
        super(HDDiceSoftBCELoss, self).__init__()
        self.smooth = smooth
        self.hd_weight = hd_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for i, c in enumerate(zip(input, target)):
            i_flat = torch.sigmoid(c[0]).view(-1)
            t_flat = c[1].view(-1)
            
            #dice
            a_sum = torch.sum(i_flat * i_flat)
            b_sum = torch.sum(t_flat * t_flat)
            
            intersection = (i_flat * t_flat).sum()                            
            dice_loss = 1 - (2. * intersection + self.smooth) / (a_sum + b_sum + self.smooth)  
            
            #bce
            bce = F.binary_cross_entropy(i_flat, t_flat, reduction='mean')
            
            #HD
            hd_input = torch.sigmoid(c[0]).type(torch.float32)
            hd_target = c[1].type(torch.float32)
            
            with torch.no_grad():
                input_dist = compute_edts_forhdloss(hd_input.cpu().numpy()>0.5)
                target_dist = compute_edts_forhdloss(hd_target.cpu().numpy()>0.5)
    
            pred_error = (hd_target - hd_input)**2
            dist = input_dist**2 + target_dist**2 

            dist = torch.from_numpy(dist)
            if dist.device != pred_error.device:
                dist = dist.to(pred_error.device).type(torch.float32)

            multipled = torch.einsum("cxy,cxy->cxy", pred_error, dist)
            hd_loss = multipled.mean()
            
            s += hd_loss * self.hd_weight + dice_loss * self.dice_weight + bce * self.bce_weight 
        
        return s / (i + 1)


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask
        
        res[i] = pos_edt/np.max(pos_edt) + neg_edt/np.max(neg_edt)
    return res

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        if logits.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()
        
        for x, c in enumerate(zip(logits, gt)):
            logits = torch.sigmoid(c[0])
            gt = c[1]
            
            logits = softmax_helper(logits)
            # one hot code for gt
            with torch.no_grad():
                if len(logits.shape) != len(gt.shape):
                    gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

                if all([i == j for i, j in zip(logits.shape, gt.shape)]):
                    # if this is the case then gt is probably already a one hot encoding
                    y_onehot = gt
                else:
                    gt = gt.long()
                    y_onehot = torch.zeros(logits.shape)
                    if logits.device.type == "cuda":
                        y_onehot = y_onehot.cuda(logits.device.index)
                    y_onehot.scatter_(1, gt, 1)
            
            gt_temp = gt.type(torch.float32)
            with torch.no_grad():
                dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
            # print('dist.shape: ', dist.shape)
            dist = torch.from_numpy(dist)

            if dist.device != logits.device:
                dist = dist.to(logits.device).type(torch.float32)
            
            tp = logits * y_onehot
            tp = torch.sum(tp * dist, (1,2))
            
            dc = (2 * tp + self.smooth) / (torch.sum(logits, (1,2)) + torch.sum(y_onehot, (1,2)) + self.smooth)

            dc = -dc.mean()
            s += dc

        return s / (x + 1)




        
        
        