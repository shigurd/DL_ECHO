import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

import numpy as np

''' all loss functions use logits as input '''

class DSNTDoubleLoss(nn.Module):
    def __init__(self):
        super(DSNTDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_mse = mse_loss(pred_coords, true_coords)

                s += ed_loss

        return s / (i + 1)


class DSNTDistanceDoubleLoss(nn.Module):
    def __init__(self):
        super(DSNTDistanceDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_mse = mse_loss(pred_coords, true_coords)

                s += ed_loss

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_dist = torch.sqrt(torch.sum((pred_dist_list_array[0] - pred_dist_list_array[1]) ** 2))
            true_dist = torch.sqrt(torch.sum((true_dist_list_array[0] - true_dist_list_array[1]) ** 2))
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2)

            s += diff_dist_abs

        return s / (i + 1)


class DistanceDoubleLoss(nn.Module):
    def __init__(self):
        super(DistanceDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_dist = torch.sqrt(torch.sum((pred_dist_list_array[0] - pred_dist_list_array[1]) ** 2))
            true_dist = torch.sqrt(torch.sum((true_dist_list_array[0] - true_dist_list_array[1]) ** 2))
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2)

            s += diff_dist_abs

        return s / (i + 1)


class DSNTDistanceAngleDoubleLoss(nn.Module):
    def __init__(self):
        super(DSNTDistanceAngleDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' possible to add MSE to ED loss '''
                # pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                # true_coords = torch.stack((true_x_coord, true_y_coord))
                # coordinate_mse = mse_loss(pred_coords, true_coords)

                s += ed_loss

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_vector = pred_dist_list_array[0] - pred_dist_list_array[1]
            true_vector = true_dist_list_array[0] - true_dist_list_array[1]

            pred_dist = torch.sqrt(torch.sum(pred_vector ** 2))
            true_dist = torch.sqrt(torch.sum(true_vector ** 2))
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2)

            s += diff_dist_abs

            ''' calculate angle between predicted and true vector '''
            cos_distance = 1 - torch.cos(torch.dot(pred_vector, true_vector) / (pred_dist * true_dist))
            s += cos_distance

            ''' option to convert angle output to radians or angles instad of cosine distance '''
            #angle_rad = torch.acos(torch.dot(pred_vector, true_vector) / (pred_dist_abs * true_dist_abs))
            #pi = torch.acos(torch.Tensor([-1])).cuda()
            #angle_rad_norm = angle_rad / pi
            #angle_deg = angle_rad * (180 / pi)

        return s / (i + 1)


class MSEDSNTDoubleLoss(nn.Module):
    def __init__(self):
        super(MSEDSNTDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' calculates softmax for ground truth '''
                true_mask_softmax = softmax(points[1].view(-1)).view(points[1].shape)

                mse_loss = nn.MSELoss()
                mse_loss = mse_loss(pred_mask_softmax, true_mask_softmax)

                ''' option to add MSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_mse = mse_loss(pred_coords, true_coords)

                s += mse_loss + ed_loss

        return s / (i + 1)


class L1DSNTDoubleLoss(nn.Module):
    def __init__(self):
        super(L1DSNTDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' calculates softmax for ground truth '''
                true_mask_softmax = softmax(points[1].view(-1)).view(points[1].shape)

                l1_loss = nn.L1Loss()
                l1_loss = l1_loss(pred_mask_softmax, true_mask_softmax)

                ''' option to add L1 to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_l1 = l1_loss(pred_coords, true_coords)

                s += l1_loss + ed_loss

        return s / (i + 1)


class RMSEDSNTDoubleLoss(nn.Module):
    def __init__(self):
        super(RMSEDSNTDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' calculates softmax for ground truth '''
                true_mask_softmax = softmax(points[1].view(-1)).view(points[1].shape)

                mse_loss = nn.MSELoss()
                rmse_loss = torch.sqrt(mse_loss(pred_mask_softmax, true_mask_softmax))

                ''' option to add RMSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_rmse = torch.sqrt(mse_loss(pred_coords, true_coords))

                s += rmse_loss + ed_loss

        return s / (i + 1)


class MSEDSNTDistanceDoubleLoss(nn.Module):
    def __init__(self):
        super(MSEDSNTDistanceDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' calculates softmax for ground truth '''
                true_mask_softmax = softmax(points[1].view(-1)).view(points[1].shape)

                mse_loss = nn.MSELoss()
                mse_loss = mse_loss(pred_mask_softmax, true_mask_softmax)

                ''' possible to add MSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_mse = mse_loss(pred_coords, true_coords)

                s += mse_loss + ed_loss

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)


            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_dist = torch.sqrt(torch.sum((pred_dist_list_array[0] - pred_dist_list_array[1]) ** 2))
            true_dist = torch.sqrt(torch.sum((true_dist_list_array[0] - true_dist_list_array[1]) ** 2))
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2)

            s += diff_dist_abs

        return s / (i + 1)


class MSEDSNTDistanceAngleDoubleLoss(nn.Module):
    def __init__(self):
        super(MSEDSNTDistanceAngleDoubleLoss, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' possible to add MSE to ED loss '''
                # pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                # true_coords = torch.stack((true_x_coord, true_y_coord))
                # coordinate_mse = mse_loss(pred_coords, true_coords)

                ''' calculates softmax for ground truth '''
                true_mask_softmax = softmax(points[1].view(-1)).view(points[1].shape)

                mse_loss = nn.MSELoss()
                mse_loss = mse_loss(pred_mask_softmax, true_mask_softmax)

                s += mse_loss + ed_loss

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_vector = pred_dist_list_array[0] - pred_dist_list_array[1]
            true_vector = true_dist_list_array[0] - true_dist_list_array[1]

            pred_dist = torch.sqrt(torch.sum(pred_vector ** 2))
            true_dist = torch.sqrt(torch.sum(true_vector ** 2))
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2)

            s += diff_dist_abs

            ''' calculate angle between predicted and true vector '''
            cos_distance = 1 - torch.cos(torch.dot(pred_vector, true_vector) / (pred_dist * true_dist))
            s += cos_distance * 0.1

            ''' option to convert angle output to radians or angles instad of cosine distance '''
            #angle_rad = torch.acos(torch.dot(pred_vector, true_vector) / (pred_dist_abs * true_dist_abs))
            #pi = torch.acos(torch.Tensor([-1])).cuda()
            #angle_rad_norm = angle_rad / pi
            #angle_deg = angle_rad * (180 / pi)

        return s / (i + 1)


''' this function converts the normalized network values back into pixel values '''
def coords_norm_to_pixel(coords_tensor_norm, x_size, y_size):
    x_pixel = coords_tensor_norm[0] * x_size
    y_pixel = coords_tensor_norm[1] * y_size

    return x_pixel, y_pixel


class PixelDSNTDoubleEval(nn.Module):
    def __init__(self):
        super(PixelDSNTDoubleEval, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' converts normalized values to pixel '''
                pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                true_coords = torch.stack((true_x_coord, true_y_coord))

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel(pred_coords, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel(true_coords, x_size, y_size)

                ed_loss_pixel = torch.sqrt(
                    (true_x_coord_pixel - pred_x_coord_pixel) ** 2 + (true_y_coord_pixel - pred_y_coord_pixel) ** 2)

                ''' option to use normalized values '''
                #ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                if o == 0:
                    s_i += ed_loss_pixel
                else:
                    s_s += ed_loss_pixel

        ''' outputs absolute inferior loss, superior loss and total loss '''
        return s_i / (i + 1), s_s / (i + 1), (s_i + s_s) / (i + 1)


class PixelDSNTDistanceDoubleEval(nn.Module):
    def __init__(self):
        super(PixelDSNTDistanceDoubleEval, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
            s_diam_diff_abs = torch.FloatTensor(1).cuda().zero_()
        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()
            s_diam_diff_abs = torch.FloatTensor(1).cuda().zero_()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' converts normalized values to pixel '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel(pred_coords_stack, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel(true_coords_stack, x_size, y_size)

                ed_loss_pixel = torch.sqrt(
                    (true_x_coord_pixel - pred_x_coord_pixel) ** 2 + (true_y_coord_pixel - pred_y_coord_pixel) ** 2)

                ''' option to use normalized values '''
                # ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' add pixel points to distance calculation tensor '''
                pred_coords_stack_pixel = torch.stack((pred_x_coord_pixel, pred_y_coord_pixel))
                true_coords_stack_pixel = torch.stack((true_x_coord_pixel, true_y_coord_pixel))

                pred_dist_list_array.append(pred_coords_stack_pixel)
                true_dist_list_array.append(true_coords_stack_pixel)

                if o == 0:
                    s_i += ed_loss_pixel
                else:
                    s_s += ed_loss_pixel

            ''' outputs absolute pixelwise distance '''
            vector_pred = pred_dist_list_array[0] - pred_dist_list_array[1]
            vector_true = true_dist_list_array[0] - true_dist_list_array[1]

            pred_distance = torch.sqrt(torch.sum(vector_pred ** 2))
            true_distance = torch.sqrt(torch.sum(vector_true ** 2))
            diff_distance_abs = torch.sqrt((pred_distance - true_distance) ** 2)

            #print('diff distance:', diff_distance)
            s_diam_diff_abs += diff_distance_abs

        ''' outputs absolute inferior loss, superior loss, total loss and diameter difference '''
        return s_i / (i + 1), s_s / (i + 1), (s_i + s_s) / (i + 1), s_diam_diff_abs / (i + 1)


''' not that this only works when images are predicted with batch size of 1 '''
class PixelDSNTDistanceDoublePredict(nn.Module):
    def __init__(self):
        super(PixelDSNTDistanceDoublePredict, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
            s_diam_diff = torch.FloatTensor(1).cuda().zero_()
        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()
            s_diam_diff = torch.FloatTensor(1).zero_()

        pred_coordinate_list = []
        true_coordinate_list = []

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                x_size = points[0].shape[-1]
                y_size = points[0].shape[-2]

                x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
                y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

                for p in range(y_size):
                    y_soft_argmax[p, :] = (p + 1) / y_size

                for j in range(x_size):
                    x_soft_argmax[:, j] = (j + 1) / x_size

                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()

                ''' converts normalized values to pixel '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel(pred_coords_stack, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel(true_coords_stack, x_size, y_size)

                ''' returns the actual coordinate which requires a -1 since it is an index '''
                pred_coordinate_list.append([pred_x_coord_pixel.item() - 1, pred_y_coord_pixel.item() - 1])
                true_coordinate_list.append([true_x_coord_pixel.item() - 1, true_y_coord_pixel.item() - 1])

                ed_loss_pixel = torch.sqrt(
                    (true_x_coord_pixel - pred_x_coord_pixel) ** 2 + (true_y_coord_pixel - pred_y_coord_pixel) ** 2)

                ''' option to use normalized values '''
                # ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' add pixel points to distance calculation tensor '''
                pred_coords_stack_pixel = torch.stack((pred_x_coord_pixel, pred_y_coord_pixel))
                true_coords_stack_pixel = torch.stack((true_x_coord_pixel, true_y_coord_pixel))

                pred_dist_list_array.append(pred_coords_stack_pixel)
                true_dist_list_array.append(true_coords_stack_pixel)

                if o == 0:
                    s_i += ed_loss_pixel
                else:
                    s_s += ed_loss_pixel

            ''' outputs absolute pixelwise distance '''
            vector_pred = pred_dist_list_array[0] - pred_dist_list_array[1]
            vector_true = true_dist_list_array[0] - true_dist_list_array[1]

            pred_distance = torch.sqrt(torch.sum(vector_pred ** 2))
            true_distance = torch.sqrt(torch.sum(vector_true ** 2))
            diff_distance = pred_distance - true_distance

            #print(f'loss lvot diam pix {pred_distance.item()}')
            #print(f'loss lvot diam diff pix {diff_distance.item()}')

            s_diam_diff += diff_distance

        ''' outputs absolute inferior loss, superior loss, total loss, diameter difference and coordiante lists for pred and gt '''
        return s_i / (i + 1), s_s / (i + 1), (s_i + s_s) / (i + 1), s_diam_diff / (i + 1), pred_coordinate_list, true_coordinate_list
