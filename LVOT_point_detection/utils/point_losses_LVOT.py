import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

import numpy as np

''' all loss functions use logits as input '''

def jensen_shannon_divergence(input, target):
    input = input.view(-1)
    target = target.view(-1)

    kl = nn.KLDivLoss()

    m = 0.5 * (input + target)
    kl_im = kl(input, m)
    kl_tm = kl(target, m)

    jsd = 0.5 * kl_im + 0.5 * kl_tm

    return jsd


def coordinate_map_sigmoid_range(y_size, x_size):
    ''' make the evenly spaced x and y coordinate masks '''
    x_soft_argmax = torch.zeros((y_size, x_size))
    y_soft_argmax = torch.zeros((y_size, x_size))

    for p in range(y_size):
        y_soft_argmax[p, :] = (p + 1) / y_size

    for j in range(x_size):
        x_soft_argmax[:, j] = (j + 1) / x_size

    ''' remember to add output to cuda '''
    return y_soft_argmax, x_soft_argmax


def coordinate_map_tanh_range_half(y_size, x_size):
    ''' make the evenly spaced x and y coordinate masks '''
    x_soft_argmax = torch.zeros((y_size, x_size))
    y_soft_argmax = torch.zeros((y_size, x_size))

    half_y = y_size / 2
    half_x = x_size / 2

    for p in range(y_size):
        y_soft_argmax[p, :] = (p + 1 - half_y) / y_size

    for j in range(x_size):
        x_soft_argmax[:, j] = (j + 1 - half_x) / x_size

    ''' remember to add output to cuda '''
    return y_soft_argmax, x_soft_argmax


def coordinate_map_tanh_range_full(y_size, x_size):
    ''' make the evenly spaced x and y coordinate masks '''
    x_soft_argmax = torch.zeros((y_size, x_size))
    y_soft_argmax = torch.zeros((y_size, x_size))

    for p in range(y_size):
        y_soft_argmax[p, :] = (2 * (p + 1) - (y_size + 1)) / y_size

    for j in range(x_size):
        x_soft_argmax[:, j] = (2 * (j + 1) - (x_size + 1)) / x_size

    ''' remember to add output to cuda '''
    return y_soft_argmax, x_soft_argmax


class DSNTDoubleLossNew(nn.Module):
    def __init__(self):
        super(DSNTDoubleLossNew, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        #y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid_range(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                #true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                #true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                #true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                #true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                s += ed_loss

        return s / (i + 1)

class DSNTDoubleLossNewMSECnoED(nn.Module):
    def __init__(self, x_weight=1, y_weight=1):
        self.x_weight = x_weight
        self.y_weight = y_weight
        super(DSNTDoubleLossNewMSECnoED, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        # y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                # true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                # true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' option to add MSE to ED loss '''
                mse_loss = nn.MSELoss()
                pred_coords = torch.stack((pred_x_coord * self.x_weight, pred_y_coord * self.y_weight))
                true_coords = torch.stack((true_x_coord * self.x_weight, true_y_coord * self.y_weight))
                coordinate_mse = mse_loss(pred_coords, true_coords)

                s += coordinate_mse

        return s / (i + 1)


class DSNTDoubleLossNewMSEC(nn.Module):
    def __init__(self, x_weight=1, y_weight=1):
        self.x_weight = x_weight
        self.y_weight = y_weight
        super(DSNTDoubleLossNewMSEC, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                #true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                #true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                mse_loss = nn.MSELoss()
                pred_coords = torch.stack((pred_x_coord * self.x_weight, pred_y_coord * self.y_weight))
                true_coords = torch.stack((true_x_coord * self.x_weight, true_y_coord * self.y_weight))
                coordinate_mse = mse_loss(pred_coords, true_coords)

                s += ed_loss + coordinate_mse

        return s / (i + 1)


class DSNTJSDDoubleLossNew(nn.Module):
    def __init__(self):
        super(DSNTJSDDoubleLossNew, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        # y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):
            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                # true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                # true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                #pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                #true_coords = torch.stack((true_x_coord, true_y_coord))
                #coordinate_mse = mse_loss(pred_coords, true_coords)

                ''' jsd for gt masks and logits '''
                jsd = jensen_shannon_divergence(points[1], pred_mask_softmax)

                s += ed_loss + jsd

        return s / (i + 1)


class DSNTJSDDistanceDoubleLossNew(nn.Module):
    def __init__(self):
        super(DSNTJSDDistanceDoubleLossNew, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        # y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                # true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                # true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                # pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                # true_coords = torch.stack((true_x_coord, true_y_coord))
                # coordinate_mse = mse_loss(pred_coords, true_coords)

                ''' jsd for gt masks and logits '''
                jsd = jensen_shannon_divergence(points[1], pred_mask_softmax)

                s += ed_loss + jsd

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_dist = torch.sqrt(torch.sum((pred_dist_list_array[0] - pred_dist_list_array[1]) ** 2))
            true_dist = torch.sqrt(torch.sum((true_dist_list_array[0] - true_dist_list_array[1]) ** 2))

            ''' dist error proportion of relative gt dist, countermeasure against exploding gradient '''
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / true_dist

            ''' dist error proportion of max distance in 256x256, countermeasure against exploding gradient '''
            #diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / (torch.sqrt(torch.Tensor([x_size]).cuda() ** 2 + torch.Tensor([y_size]).cuda() ** 2) - true_dist)

            s += diff_dist_abs

        return s / (i + 1)


class DSNTDistanceDoubleLossNew(nn.Module):
    def __init__(self):
        super(DSNTDistanceDoubleLossNew, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        # y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                # true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                # true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                mse_loss = nn.MSELoss()
                pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                true_coords = torch.stack((true_x_coord, true_y_coord))
                coordinate_mse = mse_loss(pred_coords, true_coords)

                s += ed_loss + coordinate_mse

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_dist = torch.sqrt(torch.sum((pred_dist_list_array[0] - pred_dist_list_array[1]) ** 2))
            true_dist = torch.sqrt(torch.sum((true_dist_list_array[0] - true_dist_list_array[1]) ** 2))

            ''' dist error proportion of relative gt dist, countermeasure against exploding gradient '''
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / true_dist

            ''' dist error proportion of max distance in 256x256, countermeasure against exploding gradient '''
            #diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / (torch.sqrt(torch.Tensor([x_size]).cuda() ** 2 + torch.Tensor([y_size]).cuda() ** 2) - true_dist)

            s += diff_dist_abs

        return s / (i + 1)


class DSNTJSDDistAnglDoubleLossNew(nn.Module):
    def __init__(self):
        super(DSNTJSDDistAnglDoubleLossNew, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        # y_soft_argmax, x_soft_argmax = coodinate_map_sigmoid(y_size, x_size)
        y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        # y_soft_argmax, x_soft_argmax = coodinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember to convert euqlician coords to tanh '''
                coord_argmax = torch.argmax(points[1]).detach()
                # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                # true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                # true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' euclidian distance with DSNT, ED is naturally a loss since distance should be minimized '''
                ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' option to add MSE to ED loss '''
                # pred_coords = torch.stack((pred_x_coord, pred_y_coord))
                # true_coords = torch.stack((true_x_coord, true_y_coord))
                # coordinate_mse = mse_loss(pred_coords, true_coords)

                ''' jsd for gt masks and logits '''
                jsd = jensen_shannon_divergence(points[1], pred_mask_softmax)

                s += ed_loss + jsd

                ''' add point to distance calculation tensor '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_dist_list_array.append(pred_coords_stack)
                true_dist_list_array.append(true_coords_stack)

            ''' calculate true absolute distance and predicted absolute distance to find the absolute difference '''
            pred_vector = pred_dist_list_array[0] - pred_dist_list_array[1]
            true_vector = true_dist_list_array[0] - true_dist_list_array[1]

            pred_dist = torch.sqrt(torch.sum((pred_vector) ** 2))
            true_dist = torch.sqrt(torch.sum((true_vector) ** 2))

            ''' dist error proportion of relative gt dist, countermeasure against exploding gradient '''
            diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / true_dist

            ''' dist error proportion of max distance in 256x256, countermeasure against exploding gradient '''
            #diff_dist_abs = torch.sqrt((pred_dist - true_dist) ** 2) / (torch.sqrt(torch.Tensor([x_size]).cuda() ** 2 + torch.Tensor([y_size]).cuda() ** 2) - true_dist)

            s += diff_dist_abs

            ''' angle error between predicted and true vector '''
            cos_distance = 1 - torch.cos(torch.dot(true_vector, true_dist_list_array[0] - true_vector) / (pred_dist * true_dist))

            s += cos_distance

        return s / (i + 1)


''' this function converts the normalized network values back into pixel values '''
def coords_norm_to_pixel_sigmoid(coords_tensor_norm, x_size, y_size):
    x_pixel = coords_tensor_norm[0] * x_size
    y_pixel = coords_tensor_norm[1] * y_size

    return x_pixel, y_pixel


def coords_norm_to_pixel_tanh_half(coords_tensor_norm, x_size, y_size):
    x_pixel = coords_tensor_norm[0] * x_size + x_size / 2
    y_pixel = coords_tensor_norm[1] * y_size + y_size / 2

    return x_pixel, y_pixel


def coords_norm_to_pixel_tanh_full(coords_tensor_norm, x_size, y_size):
    x_pixel = (coords_tensor_norm[0] * x_size + (x_size + 1)) / 2
    y_pixel = (coords_tensor_norm[1] * y_size + (y_size + 1)) / 2

    return x_pixel, y_pixel


class PixelDSNTDistanceDoubleEval(nn.Module):
    def __init__(self):
        super(PixelDSNTDistanceDoubleEval, self).__init__()

    def forward(self, input, target):

        tot_list_np = np.array([])
        diam_list_np = np.array([])

        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
            s_diam_diff_abs = torch.FloatTensor(1).cuda().zero_()
            s_x = torch.FloatTensor(1).cuda().zero_()
            s_y = torch.FloatTensor(1).cuda().zero_()
        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()
            s_diam_diff_abs = torch.FloatTensor(1).zero_()
            s_x = torch.FloatTensor(1).zero_()
            s_y = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid_range(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []
            tot_ed_for_median = 0

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember this must match the range of the CC '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                #true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                #true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                #true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                #true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' converts normalized values to pixel '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_sigmoid(pred_coords_stack, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_sigmoid(true_coords_stack, x_size, y_size)
                #pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_tanh_half(pred_coords_stack, x_size, y_size)
                #true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_tanh_half(true_coords_stack, x_size, y_size)
                #pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_tanh_full(pred_coords_stack, x_size, y_size)
                #true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_tanh_full(true_coords_stack, x_size, y_size)

                x_diff = true_x_coord_pixel - pred_x_coord_pixel
                y_diff = true_y_coord_pixel - pred_y_coord_pixel
                s_x += torch.sqrt(x_diff ** 2)
                s_y += torch.sqrt(y_diff ** 2)

                ed_loss_pixel = torch.sqrt(x_diff ** 2 + y_diff ** 2)

                ''' option to use normalized values '''
                # ed_loss = torch.sqrt((true_x_coord - pred_x_coord) ** 2 + (true_y_coord - pred_y_coord) ** 2)

                ''' add pixel points to distance calculation tensor '''
                pred_coords_stack_pixel = torch.stack((pred_x_coord_pixel, pred_y_coord_pixel))
                true_coords_stack_pixel = torch.stack((true_x_coord_pixel, true_y_coord_pixel))

                pred_dist_list_array.append(pred_coords_stack_pixel)
                true_dist_list_array.append(true_coords_stack_pixel)

                tot_ed_for_median += ed_loss_pixel.item()
                if o == 0:
                    s_i += ed_loss_pixel
                else:
                    s_s += ed_loss_pixel

            tot_list_np = np.append(tot_list_np, tot_ed_for_median)

            ''' outputs absolute pixelwise distance '''
            vector_pred = pred_dist_list_array[0] - pred_dist_list_array[1]
            vector_true = true_dist_list_array[0] - true_dist_list_array[1]

            pred_distance = torch.sqrt(torch.sum(vector_pred ** 2))
            true_distance = torch.sqrt(torch.sum(vector_true ** 2))
            diff_distance_abs = torch.sqrt((pred_distance - true_distance) ** 2)
            diam_list_np = np.append(diam_list_np, diff_distance_abs.item())

            #print('diff distance:', diff_distance)
            s_diam_diff_abs += diff_distance_abs

        ''' outputs absolute inferior loss, superior loss, total loss and diameter difference and median lists '''
        return s_i, s_s, (s_i + s_s), s_diam_diff_abs, tot_list_np, diam_list_np, s_x , s_y, (i + 1)


''' note that this is only intended to work with BS 1 '''
class PixelDSNTDistanceDoublePredict(nn.Module):
    def __init__(self):
        super(PixelDSNTDistanceDoublePredict, self).__init__()

    def forward(self, input, target):

        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
            s_diam_diff_abs = torch.FloatTensor(1).cuda().zero_()

        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()
            s_diam_diff_abs = torch.FloatTensor(1).zero_()

        ''' make the evenly spaced x and y coordinate masks '''
        x_size = input.shape[-1]
        y_size = input.shape[-2]

        y_soft_argmax, x_soft_argmax = coordinate_map_sigmoid_range(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_half(y_size, x_size)
        #y_soft_argmax, x_soft_argmax = coordinate_map_tanh_range_full(y_size, x_size)
        y_soft_argmax = y_soft_argmax.cuda()
        x_soft_argmax = x_soft_argmax.cuda()

        pred_coordinate_list = []
        true_coordinate_list = []

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):
                ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
                softmax = nn.Softmax(0)
                pred_mask_softmax = softmax(points[0].view(-1)).view(points[0].shape)

                pred_x_coord = torch.sum(pred_mask_softmax * x_soft_argmax).cuda()
                pred_y_coord = torch.sum(pred_mask_softmax * y_soft_argmax).cuda()

                ''' argmax for ground truth, remember this must match the range of the CC '''
                coord_argmax = torch.argmax(points[1]).detach()
                true_x_coord = ((coord_argmax % x_size + 1).float() / x_size).cuda()
                true_y_coord = ((coord_argmax // x_size + 1).float() / y_size).cuda()
                #true_x_coord = ((coord_argmax % x_size + 1 - x_size / 2).float() / x_size).cuda()
                #true_y_coord = ((coord_argmax // x_size + 1 - y_size / 2).float() / y_size).cuda()
                #true_x_coord = (((coord_argmax % x_size + 1) * 2 - (x_size + 1)).float() / x_size).cuda()
                #true_y_coord = (((coord_argmax // x_size + 1) * 2 - (y_size + 1)).float() / y_size).cuda()

                ''' converts normalized values to pixel '''
                pred_coords_stack = torch.stack((pred_x_coord, pred_y_coord))
                true_coords_stack = torch.stack((true_x_coord, true_y_coord))

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_sigmoid(pred_coords_stack, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_sigmoid(true_coords_stack, x_size, y_size)
                #pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_tanh_half(pred_coords_stack, x_size, y_size)
                #true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_tanh_half(true_coords_stack, x_size, y_size)
                #pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_tanh_full(pred_coords_stack, x_size, y_size)
                #true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_tanh_full(true_coords_stack, x_size, y_size)

                ''' returns the actual coordinate which requires a -1 since it is an index '''
                pred_coordinate_list.append([pred_x_coord_pixel.item() - 1, pred_y_coord_pixel.item() - 1])
                true_coordinate_list.append([true_x_coord_pixel.item() - 1, true_y_coord_pixel.item() - 1])

                x_diff = true_x_coord_pixel - pred_x_coord_pixel
                y_diff = true_y_coord_pixel - pred_y_coord_pixel
                ed_loss_pixel = torch.sqrt(x_diff ** 2 + y_diff ** 2)

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

        ''' outputs absolute inferior loss, superior loss, total loss and diameter difference, median lists and coords '''
        return s_i / (i + 1), s_s / (i + 1), (s_i + s_s) / (i + 1), s_diam_diff_abs / (i + 1), pred_coordinate_list, true_coordinate_list









''' below is mostly deprecated '''

class DSNTDistanceAngleDoubleLoss_old(nn.Module):
    def __init__(self):
        super(DSNTDistanceAngleDoubleLoss_old, self).__init__()

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


''' note that this only works when images are predicted with batch size of 1 '''
class PixelDSNTDistanceDoublePredict_old(nn.Module):
    def __init__(self):
        super(PixelDSNTDistanceDoublePredict_old, self).__init__()

    def forward(self, input, target):
        if input.is_cuda:
            s_i = torch.FloatTensor(1).cuda().zero_()
            s_s = torch.FloatTensor(1).cuda().zero_()
            s_diam_diff = torch.FloatTensor(1).cuda().zero_()
        else:
            s_i = torch.FloatTensor(1).zero_()
            s_s = torch.FloatTensor(1).zero_()
            s_diam_diff = torch.FloatTensor(1).zero_()

        ''' calculates center of mass of the heatmap with softmax, in other words DSNT '''
        x_size = input[0].shape[-1]
        y_size = input[0].shape[-2]

        x_soft_argmax = torch.zeros((y_size, x_size)).cuda()
        y_soft_argmax = torch.zeros((y_size, x_size)).cuda()

        for p in range(y_size):
            y_soft_argmax[p, :] = (p + 1) / y_size

        for j in range(x_size):
            x_soft_argmax[:, j] = (j + 1) / x_size

        pred_coordinate_list = []
        true_coordinate_list = []

        for i, c in enumerate(zip(input, target)):

            pred_dist_list_array = []
            true_dist_list_array = []

            for o, points in enumerate(zip(c[0], c[1])):

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

                pred_x_coord_pixel, pred_y_coord_pixel = coords_norm_to_pixel_sigmoid(pred_coords_stack, x_size, y_size)
                true_x_coord_pixel, true_y_coord_pixel = coords_norm_to_pixel_sigmoid(true_coords_stack, x_size, y_size)

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
