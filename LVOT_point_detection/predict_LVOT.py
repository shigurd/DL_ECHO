import logging
import os
import os.path as path
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils.dataloader_LVOT import BasicDataset
from utils.point_losses_LVOT import DSNTDoubleLoss

import math
import csv
import ast

sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50


def mask_to_image(mask_tensor):
    mask_np = mask_tensor.squeeze().numpy()
    return Image.fromarray((mask_np * 255).astype(np.uint8))


def concat_img(img1, img2):  # pil img input
    new_img = Image.new('RGB', (img1.width, img1.height + img2.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    return new_img


def coords_from_true_mask(mask_tensor):
    x_size = mask_tensor.shape[-1]
    y_size = mask_tensor.shape[-2]

    true_coord = torch.argmax(mask_tensor)
    true_x_tensor = ((true_coord % x_size + 1).float() / x_size)
    true_y_tensor = ((true_coord // x_size + 1).float() / y_size)

    ''' -1 as the smallest index is 0 and not 1 '''
    true_x_int = round(true_x_tensor.item() * x_size - 1)
    true_y_int = round(true_y_tensor.item() * y_size - 1)

    return true_x_int, true_y_int


def coords_from_pred(pred_tensor):
    coordinate_list = []

    for batch in pred_tensor:
        for channel in batch:

            x_size = channel.shape[-1]
            y_size = channel.shape[-2]

            soft_argmax_x = torch.zeros((y_size, x_size)).cuda()
            soft_argmax_y = torch.zeros((y_size, x_size)).cuda()

            for p in range(y_size):
                soft_argmax_y[p, :] = (p + 1) / y_size

            for j in range(x_size):
                soft_argmax_x[:, j] = (j + 1) / x_size

            softmax = nn.Softmax(0)
            pred_softmax = softmax(channel.view(-1)).view(channel.shape)
            pred_x_coord = torch.sum(pred_softmax * soft_argmax_x).cuda()
            pred_y_coord = torch.sum(pred_softmax * soft_argmax_y).cuda()

            '''
            coorda = torch.argmax(point)
            true_x_coord = ((coord_argmax % x_size + 1).float() / x_size) 
            true_y_coord = ((coord_argmax // x_size + 1).float() / y_size)
            '''

            coordinate_list.append([float(pred_x_coord.item() * x_size - 1), float(
                pred_y_coord.item() * y_size - 1)])  # -1 for å gjøre dem til index der minste val er 0

    return coordinate_list


def draw_cross(np_img, x_center, y_center, radius, color=(255, 255, 255), rgb=True):
    y_size, x_size = np_img.shape[:2]

    if rgb == True:
        for y in range(y_size):
            for x in range(x_size):
                try:
                    if x == x_center and (y_center - radius <= y <= y_center + radius):
                        np_img[y, x] = color
                    elif y == y_center and (x_center - radius <= x <= x_center + radius):
                        np_img[y, x] = color
                except:
                    print('skipped pixel because coordinate out of bounds')

        return np_img
    else:
        print('np_img is not rgb, change color format')


def predict_plot(mask_l_tensor, mask_r_tensor, coordinate_list):
    img_pil = img_pil.convert('RGB')

    pred_i_x, pred_i_y = coordinate_list[0]
    pred_s_x, pred_s_y = coordinate_list[1]

    true_i_x, true_i_y = coords_from_true_mask(mask_l_tensor)
    true_s_x, true_s_y = coords_from_true_mask(mask_r_tensor)

    mask_sum_tensor = mask_l_tensor + mask_r_tensor
    mask_pil = mask_to_image(mask_sum_tensor)  # for å gjøre ting RGB
    mask_pil = mask_pil.convert('RGB')
    mask_np = np.array(mask_pil)

    zeros_true = np.zeros(mask_np.shape)
    zeros_pred = np.zeros(mask_np.shape)

    true_coords_np = zeros_true
    true_coords_np = draw_cross(true_coords_np, true_i_x, true_i_y, 4, color=[0, 255, 0])
    true_coords_np = draw_cross(true_coords_np, true_s_x, true_s_y, 4, color=[0, 255, 0])
    # true_coords_np[true_i_y, true_i_x] = [0, 255, 0] #tegner en pixel
    # true_coords_np[true_s_y, true_s_x] = [0, 255, 0]

    pred_coords_np = zeros_pred
    pred_coords_np = draw_cross(pred_coords_np, pred_i_x, pred_i_y, 4, color=[255, 0, 0])
    pred_coords_np = draw_cross(pred_coords_np, pred_s_x, pred_s_y, 4, color=[255, 0, 0])
    # pred_coords_np[pred_i_y, pred_i_x] = [255, 0, 0] #tegner en pixel
    # pred_coords_np[pred_s_y, pred_s_x] = [255, 0, 0]

    absolutt_diff_coords = np.absolute(pred_coords_np - true_coords_np).astype(np.uint8)

    for y in range(absolutt_diff_coords.shape[0]):
        for x in range(absolutt_diff_coords.shape[1]):

            if np.sum(absolutt_diff_coords[y, x]) != 0:
                mask_np[y, x] = absolutt_diff_coords[y, x]
            else:
                pass

    plot = Image.fromarray(mask_np.astype(np.uint8), 'RGB')
    # plot.show()

    return plot  # pil img


def predict_plot_img(img_pil, mask_l_tensor, mask_r_tensor, coordinate_list):
    coordinate_list = [[round(coordinate_list[0][0]), round(coordinate_list[0][1])],
                       [round(coordinate_list[1][0]), round(coordinate_list[1][1])]]

    pred_l_x, pred_l_y = coordinate_list[0]
    pred_r_x, pred_r_y = coordinate_list[1]

    true_l_x, true_l_y = coords_from_true_mask(mask_l_tensor)
    true_r_x, true_r_y = coords_from_true_mask(mask_r_tensor)

    img_np = np.array(img_pil)

    zeros_true = np.zeros(img_np.shape)
    zeros_pred = np.zeros(img_np.shape)

    true_coords_np = zeros_true
    true_coords_np = draw_cross(true_coords_np, true_l_x, true_l_y, 4, color=[0, 255, 0])
    true_coords_np = draw_cross(true_coords_np, true_r_x, true_r_y, 4, color=[0, 255, 0])
    # true_coords_np[true_l_y, true_l_x] = [0, 255, 0] #tegner en pixel
    # true_coords_np[true_r_y, true_r_x] = [0, 255, 0]

    pred_coords_np = zeros_pred
    pred_coords_np = draw_cross(pred_coords_np, pred_l_x, pred_l_y, 4, color=[255, 0, 0])
    pred_coords_np = draw_cross(pred_coords_np, pred_r_x, pred_r_y, 4, color=[255, 0, 0])
    # pred_coords_np[pred_l_y, pred_l_x] = [255, 0, 0] #tegner en pixel
    # pred_coords_np[pred_r_y, pred_r_x] = [255, 0, 0]

    absolutt_diff_coords = np.absolute(pred_coords_np - true_coords_np).astype(np.uint8)

    for y in range(absolutt_diff_coords.shape[0]):
        for x in range(absolutt_diff_coords.shape[1]):

            if np.sum(absolutt_diff_coords[y, x]) != 0:
                img_np[y, x] = absolutt_diff_coords[y, x]
            else:
                pass

    plot = Image.fromarray(img_np.astype(np.uint8), 'RGB')
    # plot.show()

    return plot  # pil img


def pil_overlay(foreground, background):  # pil img input
    img1 = foreground.convert("RGBA")
    img2 = background.convert("RGBA")

    overlay = Image.blend(img2, img1, alpha=.1)

    return overlay


def img_as_tensor_pil(img_pth, get_pil=True, rgb=False):
    img = Image.open(img_pth)

    if rgb == True:
        img = img.convert('RGB')
    else:
        img = img.convert('L')

    img_pil = img
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)  # adder channel(unet format)
    else:
        img = img.transpose((2, 0, 1))

    img = np.expand_dims(img, axis=0)  # adder batch(unet format)
    img = img / 255  # normalisering(unet format)
    img_tensor = torch.from_numpy(img).float()
    if get_pil == True:
        return img_tensor, img_pil
    else:
        return img_tensor


def calc_lvot_diam(x1, y1, x2, y2):  # sigurd
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    diam = float('{:.4f}'.format(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)))

    return diam


def pixel_too_scancovert(pixel_coords, sc_params):  # sigurd
    # input er np pix_coords [[x1, y1] [x2, y2]]
    '''
    print("x range")
    print(sc_params["xmin"], sc_params["xmax"])
    print("y range")
    print(sc_params["ymin"], sc_params["ymax"])
    print("siz)")
    print(sc_params["shape"])
    '''
    sub_factor = np.array([sc_params["ymin"], -1 * sc_params["xmax"]])
    norm_factor = np.array([sc_params["shape"][0] / (sc_params["ymax"] - sc_params["ymin"]),
                            sc_params["shape"][1] / (sc_params["xmax"] - sc_params["xmin"])])

    pixel_coords = np.array(pixel_coords)
    pixel_coords = pixel_coords[:, ::-1]

    pixel_coords /= norm_factor
    pixel_coords += sub_factor

    pixel_coords /= np.array([1, -1])
    sc_coords = pixel_coords.copy()[:, ::-1]

    coords_cm = sc_coords * 100  # from meter to cm

    return coords_cm


def predict_cm_coords_and_diameter(file_id, pix_coords, keyfile_csv):
    # pix_coords format [[x1, y1] [x2, y2]]

    file_id = file_id.rsplit('_', 4)[0]  # pga nytt filformat

    with open(keyfile_csv, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        next(csv_reader, None)  # skipper header
        found = False

        for row in csv_reader:
            _, exam_id, patient_id, measure_type, img_quality, gt_quality, img_view, x1, y1, x2, y2, x1c, y1c, x2c, y2c, diam, sc_param = row

            sc_param = ast.literal_eval(sc_param)

            if file_id == patient_id:
                found = True

                pred_pix_diam = calc_lvot_diam(pix_coords[0][0], pix_coords[0][1], pix_coords[1][0], pix_coords[1][1])
                true_pix_diam = calc_lvot_diam(x1, y1, x2, y2)
                diff_pix_diam = pred_pix_diam - true_pix_diam  # negative er for kort, positiv er for lang

                pix_coords = np.array(pix_coords)
                coords_cm = pixel_too_scancovert(pix_coords, sc_param)

                pred_cm_diam = calc_lvot_diam(coords_cm[0][0], coords_cm[0][1], coords_cm[1][0], coords_cm[1][1])
                true_cm_diam = calc_lvot_diam(x1c, y1c, x2c, y2c)
                diff_cm_diam = pred_cm_diam - true_cm_diam  # negative er for kort, positiv er for lang

                '''
                #OBS dette er fordi enkelte koordinater var reversert laget i echopac der x2, y2 kommer først. har korrigert for dette men det blir et nytt problem når man bruker scanconvert
                if diff_cm_diam * diff_pix_diam < 0:
                    diff_cm_diam = diff_cm_diam * -1
                elif diff_pix_diam == 0:
                    diff_cm_diam = 0
                '''
                return diff_pix_diam, diff_cm_diam, pred_pix_diam, pred_cm_diam, diam

        if found == False:
            print(file_id, 'NOT FOUND')
            return 'nan', 'nan', 'nan', 'nan'

if __name__ == "__main__":
    
    ''' define model name, prediction dataset and model parameters '''
    model_file = 'Aug16_00-54-49_RES50_DSNT_ADAM_T-AVA1314Y1_HML_V-_EP30_LR0.001_BS20_SCL1.pth'
    data_name = 'AVA1314Y1_HML'
    scaling = 1
    mask_threshold = 0.5
    compare_with_ground_truth = True

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
    
    if compare_with_ground_truth == True:
        file = open(path.join(predictions_output, f'DICEDATA_{model_name}.txt'), 'w+')
        file.write('file_name,dice_score\n')
        file1 = open(path.join(predictions_output, 'temp.txt'), 'w+')
        file1.close()
        file2 = open(path.join(predictions_output, 'temp1.txt'), 'w+')
        file2.close()

        median_list = np.array([])
        total_dice = 0

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
                               device=device)

            ''' converting predicted tensor to pil mask '''
            mask_pil_predicted = convert_tensor_mask_to_pil(mask_tensor_predicted)

            ''' if ground truth is available, make overlays and calculate mean and median dice '''
            if compare_with_ground_truth == True:
                mask_path_true = path.join(dir_mask, f'{fn.rsplit(".", 1)[0]}_mask.png')
                mask_pil_true = Image.open(mask_path_true)
                mask_pil_true = mask_pil_true.convert('L')
                mask_np_true = BasicDataset.preprocess(mask_pil_true, scaling)
                mask_tensor_true = torch.from_numpy(mask_np_true).cuda() # to cuda as this is loaded with cpu

                criterion = DSNTDoubleLoss()
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
