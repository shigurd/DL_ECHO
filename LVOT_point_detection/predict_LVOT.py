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
import math
import csv
import ast

from utils.dataloader_LVOT import BasicDataset
from utils.point_losses_LVOT import PixelDSNTDistanceDoublePredict

sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from networks.unet import UNet
from dicom_extraction_utils_GE.LVOT_coords import get_cm_coordinates


def predict_tensor(net,
                   img_pil,
                   device,
                   scale_factor=1):
    net.eval()

    img_np = BasicDataset.preprocess(img_pil, scale_factor)
    img_tensor = torch.from_numpy(img_np)

    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img_tensor)
        output = output['out']

    return output


def mask_to_image(mask_tensor):
    mask_np = mask_tensor.squeeze().numpy()
    return Image.fromarray((mask_np * 255).astype(np.uint8))


def concat_img(img1_pil, img2_pil):
    new_img = Image.new('RGB', (img1_pil.width, img1_pil.height + img2_pil.height))
    new_img.paste(img1_pil, (0, 0))
    new_img.paste(img2_pil, (0, img1_pil.height))
    return new_img


def draw_cross(img_np, x_center, y_center, radius, color=(255, 255, 255), rgb=True):
    ''' draws cross on given center coordinates with radius as length of appendages '''
    y_size, x_size = img_np.shape[:2]
    if rgb == True:
        for y in range(y_size):
            for x in range(x_size):
                try:
                    if x == x_center and (y_center - radius <= y <= y_center + radius):
                        img_np[y, x] = color
                    elif y == y_center and (x_center - radius <= x <= x_center + radius):
                        img_np[y, x] = color
                except:
                    print('skipped pixel because coordinate out of bounds')

        return img_np
    else:
        print('np_img is not rgb, change color format')


def predict_plot_on_truth_mask(true_mask_i_tensor, true_mask_s_tensor, true_coordinate_list, pred_coordinate_list):
    ''' plots ground truth and prediction og ground truth masks '''
    pred_i_x, pred_i_y = pred_coordinate_list[0]
    pred_s_x, pred_s_y = pred_coordinate_list[1]

    true_i_x, true_i_y = true_coordinate_list[0]
    true_s_x, true_s_y = true_coordinate_list[1]

    ''' converts normalized values back into 0-255 color range and makes it rgb for 3 channels '''
    mask_sum_tensor = true_mask_i_tensor + true_mask_s_tensor
    mask_pil = mask_to_image(mask_sum_tensor)
    mask_pil = mask_pil.convert('RGB')
    mask_np = np.array(mask_pil)

    zeros_true = np.zeros(mask_np.shape)
    zeros_pred = np.zeros(mask_np.shape)

    ''' draws a + on each true coordinate '''
    true_coords_np = zeros_true
    true_coords_np = draw_cross(true_coords_np, true_i_x, true_i_y, 4, color=[0, 255, 0])
    true_coords_np = draw_cross(true_coords_np, true_s_x, true_s_y, 4, color=[0, 255, 0])
    ''' only draws 1 pixel for true '''
    #true_coords_np[true_i_y, true_i_x] = [0, 255, 0]
    #true_coords_np[true_s_y, true_s_x] = [0, 255, 0]

    ''' draws a + on each predicted coordinate '''
    pred_coords_np = zeros_pred
    pred_coords_np = draw_cross(pred_coords_np, pred_i_x, pred_i_y, 4, color=[255, 0, 0])
    pred_coords_np = draw_cross(pred_coords_np, pred_s_x, pred_s_y, 4, color=[255, 0, 0])
    ''' only draws 1 pixel for prediction '''
    #pred_coords_np[pred_i_y, pred_i_x] = [255, 0, 0]
    #pred_coords_np[pred_s_y, pred_s_x] = [255, 0, 0]

    ''' removes negative colors '''
    absolutt_diff_coords = np.absolute(pred_coords_np - true_coords_np).astype(np.uint8)

    ''' draws colored pixels from abs diff on true mask  '''
    for y in range(absolutt_diff_coords.shape[0]):
        for x in range(absolutt_diff_coords.shape[1]):
            if np.sum(absolutt_diff_coords[y, x]) != 0:
                mask_np[y, x] = absolutt_diff_coords[y, x]
            else:
                pass

    plot_pil = Image.fromarray(mask_np.astype(np.uint8), 'RGB')
    #plot_pil.show()

    return plot_pil


def predict_plot_on_image(img_pil, pred_coordinate_list, true_coordinate_list, plot_gt=False):
    ''' pred and true  [[x_i, y_i], [x_s, y_s]] '''
    pred_coordinate_list = [[round(pred_coordinate_list[0][0]), round(pred_coordinate_list[0][1])],
                            [round(pred_coordinate_list[1][0]), round(pred_coordinate_list[1][1])]]
    true_coordinate_list = [[round(true_coordinate_list[0][0]), round(true_coordinate_list[0][1])],
                            [round(true_coordinate_list[1][0]), round(true_coordinate_list[1][1])]]

    pred_i_x, pred_i_y = pred_coordinate_list[0]
    pred_s_x, pred_s_y = pred_coordinate_list[1]

    true_i_x, true_i_y = true_coordinate_list[0]
    true_s_x, true_s_y = true_coordinate_list[1]

    img_pil = img_pil.convert('RGB')
    img_np = np.array(img_pil)

    zeros_true = np.zeros(img_np.shape)
    zeros_pred = np.zeros(img_np.shape)

    ''' draws a + on each predicted coordinate '''
    pred_coords_np = zeros_pred
    pred_coords_np = draw_cross(pred_coords_np, pred_i_x, pred_i_y, 4, color=[255, 0, 0])
    pred_coords_np = draw_cross(pred_coords_np, pred_s_x, pred_s_y, 4, color=[255, 0, 0])
    ''' only draws 1 pixel for prediction '''
    #pred_coords_np[pred_i_y, pred_i_x] = [255, 0, 0]
    #pred_coords_np[pred_s_y, pred_s_x] = [255, 0, 0]

    if plot_gt == True:
        ''' draws a + on each true coordinate '''
        true_coords_np = zeros_true
        true_coords_np = draw_cross(true_coords_np, true_i_x, true_i_y, 4, color=[0, 255, 0])
        true_coords_np = draw_cross(true_coords_np, true_s_x, true_s_y, 4, color=[0, 255, 0])
        ''' only draws 1 pixel for true '''
        # true_coords_np[true_i_y, true_i_x] = [0, 255, 0]
        # true_coords_np[true_s_y, true_s_x] = [0, 255, 0]

        ''' removes negative colors '''
        crosses_to_plot = np.absolute(pred_coords_np - true_coords_np).astype(np.uint8)
    else:
        crosses_to_plot = pred_coords_np

    ''' draws colored pixels from abs diff on true mask  '''
    for y in range(crosses_to_plot.shape[0]):
        for x in range(crosses_to_plot.shape[1]):
            if np.sum(crosses_to_plot[y, x]) != 0:
                img_np[y, x] = crosses_to_plot[y, x]
            else:
                pass

    plot_pil = Image.fromarray(img_np.astype(np.uint8), 'RGB')
    #plot_pil.show()

    return plot_pil


def pil_overlay(foreground_pil, background_pil):  # pil img input
    img1 = foreground_pil.convert("RGBA")
    img2 = background_pil.convert("RGBA")

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


def calculate_lvot_diameter(x1, y1, x2, y2):
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    diam = float('{:.4f}'.format(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)))

    return diam

def predict_cm_coords_and_diameter(file_id, pred_coord_list_pix, true_coord_list_pix, keyfile_csv):
    ''' input format of pix_coords is [[x1_cm, y1_pix] [x2, y2_pix]] '''

    ''' file format is patient_number, img_number, img_type, img_zoom, i_quality, m_quality '''
    file_id = file_id.rsplit('_', 4)[0]

    with open(keyfile_csv, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        ''' skips header '''
        next(csv_reader, None)
        found = False

        for row in csv_reader:
            dcm_id, exam_id, patient_id, measure_type, img_quality, gt_quality, img_view, x1_pix, y1_pix, x2_pix, y2_pix, x1_cm, y1_cm, x2_cm, y2_cm, true_diam_cm_csv, sc_param = row

            sc_param = ast.literal_eval(sc_param)

            if file_id == patient_id:
                found = True

                ''' caluculate pixel diameter, negative number indicate too short diameter, positive number indicate too long diameter '''
                pred_diam_pix = calculate_lvot_diameter(pred_coord_list_pix[0][0], pred_coord_list_pix[0][1], pred_coord_list_pix[1][0], pred_coord_list_pix[1][1])
                true_diam_pix = calculate_lvot_diameter(true_coord_list_pix[0][0], true_coord_list_pix[0][1], true_coord_list_pix[1][0], true_coord_list_pix[1][1])
                #diff_diam_pix = pred_diam_pix - true_diam_pix

                ''' convert pixel to cm with scanconverted parameters '''
                pred_coord_list_pix = np.array(pred_coord_list_pix)
                pred_coords_cm = get_cm_coordinates(pred_coord_list_pix, sc_param)

                true_coord_list_pix = np.array(true_coord_list_pix)
                true_coords_cm = get_cm_coordinates(true_coord_list_pix, sc_param)

                ''' caluculate cm diameter, negative number indicate too short diameter, positive number indicate too long diameter '''
                pred_diam_cm = calculate_lvot_diameter(pred_coords_cm[0][0], pred_coords_cm[0][1], pred_coords_cm[1][0], pred_coords_cm[1][1])
                true_diam_cm = calculate_lvot_diameter(true_coords_cm[0][0], true_coords_cm[0][1], true_coords_cm[1][0], true_coords_cm[1][1])
                diff_diam_cm = pred_diam_cm - true_diam_cm

                #print('\nfrom csv true coords pix: ', x2_pix, y2_pix, x1_pix, y1_pix)
                #print('from calculated true coords pix', true_coord_list_pix[0][0], true_coord_list_pix[0][1], true_coord_list_pix[1][0], true_coord_list_pix[1][1])
                #print('\nfrom csv true coords cm: ', x2_cm, y2_cm, x1_cm, y1_cm)
                #print('from calculated true coords cm', true_coords_cm[0][0], true_coords_cm[0][1], true_coords_cm[1][0], true_coords_cm[1][1])

                '''
                #OBS dette er fordi enkelte koordinater var reversert lagret i echopac der x2_pix, y2_pix kommer først. har korrigert for dette men det blir et nytt problem når man bruker scanconvert
                if diff_diam_cm * diff_diam_pix < 0:
                    diff_diam_cm = diff_diam_cm * -1
                elif diff_diam_pix == 0:
                    diff_diam_cm = 0
                '''
                #print(f'calculated lvot diam pix: {pred_diam_pix}')
                #print(f'calculated lvot diam diff pix: {diff_diam_pix}')

                return pred_diam_cm, diff_diam_cm

        if found == False:
            print(file_id, 'NOT FOUND')
            return 'nan', 'nan'

def get_output_filenames(in_file):
    in_files = in_file
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], '.png'))

    return out_files

if __name__ == "__main__":
    
    ''' define model name, prediction dataset and model parameters '''
    keyfile_csv = r'H:/ML_LVOT/backup_keyfile_and_duplicate/keyfile_GE1424_QC.csv'
    model_file = 'Aug25_01-17-15_RES50_DSNT_ADAM_T-AVA1314X5_HMHM_MA4_V-_EP30_LR0.001_BS20_SCL1.pth'
    data_name = 'AVA1314X5_HMLHML'
    n_channels = 1
    n_classes = 2
    scaling = 1
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
        file = open(path.join(predictions_output, f'COORD_DATA_{model_name}.txt'), 'w+')
        file.write('file_name,measure_type,view_type,img_quality,gt_quality,diff_diam_pix,diff_diam_cm,diam_cm\n')
        file1 = open(path.join(predictions_output, 'temp.txt'), 'w+')
        file1.close()
        file2 = open(path.join(predictions_output, 'temp1.txt'), 'w+')
        file2.close()
        file3 = open(path.join(predictions_output, 'temp2.txt'), 'w+')
        file3.close()
        file4 = open(path.join(predictions_output, 'temp3.txt'), 'w+')
        file4.close()
        file5 = open(path.join(predictions_output, 'temp4.txt'), 'w+')
        file5.close()

        ''' all values here are in absolute values '''
        median_lvot_diam_absdiff_pix = np.array([])
        median_lvot_diam_absdiff_cm = np.array([])
        total_i_ed_pix = 0
        total_s_ed_pix = 0
        total_sum_ed_pix = 0
        total_lvot_diam_absdiff_pix = 0
        total_lvot_diam_absdiff_cm = 0

    with tqdm(total=len(input_files), desc='Predictions', unit='imgs', leave=False) as pbar:

        for i, fn in enumerate(input_files):
            out_fn = out_files[i]
            logging.info("\nPredicting image {} ...".format(fn))
            img_pil = Image.open(path.join(dir_img, fn))
            img_pil = img_pil.convert('L')

            ''' predict_tensor returns logits '''
            masks_tensors_predicted = predict_tensor(net=net,
                                                     img_pil=img_pil,
                                                     scale_factor=scaling,
                                                     device=device)

            ''' if ground truth is available, make overlays and calculate mean and median dice '''
            if compare_with_ground_truth == True:

                ''' load masks as tensor and concatenate for loss '''
                masks_paths_true = glob(path.join(dir_mask, fn.rsplit(".", 1)[0]) + '*')
                masks_tensors_true = []
                for mask in masks_paths_true:
                    mask_pil_true = Image.open(mask)
                    mask_pil_true = mask_pil_true.convert('L')
                    mask_np_true = BasicDataset.preprocess(mask_pil_true, scaling)
                    mask_np_true = np.expand_dims(mask_np_true, axis=0) #adds the batch size
                    mask_tensor_true = torch.from_numpy(mask_np_true)
                    masks_tensors_true.append(mask_tensor_true)
                masks_tensors_true_cat = torch.cat((masks_tensors_true[0], masks_tensors_true[1]), 1).to(device=device, dtype=torch.float32)

                criterion = PixelDSNTDistanceDoublePredict()
                loss_list_tensor = criterion(masks_tensors_predicted, masks_tensors_true_cat)
                ed_i_pix = loss_list_tensor[0].item()
                ed_s_pix = loss_list_tensor[1].item()
                ed_tot_pix = loss_list_tensor[2].item()
                diff_diam_pix = loss_list_tensor[3].item()
                absdiff_diam_pix = abs(diff_diam_pix)

                pred_coordinate_list = loss_list_tensor[4]
                true_coordinate_list = loss_list_tensor[5]

                ''' calculate inferior ED, superior ED, summed ED, summed absolute lvot diameter and median absolute lvot diameter in pixels '''
                total_i_ed_pix += ed_i_pix
                total_s_ed_pix += ed_s_pix
                total_sum_ed_pix += ed_tot_pix
                total_lvot_diam_absdiff_pix += absdiff_diam_pix

                median_lvot_diam_absdiff_pix = np.append(median_lvot_diam_absdiff_pix, absdiff_diam_pix)

                ''' converting pixel lvot predicitons to cm '''
                pred_diam_cm, diff_diam_cm = predict_cm_coords_and_diameter(fn, pred_coordinate_list, true_coordinate_list, keyfile_csv)

                ''' calculate total and median for lvot diameter cm '''
                absdiff_diam_cm = abs(diff_diam_cm)
                total_lvot_diam_absdiff_cm += absdiff_diam_cm
                median_lvot_diam_absdiff_cm = np.append(median_lvot_diam_absdiff_cm, absdiff_diam_cm)

                ''' log the data '''
                diff_diam_pix = '{:.4f}'.format(diff_diam_pix)
                absdiff_diam_pix = '{:.4f}'.format(absdiff_diam_pix)
                diff_diam_cm = '{:.4f}'.format(diff_diam_cm)
                absdiff_diam_cm = '{:.4f}'.format(absdiff_diam_cm)
                pred_diam_cm = '{:.4f}'.format(pred_diam_cm)
                patient_id, measure_type, view_type, img_quality, gt_quality = fn.rsplit('.', 1)[0].rsplit('_', 4)
                file.write(f'{fn},{measure_type},{view_type},{img_quality},{gt_quality},{diff_diam_pix},{diff_diam_cm},{pred_diam_cm}\n')

                ''' plotting and saving coordinate overlay on original image with gt '''
                pred_plot = predict_plot_on_image(img_pil, pred_coordinate_list, true_coordinate_list, plot_gt=compare_with_ground_truth)
                pred_plot.save(path.join(predictions_output, f'{str(absdiff_diam_pix)}_{out_fn}'))

            else:
                ''' just save coordinate overlay on original image '''
                pred_plot = predict_plot_on_image(img_pil, pred_coordinate_list, true_coordinate_list, plot_gt=compare_with_ground_truth)
                pred_plot.save(path.join(predictions_output, out_fn))

            logging.info("Mask saved to {}".format(out_files[i]))
            pbar.update()

        if compare_with_ground_truth == True:
            file.close()
            avg_sum_ed_pix = total_sum_ed_pix / (i + 1)
            avg_lvot_diam_absdiff_pix = total_lvot_diam_absdiff_pix / (i + 1)
            avg_lvot_diam_absdiff_cm = total_lvot_diam_absdiff_cm / (i + 1)
            avg_sum_ed_pix = '{:.4f}'.format(avg_sum_ed_pix)
            avg_lvot_diam_absdiff_pix = '{:.4f}'.format(avg_lvot_diam_absdiff_pix)
            avg_lvot_diam_absdiff_cm = '{:.4f}'.format(avg_lvot_diam_absdiff_cm)
            os.rename(path.join(predictions_output, 'temp.txt'), path.join(predictions_output, f'AVG_SUM_ED_PIX_{avg_sum_ed_pix}_{model_name}.txt'))
            os.rename(path.join(predictions_output, 'temp1.txt'), path.join(predictions_output, f'AVD_LVOTD_PIX_{avg_lvot_diam_absdiff_pix}_{model_name}.txt'))
            os.rename(path.join(predictions_output, 'temp2.txt'), path.join(predictions_output, f'AVG_LVOTD_CM_{avg_lvot_diam_absdiff_cm}_{model_name}.txt'))
            os.rename(path.join(predictions_output, 'temp3.txt'), path.join(predictions_output, f'MEDIAN_LVOTD_PIX_{np.median(median_lvot_diam_absdiff_pix)}_{model_name}.txt'))
            os.rename(path.join(predictions_output, 'temp4.txt'), path.join(predictions_output, f'MEDIAN_LVOTD_CM_{np.median(median_lvot_diam_absdiff_cm)}_{model_name}.txt'))




''' deprecated as all is done in loss function now '''
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

            ''' alternative argmax method of getting max value, but this is not the one used in training '''
            # coorda = torch.argmax(point)
            # true_x_coord = ((coord_argmax % x_size + 1).float() / x_size)
            # true_y_coord = ((coord_argmax // x_size + 1).float() / y_size)

            ''' converting each coordinate value back into index which requires - 1 '''
            x_index = float(pred_x_coord.item() * x_size - 1)
            y_index = float(pred_y_coord.item() * y_size - 1)
            coordinate_list.append([x_index, y_index])

    ''' output is in the format [[x1, y1], [x2, y2]] '''
    return coordinate_list


''' deprecated as all is done in loss function now '''
def coords_from_true_mask(mask_tensor):
    ''' converts true tensors into coordinates, more efficient as it only uses argmax compared to pred '''
    x_size = mask_tensor.shape[-1]
    y_size = mask_tensor.shape[-2]

    true_coord = torch.argmax(mask_tensor)
    true_x_tensor = ((true_coord % x_size + 1).float() / x_size)
    true_y_tensor = ((true_coord // x_size + 1).float() / y_size)

    ''' adds -1 as the smallest index is 0 and not 1 '''
    true_x_int = round(true_x_tensor.item() * x_size - 1)
    true_y_int = round(true_y_tensor.item() * y_size - 1)

    return true_x_int, true_y_int