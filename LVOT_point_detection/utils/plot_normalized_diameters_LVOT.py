import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

def get_mag(vector_np):
    dist = np.sqrt(vector_np.dot(vector_np))

    return dist


def get_normalized_vectors(vector_np):
    unit_vector_np = vector_np / np.linalg.norm(vector_np)

    return unit_vector_np


def get_vector_angles(vector1, vector2):
    unit_vector1 = get_normalized_vectors(vector1)
    unit_vector2 = get_normalized_vectors(vector2)

    dot = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot) #radians

    return angle


def rotate_unit_vector(vector_np, angle_rad):
    rot = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]])
    rot_vector_np = np.dot(rot, vector_np)

    return rot_vector_np

def calculate_lvot_angle_deviation(true_coordinate_list, pred_coordinate_list):
    pred_i_x, pred_i_y = pred_coordinate_list[0]
    pred_s_x, pred_s_y = pred_coordinate_list[1]

    true_i_x, true_i_y = true_coordinate_list[0]
    true_s_x, true_s_y = true_coordinate_list[1]

    angle_rad_lvotd = get_vector_angles((pred_i_x - pred_s_x, pred_i_y - pred_s_y), (true_i_x - true_s_x, true_i_y - true_s_y))
    angle_deg_lvotd = angle_rad_lvotd * 180 / math.pi

    return angle_deg_lvotd


def calculate_scaled_points(true_coordinate_list, pred_coordinate_list, norm_lvot_pix_length=50, canvas_center_x=128, canvas_center_y=102):
    ''' plots ground truth and prediction og ground truth masks '''
    pred_i_x, pred_i_y = pred_coordinate_list[0]
    pred_s_x, pred_s_y = pred_coordinate_list[1]

    true_i_x, true_i_y = true_coordinate_list[0]
    true_s_x, true_s_y = true_coordinate_list[1]

    ''' define center '''
    axis_center_x, axis_center_y = true_s_x, true_s_y

    ''' these are used for scaling '''
    true_vector = np.array([true_i_x - axis_center_x, true_i_y - axis_center_y])
    true_vector_unit = get_normalized_vectors(true_vector)
    true_dist = get_mag(true_vector)

    true_s_to_pred_i_vector = np.array([pred_i_x - axis_center_x, pred_i_y - axis_center_y])
    true_s_to_pred_i_vector_unit = get_normalized_vectors(true_s_to_pred_i_vector)
    true_s_to_pred_i_vector_relative_dist = get_mag(true_s_to_pred_i_vector) / true_dist

    true_s_to_pred_s_vector = np.array([pred_s_x - axis_center_x, pred_s_y - axis_center_y])
    true_s_to_pred_s_vector_unit = get_normalized_vectors(true_s_to_pred_s_vector)
    true_s_to_pred_s_vector_relative_dist = get_mag(true_s_to_pred_s_vector) / true_dist

    canvas_center = np.array([canvas_center_x, canvas_center_y])

    canvas_vector_unit = np.array([0, 1])
    canvas_true_dist = norm_lvot_pix_length

    scale_angle_rad = get_vector_angles(canvas_vector_unit, true_vector_unit)

    if true_i_x < true_s_x:
        scale_angle_rad = -scale_angle_rad
    elif true_i_x == true_s_x:
        scale_angle_rad = 0

    true_s_to_pred_i_vector_unit_rotated = rotate_unit_vector(true_s_to_pred_i_vector_unit, scale_angle_rad)
    true_s_to_pred_s_vector_unit_rotated = rotate_unit_vector(true_s_to_pred_s_vector_unit, scale_angle_rad)

    scaled_pred_i = canvas_center + true_s_to_pred_i_vector_unit_rotated * canvas_true_dist * true_s_to_pred_i_vector_relative_dist
    scaled_pred_s = canvas_center + true_s_to_pred_s_vector_unit_rotated * canvas_true_dist * true_s_to_pred_s_vector_relative_dist

    return [int(scaled_pred_i[0]), int(scaled_pred_i[1])], [int(scaled_pred_s[0]), int(scaled_pred_s[1])]


def plot_point_and_lines_to_canvas(coordinate_list, point_cfg, line_cfg):
    x_i, y_i = coordinate_list[0]
    x_s, y_s = coordinate_list[1]

    ''' convert from np idx to cartesian '''
    y_i = 255 - y_i
    y_s = 255 - y_s

    ''' plot points and lines '''
    plt.plot(x_i, y_i, point_cfg)
    plt.plot(x_s, y_s, point_cfg)
    plt.plot([x_i, x_s], [y_i, y_s], line_cfg)


def plot_lines_lvot_gt_and_pred(true_coordinate_list, pred_coordinate_list):
    h, w = 256, 256
    dpi = 96
    plt.figure(figsize=(h / dpi, w / dpi), dpi=dpi)
    plt.ylim([0, 255])
    plt.xlim([0, 255])
    plt.gca().set_aspect('equal')

    true_x_i, true_y_i = true_coordinate_list[0]
    true_x_s, true_y_s = true_coordinate_list[1]

    pred_x_i, pred_y_i = pred_coordinate_list[0]
    pred_x_s, pred_y_s = pred_coordinate_list[1]

    ''' convert from np idx to cartesian '''
    true_y_i = 255 - true_y_i
    true_y_s = 255 - true_y_s
    pred_y_i = 255 - pred_y_i
    pred_y_s = 255 - pred_y_s

    ''' plot gt '''
    plt.plot(true_x_i, true_y_i, 'g+')
    plt.plot(true_x_s, true_y_s, 'g+')
    plt.plot([true_x_i, true_x_s], [true_y_i, true_y_s], 'g--')

    ''' plot pred '''
    plt.plot(pred_x_i, pred_y_i, 'r,')
    plt.plot(pred_x_s, pred_y_s, 'r,')
    plt.plot([pred_x_i, pred_x_s], [pred_y_i, pred_y_s], 'r-')

    plt.show()


if __name__ == '__main__':
    pred = [[180, 200], [200, 180]]
    pred_x_i, pred_y_i = pred[0]
    pred_x_s, pred_y_s = pred[1]

    true = [[200, 150], [200, 100]]
    true_x_i, true_y_i = true[0]
    true_x_s, true_y_s = true[1]

    canvas = np.zeros((256, 256, 3))
    canvas[true_y_i, true_x_i, 1] = 255
    canvas[true_y_s, true_x_s, 1] = 255
    canvas[pred_y_i, pred_x_i, 0] = 255
    canvas[pred_y_s, pred_x_s, 0] = 255

    canvas_pil = Image.fromarray(canvas.astype(np.uint8))
    canvas_pil.show()

    canvas_norm_lvot = 50
    canvas_s_point = [128, 102]
    canvas_x_s, canvas_y_s = canvas_s_point

    i_pred_scale, s_pred_scale = calculate_scaled_points(true, pred, canvas_norm_lvot, canvas_s_point[0], canvas_s_point[1])

    canvas_scaled = np.zeros((256, 256, 3))
    canvas_scaled[canvas_y_s, canvas_x_s, 1] = 255
    canvas_scaled[canvas_y_s + canvas_norm_lvot, canvas_x_s, 1] = 255
    canvas_scaled[i_pred_scale[1], i_pred_scale[0], 0] = 255
    canvas_scaled[s_pred_scale[1], s_pred_scale[0], 0] = 255

    canvas_scaled_pil = Image.fromarray(canvas_scaled.astype(np.uint8))
    canvas_scaled_pil.show()

    ''' angle deviation '''
    angle_deviation = calculate_lvot_angle_deviation(true, pred)
    print('LVOTd angle deviation', angle_deviation)

    ''' connected plot '''
    plot_lines_lvot_gt_and_pred(true, pred)
