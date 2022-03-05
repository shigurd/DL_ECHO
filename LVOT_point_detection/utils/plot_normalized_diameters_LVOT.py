import numpy as np
import math
from PIL import Image

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

    true_s_to_pred_i_vector_unit_rotated = rotate_unit_vector(true_s_to_pred_i_vector_unit, scale_angle_rad)
    true_s_to_pred_s_vector_unit_rotated = rotate_unit_vector(true_s_to_pred_s_vector_unit, scale_angle_rad)

    scaled_pred_i = canvas_center + true_s_to_pred_i_vector_unit_rotated * canvas_true_dist * true_s_to_pred_i_vector_relative_dist
    scaled_pred_s = canvas_center + true_s_to_pred_s_vector_unit_rotated * canvas_true_dist * true_s_to_pred_s_vector_relative_dist

    return [int(scaled_pred_i[0]), int(scaled_pred_i[1])], [int(scaled_pred_s[0]), int(scaled_pred_s[1])]


if __name__ == '__main__':
    pred = [[220, 220], [230, 100]]
    pred_x_i, pred_y_i = pred[0]
    pred_x_s, pred_y_s = pred[1]

    true = [[180, 180], [100, 100]]
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