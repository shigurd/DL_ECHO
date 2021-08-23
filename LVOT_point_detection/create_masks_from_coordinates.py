import os
from shutil import copyfile
import argparse
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from operator import itemgetter

def calculate_angle_loss_function(pred_i, pred_s, true_i, true_s):
    vector_pred = pred_i - pred_s
    vector_true = true_i - true_s
    
    angle_rad = math.acos(np.dot(vector_pred, vector_true) / (math.sqrt(np.dot(vector_pred, vector_pred)) * math.sqrt(np.dot(vector_true, vector_true))))
    angle_deg = angle_rad * (180 / math.pi)
    
    return angle_deg

def calculate_angle(x_s, y_s, x_i, y_i):
    ''' x_s and y_s is superior point and x_i and y_i is inferior point '''
    vector_i = np.array([x_i - x_s, y_i - y_s]) #direction caudal
    vector_s = np.array([x_s - x_i, y_s - y_i]) #direction cranial
    
    rot_clockwise = np.array([[0, 1], [-1, 0]])
    rot_counterclock = np.array([[0, -1], [1, 0]])
    
    tan_superior = np.dot(rot_counterclock, vector_i) #direction arcus aorta
    tan_inferior = np.dot(rot_clockwise, vector_s) #direction arcus arcus
    
    x_axis_vector = np.array([1, 0])
    
    angle_superior = math.acos(np.dot(tan_superior, x_axis_vector) / (math.sqrt(np.dot(tan_superior, tan_superior)) * math.sqrt(np.dot(x_axis_vector, x_axis_vector))))
    angle_inferior = math.acos(np.dot(tan_inferior, x_axis_vector) / (math.sqrt(np.dot(tan_inferior, tan_inferior)) * math.sqrt(np.dot(x_axis_vector, x_axis_vector))))
    
    return angle_superior, angle_inferior


def arrange_coordinates(coordinate_list):
    ''' coordiantes are sorted by lowes y value, superior will be index 0 and inferior will be index 1 '''
    return sorted(coordinate_list, key=itemgetter(0))


def create_gaussian(x_coord_center, y_coord_center, height, width, angle, x_warp, y_warp, sigma=5):
    channel = np.zeros((height, width))
    
    for x in range(width):
        for y in range(height):
            channel[y][x] = math.exp(-(((x - x_coord_center) * math.cos(angle) - (y - y_coord_center) * math.sin(angle)) ** 2 / x_warp + ((x - x_coord_center) * math.sin(angle) + (y - y_coord_center) * math.cos(angle)) ** 2 / y_warp) / (2 * sigma ** 2)) #OBS! husk at np array flipper x og y koordinater ved indexing
            
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(height, width))

    '''
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(combined_channel)
    rgb_img = rgba_img[:, :, :-1] #sletter alphachannel
    plot = Image.fromarray((rgb_img * 255).astype(np.uint8))
    plot.show()
    '''
    
    return channel * 255

def get_coordinates(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img_np = np.asarray(img)
    size = img_np.shape
    height = size[0]
    width = size[1]
    
    coordinate_list = []
    for x in range(width):
        for y in range(height):
            if img_np[y][x] == 255: #note that y is before x when indexing
                coordinate_list.append([x, y])
            else:
                pass
    
    return coordinate_list, height, width

def walk(input_dir, output_folder):
    """Walk through the inputDir to find dcm file"""
    for root, subFolders, files in os.walk(input_dir):
        for file in files:
            org_path = os.path.join(root, file)
        
            coordinate_list, height, width = get_coordinates(org_path)
            coord_sorted = arrange_coordinates(coordinate_list)
            angle_s, angle_i = calculate_angle(coord_sorted[0][0], coord_sorted[0][1], coord_sorted[1][0],
            coord_sorted[1][1])
            angle_s = -angle_s
            angle_i = -angle_i

            ''' warp defines the shape of the gaussian, whether or not it will be an oval or circle '''
            x_warp = 5 
            y_warp = 1
            
            ''' upper point save '''
            img1 = Image.fromarray(create_gaussian(coord_sorted[0][0], coord_sorted[0][1], height, width, angle_s, x_warp, y_warp).astype(np.uint8))
            img1 = img1.convert('L')
            img1.save(os.path.join(output_folder, f'{file.rsplit("_", 1)[0]}_smask.png'))
            
            ''' lower point save '''
            img2 = Image.fromarray(create_gaussian(coord_sorted[1][0], coord_sorted[1][1], height, width, angle_i, x_warp, y_warp).astype(np.uint8))
            img2 = img2.convert('L')
            img2.save(os.path.join(output_folder, f'{file.rsplit("_", 1)[0]}_imask.png'))
            
            
def main():
    parser = argparse.ArgumentParser(
        description='Create gaussian blobs on images with 1x1 white pixel on black background')
    parser.add_argument('input_dir')
    args = parser.parse_args()
    
    output_dir = f'{args.input_dir}_gaussian_elongated'
    
    os.mkdir(output_dir)
    walk(args.input_dir, output_dir)

main()