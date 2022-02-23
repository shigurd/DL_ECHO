import cv2
import os
import numpy as np
from PIL import Image
from operator import itemgetter
import csv
np.set_printoptions(threshold=np.inf)

def get_all_3_masks(img_pth):
    img_pil = Image.open(img_pth).convert('L')
    img_np = np.array(img_pil)

    #print(img_np)
    img_endo_np = np.where(img_np == 255 / 3, 255, 0)
    img_myo_np = np.where(img_np == 170, 255, 0)
    img_atri_np = np.where(img_np == 255 / 1, 255, 0)

    #temp1 = Image.fromarray(img_endo_np.astype(np.uint8)).convert('L')
    #temp2 = Image.fromarray(img_myo_np.astype(np.uint8)).convert('L')
    #temp3 = Image.fromarray(img_atri_np.astype(np.uint8)).convert('L')
    #temp1.show()
    #temp2.show()
    #temp3.show()
    return img_endo_np, img_myo_np, img_atri_np


def harris_corner_detection(img_gray_np):
    img_gray = img_gray_np.astype(np.uint8)

    img_gray = np.float32(img_gray)
    dst = cv2.cornerHarris(img_gray, 10, 1, 0.04)

    y_size, x_size = img_gray_np.shape
    corner_plot = np.zeros([y_size, x_size])
    corner_plot[dst > 0.1 * dst.max()] = 255
    '''
    cv2.imshow('dst', corner_plot)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    '''

    return corner_plot.astype(np.uint8)


def find_overlapping_corners_from_masks(img_endo_np, img_myo_np, img_atri_np):

    endo_corners = harris_corner_detection(img_endo_np)
    myo_corners = harris_corner_detection(img_myo_np)
    atri_corners = harris_corner_detection(img_atri_np)

    intersection_endo_myo = np.where(endo_corners == myo_corners, endo_corners, 0)
    intersection_endo_myo_atri = np.where(intersection_endo_myo == atri_corners, intersection_endo_myo, 0)

    #temp = Image.fromarray(intersection_endo_myo_atri)
    #temp.show()

    return intersection_endo_myo_atri


def locate_centroids(img_gray_overlap_np):
    img_gray_overlap_np = img_gray_overlap_np.astype(np.uint8)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(img_gray_overlap_np, 127, 255, 0)
    # find contours in the binary image
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    y_size, x_size = img_gray_overlap_np.shape
    coords_mask_np = np.zeros([y_size, x_size])

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        #print(M)
        # calculate x,y coordinate of center

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        coords_mask_np[cY, cX] = 255
        '''
        print(cX, cY)
        cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

        # display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    return coords_mask_np


def get_coords_list(coords_mask_np):
    coord_array_np = np.argwhere(coords_mask_np == 255)
    coord_list = []

    if coord_array_np.shape == (2, 2):
        ''' switch from matrix indices to coordinates in x,y '''
        [coord_list.append([e[1], e[0]]) for e in coord_array_np]

        ''' note that np indexes had been swapped into x,y coords in last step. therefore appropriate to use itemgetter 0 as x is changed from 1 to 0 '''
        coord_list = sorted(coord_list, key=itemgetter(0))

        l_x, l_y = coord_list[0]
        r_x, r_y = coord_list[1]

        return [[l_x, l_y], [r_x, r_y]]

    else:
        print(f'too few or too many points, found {coord_array_np.shape}: [points, coords per point]')

        return [['nan', 'nan'], ['nan', 'nan']]


def coord_to_pil_mask(x_size, y_size, x_index, y_index):
    mask_np = np.zeros([y_size, x_size])
    mask_np[y_index, x_index] = 255

    mask_pil = Image.fromarray(mask_np.astype(np.uint8))

    return mask_pil


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


if __name__ == '__main__':
    imgs_dir = r'C:\Users\Brekke\Downloads\CAMUS1800_complete\imgs'
    masks_dir = r'C:\Users\Brekke\Downloads\CAMUS1800_complete\masks'
    mask_output = r'C:\Users\Brekke\Downloads\CAMUS1800_complete\masks_MV'
    overlay_output = r'C:\Users\Brekke\Downloads\CAMUS1800_complete\overlay_MV'
    os.mkdir(mask_output)
    os.mkdir(overlay_output)

    csv_name = r'C:\Users\Brekke\Downloads\CAMUS1800_complete\masks_MV\CAMUS1800HML_coords.csv'
    csv_log = open(csv_name, 'w', newline='')
    writer = csv.writer(csv_log)
    writer.writerow(['file_name', 'l_x_pix', 'l_y_pix', 'r_x_pix', 'r_y_pix'])

    for x in os.listdir(masks_dir):
        input_mask = os.path.join(masks_dir, x)

        img_endo_np, img_myo_np, img_atri_np = get_all_3_masks(input_mask)

        overlap_greyscale_np = find_overlapping_corners_from_masks(img_endo_np, img_myo_np, img_atri_np)

        try:
            two_dots_mask_np = locate_centroids(overlap_greyscale_np)

            ''' coord_list: [[l_x, l_y], [r_x, r_y]] '''
            coord_list = get_coords_list(two_dots_mask_np)

            writer.writerow([x, coord_list[0][0], coord_list[0][1], coord_list[1][0], coord_list[1][1]])

            ''' will fail if there are not 2 coordinates, but these will be logged '''
            try:
                mask_l_pil = coord_to_pil_mask(256, 256, coord_list[0][0], coord_list[0][1])
                mask_r_pil = coord_to_pil_mask(256, 256, coord_list[1][0], coord_list[1][1])

                mask_l_pil.save(os.path.join(mask_output, f'{x.rsplit("_", 1)[0]}_lmask.png'))
                mask_r_pil.save(os.path.join(mask_output, f'{x.rsplit("_", 1)[0]}_rmask.png'))

                img_name = f'{x.rsplit("_", 1)[0]}.png'
                img_pth = os.path.join(imgs_dir, img_name)
                img_pil = Image.open(img_pth).convert('RGB')
                img_np = np.array(img_pil)

                img_1_point_np = draw_cross(img_np, coord_list[0][0], coord_list[0][1], radius=4, color=[0, 255, 0], rgb=True)
                img_2_point_np = draw_cross(img_1_point_np, coord_list[1][0], coord_list[1][1], radius=4, color=[0, 255, 0], rgb=True)

                img_2_point_pil = Image.fromarray(img_2_point_np)
                img_2_point_pil.save(os.path.join(overlay_output, img_name))
            except:
                print(x, 'wrong number of points, skipped')
        except:
            print(x, 'missing centroid, skipped')

    csv_log.close()







''' depracated '''

def locate_centroids_old(img_path):
    # read image through command line
    img = cv2.imread(img_path)
    # convert the image to grayscale

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    # find contours in the binary image
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    y_size, x_size = img.shape
    coords_mask_np = np.zeros([y_size, x_size])

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        coords_mask_np[cY, cX] = 255
        '''
        print(cX, cY)
        cv2.circle(img, (cX, cY), 1, (0, 255, 0), -1)

        # display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    return coords_mask_np


def iterate_dir(dir_path, output_path):
    """Walk through the inputDir to find dcm file"""
    for root, subFolders, files in os.walk(dir_path):
        for file in files:
            org_path = os.path.join(root, file)

            coords_mask_np = locate_centroids_old(org_path)

            coords_mask_pil = Image.fromarray((coords_mask_np).astype(np.uint8))
            coords_mask_pil = coords_mask_pil.convert('L')
            coords_mask_pil.save(os.path.join(output_path, f'{file.rsplit("_", 1)[0]}_mask.png'))
