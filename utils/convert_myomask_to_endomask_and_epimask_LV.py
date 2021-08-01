import cv2
import os
from PIL import Image
import numpy as np

def connect_trace_and_fill(coords_list_xy):
    
    zeros = np.zeros((256, 256), np.uint8)
    for i in range(len(coords_list_xy)):
        
        x0 = int(coords_list_xy[i-1][0])
        y0 = int(coords_list_xy[i-1][1])
        x1 = int(coords_list_xy[i][0])
        y1 = int(coords_list_xy[i][1])
        
        cv2.line(zeros,(y0, x0), (y1, x1), (255, 255, 255), 1)
    '''
    cv2.imshow('trace', zeros)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    #fills contour
    thresh = cv2.threshold(zeros, 120, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(zeros, [c], -1, (255,255,255), -1)
    #
    '''
    cv2.imshow('trace', zeros)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    return zeros

def circle_scanner(center_x, center_y, edges):
    ext_roi_x = np.array([])
    ext_roi_y = np.array([])
    int_roi_x = np.array([])
    int_roi_y = np.array([])
    x_coords = np.array([])
    y_coords = np.array([])

    # create coordinates around image outer frame. to start straight up, so "black" area does not wrap
    for j in range(128, 1, -1):
        x_coords = np.append(x_coords, 0)
        y_coords = np.append(y_coords, j)
    for i in range(0, 255):
        x_coords = np.append(x_coords, i)
        y_coords = np.append(y_coords, 0)
    for j in range(1, 254):
        x_coords = np.append(x_coords, 255)
        y_coords = np.append(y_coords, j)
    for i in range(255, 0, -1):
        x_coords = np.append(x_coords, i)
        y_coords = np.append(y_coords, 255)
    for j in range(254, 129, -1):
        x_coords = np.append(x_coords, 0)
        y_coords = np.append(y_coords, j)

    # re order coordinates to it starts and ends the correct place
    consecutive_black = 0
    was_black = False
    first_black = 0
    last_black = 0

    for i in range(0, len(x_coords)):
        contour = np.zeros((256, 256), np.uint8)
        cv2.line(contour, (center_x, center_y), (int(y_coords[i]), int(x_coords[i])), (255, 255, 255))
        
        '''
        #this is the circle scanner
        cv2.imshow('contour',contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        # use circle scanner to find black area(mitral valve)
        overlap = cv2.bitwise_and(contour, edges)
        intersections = np.where(overlap != [0])
        #print(intersections)
        if np.size(intersections[0]) == 0:
            if was_black == True:
                consecutive_black += 1
                if (consecutive_black > 20):
                    first_black = i - consecutive_black

            was_black = True
        else:
            if (consecutive_black > 20):
                last_black = i
            consecutive_black = 0
            was_black = False

    # extract scanner intersections that are white
    only_white = np.split(x_coords, [first_black, last_black])
    x_coords = np.append(only_white[2], only_white[0])
    only_white = np.split(y_coords, [first_black, last_black])
    y_coords = np.append(only_white[2], only_white[0])

    # find intersections on external and internal boarder
    toshow = np.zeros((256, 256), np.uint8)
    for i in range(0, len(x_coords)):
        contour = np.zeros((256, 256), np.uint8)

        cv2.line(contour, (center_x, center_y), (int(y_coords[i]), int(x_coords[i])), (255, 255, 255), 2)
        '''
        cv2.imshow('contour',contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        overlap = cv2.bitwise_and(contour, edges)

        intersections = np.where(overlap != [0])
        
        dist = 256 * 256  # large number
        dist_furthest = 0

        point_to_add_x = 0
        point_to_add_y = 0
        point_to_add_x_furthest = 0
        point_to_add_y_furthest = 0

        # print("find intersections")
        for i in range(0, len(intersections[0])):
            # print(intersections[0][i])
            newdist = (intersections[1][i] - center_x) ** 2 + (intersections[0][i] - center_y) ** 2
            if (newdist < dist):
                point_to_add_x = intersections[0][i]
                point_to_add_y = intersections[1][i]
                dist = newdist
            if (newdist > dist_furthest):
                point_to_add_x_furthest = intersections[0][i]
                point_to_add_y_furthest = intersections[1][i]
                dist_furthest = newdist

        if (point_to_add_x > 0 and point_to_add_y > 0):
            int_roi_x = np.append(int_roi_x, point_to_add_x)
            int_roi_y = np.append(int_roi_y, point_to_add_y)
        
        if (point_to_add_x_furthest > 0 and point_to_add_y_furthest > 0):
            ext_roi_x = np.append(ext_roi_x, point_to_add_x_furthest)
            ext_roi_y = np.append(ext_roi_y, point_to_add_y_furthest)

    int_roi = [(int_roi_x[i], int_roi_y[i]) for i in range(0, len(int_roi_x))]
    ext_roi = [(ext_roi_x[j], ext_roi_y[j]) for j in range(0, len(ext_roi_x))]

    # remove duplicates
    final_coord_int = []
    for x in int_roi:
        if x not in final_coord_int:
            final_coord_int.append(x)
    final_coord_int = np.array(final_coord_int)
    
    final_coord_ext = []
    for x in ext_roi:
        if x not in final_coord_ext:
            final_coord_ext.append(x)
    final_coord_ext = np.array(final_coord_ext)
    '''
    #toshow int_contour
    toshow1 = np.zeros((256, 256), np.uint8)
    for g in final_coord:
        toshow1[int(g[0])][int(g[1])] = 255
    toshow_pil = Image.fromarray(toshow1)
    toshow_pil.show()
    '''
    int_roi = final_coord_int
    int_roi_filled = connect_trace_and_fill(int_roi)
    
    ext_roi = final_coord_ext
    ext_roi_filled = connect_trace_and_fill(ext_roi)
   
    return int_roi_filled, ext_roi_filled
    
    
def get_endocard_epicard(filePath):
    #print ("read file: " + filePath)
    mask = cv2.imread(filePath)

    #cv2.imshow('mask',mask)
    #cv2.waitKey(0)
    
    img_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(mask, 256, 256)
    
    #cv2.imshow('edges',edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    M = cv2.moments(img_gray)  # find center of mass
    Mx = int(M["m10"] / M["m00"])
    My = int(M["m01"] / M["m00"])

    int_roi_filled, ext_roi_filled = circle_scanner(Mx, My, edges)
    # print('Mask Images ready!')
    return int_roi_filled, ext_roi_filled
    
if __name__ == '__main__':
    
    data_name = "LV_3CH_HM_K1"
    mask_folder = f'data/train_masks_{data_name}'
    
    epi_folder = f'{data_name.split("_", 1)[0]}END_{data_name.split("_", 1)[-1]}'
    end_folder = f'{data_name.split("_", 1)[0]}EPI_{data_name.split("_", 1)[-1]}'
    
    os.mkdir(epi_folder)
    os.mkdir(end_folder)
    
    for m in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, m)
        
        int_roi_filled, ext_roi_filled = get_endocard_epicard(mask_path)
        
        int_roi_filled_pil = Image.fromarray(int_roi_filled.astype(np.uint8))
        ext_roi_filled_pil = Image.fromarray(ext_roi_filled.astype(np.uint8))
        int_roi_filled_pil.save(f'{end_folder}/{m}')
        ext_roi_filled_pil.save(f'{epi_folder}/{m}')
        
        
