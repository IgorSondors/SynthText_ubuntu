import numpy as np
import math
import h5py 
import re
import cv2
import random
from sklearn.linear_model import LinearRegression

import json
import time

start_time = time.time()

def lin_reg(point_x, point_y):
    x = np.array(point_x).reshape((-1, 1))
    y = np.array(point_y)

    model = LinearRegression().fit(x, y)
    #print('intercept:', model.intercept_)
    #print('slope:', model.coef_)
    k = model.coef_[0]
    b = model.intercept_
    return k, b, model

def kx_plus_b(bottom_x, bottom_y):

    x_plus_delta = []
    y_plus_delta = []
    pixel_step = 1
    num_dots_poligon = 0
    for i in range(len(bottom_x) - 1):
        next_x = int(bottom_x[i])
        next_y = bottom_y[i]
        x_plus_delta.append(next_x)
        y_plus_delta.append(round(next_y,1))
        num_dots_poligon = num_dots_poligon + 1
        try:
            k = (bottom_y[i] - bottom_y[i+1])/(bottom_x[i] - bottom_x[i+1])
            b = bottom_y[i] - k * bottom_x[i]
            num_dots = int((bottom_x[i+1] - bottom_x[i]) / pixel_step)
            
            for j in range(num_dots):
                next_x = next_x + pixel_step
                next_y = k * next_x + b
                x_plus_delta.append(next_x)
                y_plus_delta.append(round(next_y,1))
            num_dots_poligon = num_dots_poligon + num_dots
        except:
            print(bottom_y[i],bottom_y[i+1],bottom_x[i],bottom_x[i+1])
            print('ZeroDivision')
    x_plus_delta.append(bottom_x[-1])
    y_plus_delta.append(round(bottom_y[-1],1))
    num_dots_poligon = num_dots_poligon + 1

    return num_dots_poligon, x_plus_delta, y_plus_delta

def find_4_dots(point_x, point_y):
    print(point_x, point_y)
    i = 0
    curve = []  #c**2 = a**2 + b**2 - 2ab*cos_alpha
    
    while i <= len(point_x) - 2: # except last angle
        # print(i, 'of', len(point_x) - 1)
        x1 =  point_x[i]
        y1 = point_y[i]
        x2 =  point_x[i + 1]
        y2 = point_y[i + 1]
        x0 =  point_x[i - 1]
        y0 = point_y[i - 1]
        
        a = ((x2 - x1)**2+(y2 - y1)**2)**0.5
        b = ((x1 - x0)**2+(y1 - y0)**2)**0.5
        c = ((x2 - x0)**2+(y2 - y0)**2)**0.5

        cos_alpha = (a**2 + b**2 - c**2)/(2*a*b)
        # print(cos_alpha)
        curve.append(abs(cos_alpha))
        i = i + 1
    # last angle    
    x1 =  point_x[-1]
    y1 = point_y[-1]
    x2 =  point_x[0]
    y2 = point_y[0]
    x0 =  point_x[-2]
    y0 = point_y[-2]
    
    a = ((x2 - x1)**2+(y2 - y1)**2)**0.5
    b = ((x1 - x0)**2+(y1 - y0)**2)**0.5
    c = ((x2 - x0)**2+(y2 - y0)**2)**0.5   

    cos_alpha = (a**2 + b**2 - c**2)/(2*a*b)

    curve.append(abs(cos_alpha))

    #print(len(curve), curve)
    #print(len(point_x), point_x)

    curve_out_of_edges = curve.copy()
    edge_angles = []
    edge_x = []
    edge_y = []
    for i in range(4):
        edge_angles.append(min(curve_out_of_edges))
        curve_out_of_edges.remove(min(curve_out_of_edges))
        #print(len(curve_out_of_edges))
        #print(len(curve))
    #print(edge_angles)

    for i in edge_angles:
        index = curve.index(i)
        #print(index)
        edge_x.append(point_x[index])
        edge_y.append(point_y[index])

    edge_x, edge_y = zip(*sorted(zip(edge_x, edge_y)))
    return edge_x, edge_y

def find_bbox_coord(point_x, point_y):

    is_good_rect = True
    bottom_x, bottom_y = [], []
    top_x, top_y = [], []

    if len(point_x) == 4:
        is_square = False
        #((x2 - x1)**2+(y2 - y1)**2)**0.5
        if is_square:
            top_x, top_y = point_x[2:], point_y[2:]
            bottom_x, bottom_y = point_x[:-2], point_y[:-2]
        else:
            k, b, model = lin_reg(point_x, point_y)

            if point_y[0] > k * point_x[0] + b:
                is_good_rect = False     
            for i in range(len(point_x)):
                if point_y[i] > k * point_x[i] + b:
                    bottom_x.append(point_x[i])
                    bottom_y.append(point_y[i])
                else:
                    top_x.append(point_x[i])
                    top_y.append(point_y[i])
    elif len(point_x) > 4:
        edge_x, edge_y = find_4_dots(point_x, point_y)
        #print('point_x = ', point_x)
        #print('point_y = ', point_y)
        k, b, model = lin_reg(edge_x, edge_y)

        bottom_edge_x = []
        bottom_edge_y = []
        top_edge_x = []
        top_edge_y = []
        for i in range(len(edge_x)):
                if edge_y[i] > k * edge_x[i] + b:
                    bottom_edge_x.append(edge_x[i])
                    bottom_edge_y.append(edge_y[i])
                else:
                    top_edge_x.append(edge_x[i])
                    top_edge_y.append(edge_y[i])
        
        if point_x.index(top_edge_x[0]) == 0:
            print('Первая точка в левом верхнем')
            #print('point_x = ', len(point_x), point_x)
            #print('point_y = ', len(point_y), point_y)
            left_top_ind = 0
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]
            #print('top_x = ', len(top_x), top_x)
            #print('top_y = ', len(top_y), top_y)

            left_down_ind = point_x.index(bottom_edge_x[0])
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
            #print('bottom_x = ', len(bottom_x), bottom_x)
            #print('bottom_y = ', len(bottom_y), bottom_y)
        elif top_edge_x[0] == point_x[-1]:
            print('Первая точка сверху после угла')
            left_top_ind = -1
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = [point_x[-1]], [point_y[-1]]
            top_x = top_x + point_x[0 : right_top_ind + 1]
            top_y = top_y + point_y[0 : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
        elif bottom_edge_x[0] == point_x[0]:
            print('bottom_edge_x = ', bottom_edge_x)
            print('bottom_edge_y = ', bottom_edge_y)
            print('top_edge_x = ', top_edge_x)
            print('top_edge_x = ', top_edge_y)
            print('Первая точка в левом нижнем')
            print('edge_x, edge_y', edge_x, edge_y)
            print('point_x = ', len(point_x), point_x)
            print('point_y = ', len(point_y), point_y)
            left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]
            print('top_x = ', len(top_x), top_x)
            print('top_y = ', len(top_y), top_y)

            left_down_ind = 0
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
            bottom_x, bottom_y = [point_x[0]] + bottom_x, [point_y[0]] + bottom_y
            print('bottom_x = ', len(bottom_x), bottom_x)
            print('bottom_y = ', len(bottom_y), bottom_y)
        else:
            print('Первая точка в другом положении')
            left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
        #    index_bottom = point_x.index(bottom_edge_x[i])
        #    index_top = point_x.index(top_edge_x[i])

    bottom_x, bottom_y = zip(*sorted(zip(bottom_x, bottom_y)))   
    top_x, top_y = zip(*sorted(zip(top_x, top_y)))
    
    return is_good_rect, bottom_x, bottom_y, top_x, top_y
    
def draw_black_rect(img_gray, point_pairs):
    cnt = []
    for i in range(len(point_pairs) - 2, -2, -2):
        start_point = (point_pairs[i], point_pairs[i + 1])
        end_point = (point_pairs[i - 2], point_pairs[i + 1 - 2])
        cnt.append(start_point) 
        cnt.append(end_point) 
    black = cv2.fillPoly(img_gray, np.array([cnt]), (0,0,0), lineType=cv2.LINE_AA)
    return black

def top_down_points(img_counter, bottom_x, bottom_y, top_x, top_y):
    im = cv2.imread('./real_frames/image_{}.jpg'.format(img_counter))
    # Radius of circle
    radius = 3
    thickness = 3
    for i in range(len(bottom_x)):
        x, y = bottom_x[i], bottom_y[i]
        center_coordinates = (int(x), int(y))
        color = (0, 0, 255)
        im = cv2.circle(im, center_coordinates, radius, color, thickness)
    
    for i in range(len(top_y)):
        x, y = top_x[i], top_y[i] 
        center_coordinates = (int(x), int(y))
        color = (255, 0, 0)
        im = cv2.circle(im, center_coordinates, radius, color, thickness)

    cv2.imwrite('./real_frames/image_{}.jpg'.format(img_counter), im)
    return im

with open('input.txt', encoding = 'utf8') as fp:
    lines = fp.readlines()
    img_counter = -1
    my_ch_label = open('./ds_images.csv', 'w')

    for line in lines:
        js_line = json.loads(line)
        file_name = js_line['file'].split('/')
        file_name = file_name[-1].split('?')
        #file_path = '../Images/' + file_name[-1]
        file_path = './Images/' + file_name[0]
        im = cv2.imread(file_path)
        if im is None:
            continue

        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        img_counter = img_counter + 1
        cv2.imwrite('./real_frames/image_{}.jpg'.format(img_counter), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.imwrite('./real_frames/{}'.format(file_name[0]), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        height, width = img_gray.shape[:2]
        my_ch_label.write('image_{}.jpg'.format(img_counter) + ',' + str(height) + ',' + str(width))

        All_x_under_word, All_y_under_word, All_w_char_list, All_h_char_list, All_char_list = [], [], [], [], []
        All_dots_of_im = 0

        second = js_line['result']
        # second - один словарик для одного из изображений из множества {"file":"name.jpg","result":[...]}, 
        # включающий в себя [...] = [{"type":"polygon","data":[{"x":0.4,"y":0.4}, {}...], "readable":t/f}, ...]
        poligon_counter = 0
        good_rect_counter = 0
        bad_rect_couner = 0
        for s in second: # s - один из множества полигонов одной картинки
            poligon_counter = poligon_counter + 1
            color = 'red'
            if s['readable']:
                color = 'green'
            data = s['data']
            
            point_pairs = []
            point_x = []
            point_y = []

            for d in data: # Множество точек одного полигона
                x = d['x']*width
                y = d['y']*height

                point_x.append(int(x))
                point_y.append(int(y))

                point_pairs.append(int(x))
                point_pairs.append(int(y))
            try:
                if color == 'green':  
                    # функция определяющая где верх и низ полигона, возвращающая крайние точки
                    is_good_rect, bottom_x, bottom_y, top_x, top_y = find_bbox_coord(point_x, point_y)
                if color == 'red':
                    is_good_rect = False
                if is_good_rect:
                    # kx_plus_b, возвращающая точки под полигоном с шагом 1 пикс по оХ
                    num_dots, x_plus_delta, y_plus_delta = kx_plus_b( bottom_x, bottom_y)

                    num_useless, x_plus_delta_top, y_plus_delta_top = kx_plus_b( top_x, top_y)

                    img_gray = top_down_points(img_counter, x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top)

                    good_rect_counter = good_rect_counter + 1  
                    #print('good_rect_counter =', good_rect_counter)
                    #print('num_dots = ', num_dots)          
                else:
                    black = draw_black_rect(img_gray, point_pairs)
                    cv2.imwrite('./real_frames/image_{}.jpg'.format(img_counter), black, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    #cv2.imwrite('./real_frames/{}'.format(file_name[0]), black, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    num_dots = 0
                    x_plus_delta, y_plus_delta = [], []
                    bad_rect_couner = bad_rect_couner + 1
            except:
                    cv2.imwrite('./error_img/{}'.format(file_name[0]), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    

            """print('point_x = ', point_x)
            print('point_y = ', point_y)   
                        
            """

        #print(file_name[0])
        #print('poligon_counter = ', poligon_counter)
        #print('good_rect_counter = ', good_rect_counter) 
        #print('bad_rect_couner = ', bad_rect_couner) 

print('end_time = ', time.time() - start_time)

