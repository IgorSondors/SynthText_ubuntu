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

def find_bbox_coord(point_x, point_y):
    k, b, model = lin_reg(point_x, point_y)
    is_good_rect = True
    bottom_x, bottom_y = [], []
    top_x, top_y = [], []

    if point_y[0] > k * point_x[0] + b:
        is_good_rect = False     

    for i in range(len(point_x)):
        if point_y[i] > k * point_x[i] + b:
            bottom_x.append(point_x[i])
            bottom_y.append(point_y[i])
        else:
            top_x.append(point_x[i])
            top_y.append(point_y[i])
    
    topleft_x = min(top_x)       

    if topleft_x == point_x[0]:
        # find top edges
        topleft_x = point_x[0]
        topleft_y = point_y[0]

        topright_x = max(top_x) 
        topright_index = top_x.index(topright_x)
        topright_y = top_y[topright_index]
        
        # find down edges
        bottomleft_x = min(bottom_x)
        bottomleft_index = bottom_x.index(bottomleft_x)
        bottomleft_y = bottom_y[bottomleft_index]

        bottomright_x = max(bottom_x)
        bottomright_index = bottom_x.index(bottomright_x)
        bottomright_y = bottom_y[bottomright_index]

        bbox_height = ((topleft_x - bottomleft_x)**2+(topleft_y - bottomleft_y)**2)**0.5

        if int(bbox_height) > 150:
            is_good_rect = False
    else:
        is_good_rect = False
        # find top edges
        topleft_x = min(top_x) 
        topleft_index = top_x.index(topleft_x)
        topleft_y = top_y[topleft_index]

        topright_x = max(top_x) 
        topright_index = top_x.index(topright_x)
        topright_y = top_y[topright_index]

        # find down edges
        bottomleft_x = min(bottom_x)
        bottomleft_index = bottom_x.index(bottomleft_x)
        bottomleft_y = bottom_y[bottomleft_index]

        bottomright_x = max(bottom_x)
        bottomright_index = bottom_x.index(bottomright_x)
        bottomright_y = bottom_y[bottomright_index]

        #bbox_height = ((topleft_x - bottomleft_x)**2+(topleft_y - bottomleft_y)**2)**0.5    

    return is_good_rect, bottomleft_x, bottomleft_y, bottomright_x, bottomright_y, topleft_x, topleft_y, topright_x, topright_y

def kx_plus_b(bottomleft_x, bottomleft_y, bottomright_x, bottomright_y):
    pixel_step = 1
    k = (bottomleft_y - bottomright_y)/(bottomleft_x - bottomright_x)
    b = bottomleft_y - k * bottomleft_x
    num_dots = int((bottomright_x - bottomleft_x) / pixel_step)
    x_plus_delta = []
    y_plus_delta = []

    next_x = int(bottomleft_x)
    next_y = bottomleft_y
    x_plus_delta.append(next_x)
    y_plus_delta.append(round(next_y,1))
    for i in range(num_dots):
        next_x = next_x + pixel_step
        next_y = k * next_x + b
        x_plus_delta.append(next_x)
        y_plus_delta.append(round(next_y,1))
    num_dots = num_dots + 1
    return num_dots, x_plus_delta, y_plus_delta
    
def draw_black_rect(img_gray, bottomleft_x, bottomleft_y, bottomright_x, bottomright_y, topleft_x, topleft_y, topright_x, topright_y):
    cnt = [[topleft_x, topleft_y], [topright_x, topright_y], [bottomright_x, bottomright_y], [bottomleft_x, bottomleft_y]]
    black = cv2.fillPoly(img_gray, np.array(cnt), (0,255,0))
    return black

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

        height, width = img_gray.shape[:2]
        my_ch_label.write('image_{}.jpg'.format(img_counter) + ',' + str(height) + ',' + str(width))

        All_x_under_word, All_y_under_word, All_w_char_list, All_h_char_list, All_char_list = [], [], [], [], []
        All_dots_of_im = 0

        second = js_line['result']
        # second - один словарик для одного из изображений из множества {"file":"name.jpg","result":[...]}, 
        # включающий в себя [...] = [{"type":"polygon","data":[{"x":0.4,"y":0.4}, {}...], "readable":t/f}, ...]
        for s in second: # s - один из множества полигонов одной картинки
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
                
            # функция определяющая где верх и низ полигона, возвращающая крайние точки
            is_good_rect, bottomleft_x, bottomleft_y, bottomright_x, bottomright_y, topleft_x, topleft_y, topright_x, topright_y = find_bbox_coord(point_x, point_y)
            if color == 'red':
                is_good_rect = False
            if is_good_rect:
                # kx_plus_b, возвращающая точки под полигоном с шагом 1 пикс по оХ
                num_dots, x_plus_delta, y_plus_delta = kx_plus_b(bottomleft_x, bottomleft_y, bottomright_x, bottomright_y)            
            else:
                black = draw_black_rect(img_gray, bottomleft_x, bottomleft_y, bottomright_x, bottomright_y, topleft_x, topleft_y, topright_x, topright_y)
                cv2.imwrite('./real_frames/image_{}.jpg'.format(img_counter), black, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                num_dots = 0
                x_plus_delta, y_plus_delta = [], []
            print(file_name[0])
            print('point_x = ', point_x)
            print('point_y = ', point_y)
                        
            All_dots_of_im = All_dots_of_im + num_dots
            All_x_under_word = All_x_under_word + x_plus_delta
            All_y_under_word = All_y_under_word + y_plus_delta

        my_ch_label.write(',' + str(All_dots_of_im))
        All_w_char = 10
        All_h_char = 10
        All_char = '.'
        for z in range(All_dots_of_im):
            my_ch_label.write(',' + str(All_x_under_word[z]) + ',' + str(All_y_under_word[z]))
            my_ch_label.write(',' + str(All_w_char) + ',' + str(All_h_char) + ',' + str(All_char))         

        my_ch_label.write('\n')

print('end_time = ', time.time() - start_time)

