import numpy as np
import h5py 
import cv2
import json
import time
from sklearn.linear_model import LinearRegression

start_time = time.time()

def lin_reg(point_x, point_y):
    x = np.array(point_x).reshape((-1, 1))
    y = np.array(point_y)

    model = LinearRegression().fit(x, y)

    k = model.coef_[0]
    b = model.intercept_
    return k, b

def draw_black_rect(img_gray, point_pairs):
    cnt = []
    for i in range(len(point_pairs) - 2, -2, -2):
        start_point = (point_pairs[i], point_pairs[i + 1])
        end_point = (point_pairs[i - 2], point_pairs[i + 1 - 2])
        cnt.append(start_point) 
        cnt.append(end_point) 
    black = cv2.fillPoly(img_gray, np.array([cnt]), (0,0,0), lineType=cv2.LINE_AA)
    return black
    
def kx_plus_b(bottom_x, bottom_y):

    x_plus_delta = []
    y_plus_delta = []
    pixel_step = 1
    poligon_dots = 0
    for i in range(len(bottom_x) - 1):
        next_x = int(bottom_x[i])
        next_y = bottom_y[i]
        x_plus_delta.append(next_x)
        y_plus_delta.append(round(next_y,1))
        poligon_dots = poligon_dots + 1
        try:
            k = (bottom_y[i] - bottom_y[i+1])/(bottom_x[i] - bottom_x[i+1])
            b = bottom_y[i] - k * bottom_x[i]
            dots_between_edges = int((bottom_x[i+1] - bottom_x[i]) / pixel_step)
            
            for j in range(dots_between_edges):
                next_x = next_x + pixel_step
                next_y = k * next_x + b
                x_plus_delta.append(next_x)
                y_plus_delta.append(round(next_y,1))
            poligon_dots = poligon_dots + dots_between_edges
        except:
            print(bottom_y[i],bottom_y[i+1],bottom_x[i],bottom_x[i+1])
            print('ZeroDivision')
    x_plus_delta.append(bottom_x[-1])
    y_plus_delta.append(round(bottom_y[-1],1))
    poligon_dots = poligon_dots + 1

    return poligon_dots, x_plus_delta, y_plus_delta

def find_4_dots(point_x, point_y):
    #print(point_x, point_y)
    i = 0
    curve = []  #c**2 = a**2 + b**2 - 2ab*cos_alpha
    
    while i <= len(point_x) - 2: # except last angle
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

    curve_out_of_edges = curve.copy()
    edge_angles = []
    edge_x = []
    edge_y = []
    for i in range(4):
        edge_angles.append(min(curve_out_of_edges))
        curve_out_of_edges.remove(min(curve_out_of_edges))

    for i in edge_angles:
        index = curve.index(i)

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
        # Проверка соотношения сторон
        if is_square:   # Проверить наклон ребер, если он небольшой далее:
            top_x, top_y = point_x[2:], point_y[2:]
            bottom_x, bottom_y = point_x[:-2], point_y[:-2]
        else:
            k, b = lin_reg(point_x, point_y)

            #if point_y[0] > k * point_x[0] + b:
            #    is_good_rect = False     
            for i in range(len(point_x)):
                if point_y[i] > k * point_x[i] + b:
                    bottom_x.append(point_x[i])
                    bottom_y.append(point_y[i])
                else:
                    top_x.append(point_x[i])
                    top_y.append(point_y[i])
    elif len(point_x) > 4:
        edge_x, edge_y = find_4_dots(point_x, point_y)

        k, b = lin_reg(edge_x, edge_y)

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
        
        # Переделать нахождение дна и верхушки полигона
        if point_x.index(top_edge_x[0]) == 0:
            print('Первая точка в левом верхнем')
            left_top_ind = 0
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
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
            print('Первая точка в левом нижнем')
            left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = 0
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
            bottom_x, bottom_y = [point_x[0]] + bottom_x, [point_y[0]] + bottom_y
        else:
            print('Первая точка в другом положении')
            is_good_rect = False

            """left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            if bottom_edge_x[1] == top_edge_x[1]:
                right_down_ind = point_x.index(bottom_edge_x[1]) + 1
            else:
                right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]"""

    if is_good_rect:

        bottom_x, bottom_y = zip(*sorted(zip(bottom_x, bottom_y)))   
        top_x, top_y = zip(*sorted(zip(top_x, top_y)))

        bbox_height_left = ((top_x[0] - bottom_x[0])**2+(top_y[0] - bottom_y[0])**2)**0.5
        bbox_height_right = ((top_x[-1] - bottom_x[-1])**2+(top_y[-1] - bottom_y[-1])**2)**0.5
        mid_arithmetic_h = round((bbox_height_left + bbox_height_right)/2, 1)
        if mid_arithmetic_h > 150:
            is_good_rect = False
    else:
        bottom_x, bottom_y, top_x, top_y = [], [], [], []
        mid_arithmetic_h = 0
    
    return is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h

with open('input.txt', encoding = 'utf8') as fp:
    lines = fp.readlines()
    img_counter = -1
    data_annotation = open('./ds_images.csv', 'w')

    for line in lines:
        js_line = json.loads(line)
        file_name = js_line['file'].split('/')
        file_name = file_name[-1].split('?')
        file_path = './Images - Copy/' + file_name[0]
        im = cv2.imread(file_path)
        if im is None:
            continue
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape[:2]

        second = js_line['result']
        try:
            img_counter = img_counter + 1
            
            All_x_under_word, All_y_under_word, All_word_height = [], [], []
            All_image_dots = 0            
            # second - один словарик для одного из изображений из множества {"file":"name.jpg","result":[...]}, 
            # включающий в себя [...] = [{"type":"polygon","data":[{"x":0.4,"y":0.4}, {}...], "readable":t/f}, ...]
            for s in second: # s - один из множества полигонов одной картинки
                is_good_rect = False
                if s['readable']:
                    is_good_rect = True
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

                if is_good_rect:
                    # функция определяющая где верх и низ полигона и возвращающая их точки
                    is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h = find_bbox_coord(point_x, point_y)

                    # kx_plus_b, возвращающая точки под полигоном с шагом 1 пикс по оХ
                    poligon_dots, x_plus_delta, y_plus_delta = kx_plus_b(bottom_x, bottom_y)    
                else:
                    black = draw_black_rect(img_gray, point_pairs)

                    img_gray = black
                    poligon_dots = 0
                    mid_arithmetic_h = 'empty'
                    x_plus_delta, y_plus_delta = [], []
                                
                All_image_dots = All_image_dots + poligon_dots
                All_x_under_word = All_x_under_word + x_plus_delta
                All_y_under_word = All_y_under_word + y_plus_delta
                All_word_height = All_word_height + [mid_arithmetic_h for i in range(poligon_dots)]

            cv2.imwrite('./real_frames/image_{}.jpg'.format(img_counter), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            data_annotation.write('image_{}.jpg'.format(img_counter) + ',' + str(height) + ',' + str(width))
            
            data_annotation.write(',' + str(All_image_dots))
            All_w_char = 13
            All_char = '.'
            for z in range(All_image_dots):
                data_annotation.write(',' + str(All_x_under_word[z]) + ',' + str(All_y_under_word[z]))
                data_annotation.write(',' + str(All_w_char) + ',' + str(All_word_height[z]) + ',' + str(All_char))         
            data_annotation.write('\n')
            
        except:
            cv2.imwrite('./error_img/{}'.format(file_name[0]), img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print('end_time = ', time.time() - start_time)
