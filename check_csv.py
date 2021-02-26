"""
check csv writing 

"""
import numpy as np
import h5py 
import cv2
import json
import time

start_time = time.time()

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
            dots_between_edges = int((((bottom_x[i+1] - bottom_x[i])**2 + (bottom_y[i+1] - bottom_y[i])**2)**0.5) / pixel_step)
            X_step = (bottom_x[i+1] - bottom_x[i]) / dots_between_edges
            for j in range(dots_between_edges):
                next_x = next_x + X_step
                next_y = k * next_x + b
                x_plus_delta.append(next_x)
                y_plus_delta.append(round(next_y,1))
            poligon_dots = poligon_dots + dots_between_edges
        except:
            print('Расстояние между точками 0 пикселей! Пропуск точки')

    x_plus_delta.append(int(bottom_x[-1]))
    y_plus_delta.append(round(bottom_y[-1],1))
    poligon_dots = poligon_dots + 1
    return poligon_dots, x_plus_delta, y_plus_delta

def lin_reg(point_x, point_y):
    i = 0
    six_line_dict = {}
    while i <= 2:

        x1 =  point_x[i]
        y1 = point_y[i]
        x2 =  point_x[i + 1]
        y2 = point_y[i + 1]
        x0 =  point_x[i - 1]
        y0 = point_y[i - 1]
        
        a = ((x2 - x1)**2+(y2 - y1)**2)**0.5
        b = ((x1 - x0)**2+(y1 - y0)**2)**0.5
        c = ((x2 - x0)**2+(y2 - y0)**2)**0.5

        six_line_dict[a] = [[x1, x2], [y1, y2]]
        six_line_dict[b] = [[x0, x1], [y0, y1]]

        i = i + 1


    six_line_length = six_line_dict.keys()
    two_long_ribs = (sorted(six_line_length))[2:4]

    first_line_xy = six_line_dict[two_long_ribs[0]]
    second_line_xy = six_line_dict[two_long_ribs[1]]

    Y_sum1 = first_line_xy[1][0] + first_line_xy[1][1]
    Y_sum2 = second_line_xy[1][0] + second_line_xy[1][1]
    if Y_sum1 > Y_sum2:
        possible_bottom_y = first_line_xy[1]
    else:
        possible_bottom_y = second_line_xy[1]

    return possible_bottom_y

def find_bbox_coord(point_x, point_y):
    is_good_rect = True
    bottom_x, bottom_y = [], []
    top_x, top_y = [], []
    delta = 10**(-4)

    if len(point_x) == 4:
        out_of_repeats = []
        for j in range(len(point_x)): # add delta for the reason of not mess in equal ribs
            delta = 10**(-6)
            out_of_repeats.append(point_x[j] + delta*j)
        point_x = out_of_repeats

        out_of_repeats = []
        for j in range(len(point_y)): # add delta for the reason of not mess in equal ribs
            delta = 10**(-6)
            out_of_repeats.append(point_y[j] + delta*j)
        point_y = out_of_repeats

        is_square = False
        
        quadrate_width = ((point_x[1] - point_x[0])**2+(point_y[1] - point_y[0])**2)**0.5
        quadrate_height = ((point_x[1] - point_x[2])**2+(point_y[1] - point_y[2])**2)**0.5
        aspect_ratio = quadrate_width / quadrate_height
        if aspect_ratio > 0.7 and aspect_ratio < 1.3:
            is_square = True
        if quadrate_width * quadrate_height < 500:           
            is_square = True
        if is_square: 
            #print('quadrate_S = ', quadrate_width * quadrate_height)
            k1 = (point_y[0] - point_y[2])/(point_x[0] - point_x[2])
            k2 = (point_y[1] - point_y[3])/(point_x[1] - point_x[3])
            #print('Квадрат')
            if k1 < 0 or k2 > 0:
                is_good_rect = False # too much tilt more then pi/4
                #print('Квадрат сильно наклонен')
            top_x, top_y = point_x[:2], point_y[:2]
            bottom_x, bottom_y = point_x[2:], point_y[2:]
        else:
            #print('Прямоугольник')
            possible_bottom_y = lin_reg(point_x, point_y)
   
            for i in range(len(point_x)):
                if point_y[i] in possible_bottom_y:#- 0.6
                    bottom_x.append(point_x[i])
                    bottom_y.append(point_y[i])
                else:
                    top_x.append(point_x[i])
                    top_y.append(point_y[i])

    elif len(point_x) > 4:
        #print('Многоугольник')
        out_of_repeats = []
        for j in range(len(point_x)): # add delta for the reason of not mess in equal angles
            out_of_repeats.append(point_x[j] + delta*j)
        point_x = out_of_repeats

        out_of_repeats = []
        for j in range(len(point_y)): # add delta for the reason of not mess in equal angles
            out_of_repeats.append(point_y[j] + delta*j)
        point_y = out_of_repeats

        edge_x, edge_y = find_4_dots(point_x, point_y)

        possible_bottom_y = lin_reg(edge_x, edge_y)

        bottom_edge_x = []
        bottom_edge_y = []
        top_edge_x = []
        top_edge_y = []
        for i in range(len(edge_x)):
                if edge_y[i] in possible_bottom_y:
                    bottom_edge_x.append(edge_x[i])
                    bottom_edge_y.append(edge_y[i])
                else:
                    top_edge_x.append(edge_x[i])
                    top_edge_y.append(edge_y[i])

        bottom_edge_x, bottom_edge_y = zip(*sorted(zip(bottom_edge_x, bottom_edge_y)))   
        top_edge_x, top_edge_y = zip(*sorted(zip(top_edge_x, top_edge_y)))
        # Переделать нахождение дна и верхушки полигона при большом кол-ве ошибок
        if point_x.index(top_edge_x[0]) == 0:
            #print('Первая точка в левом верхнем')
            left_top_ind = 0
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            right_down_ind = point_x.index(bottom_edge_x[1])

            bottom_x, bottom_y = point_x[right_down_ind : left_down_ind + 1], point_y[right_down_ind : left_down_ind + 1]
        elif top_edge_x[0] == point_x[-1]: # Переделать
            #print('Первая точка сверху после угла')
            left_top_ind = -1
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = [point_x[-1]], [point_y[-1]]
            top_x = top_x + point_x[0 : right_top_ind + 1]
            top_y = top_y + point_y[0 : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            right_down_ind = point_x.index(bottom_edge_x[1])

            bottom_x, bottom_y = point_x[right_down_ind : left_down_ind + 1], point_y[right_down_ind : left_down_ind + 1]
            
        elif bottom_edge_x[0] == point_x[0]:
            #print('Первая точка в левом нижнем')
            left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = 0
            right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : ], point_y[right_down_ind : ]
            bottom_x, bottom_y = [point_x[0]] + bottom_x, [point_y[0]] + bottom_y
        
        elif bottom_edge_x[1] == point_x[0]:
            #print('Первая точка в правом нижнем')
            left_top_ind = point_x.index(top_edge_x[0])
            right_top_ind = point_x.index(top_edge_x[1])
            top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

            left_down_ind = point_x.index(bottom_edge_x[0])
            right_down_ind = point_x.index(bottom_edge_x[1])
            bottom_x, bottom_y = point_x[right_down_ind : left_down_ind + 1], point_y[right_down_ind :  left_down_ind + 1]
            
        else:
            #print('Первая точка в другом положении')
            is_good_rect = False

        if len(bottom_x) == 0 or len(top_x) == 0:
            top_x, top_y, bottom_x, bottom_y = bottom_x, bottom_y, top_x, top_y
            if point_x.index(top_edge_x[0]) == 0:
                #print('Первая точка в левом верхнем')
                left_top_ind = 0
                right_top_ind = point_x.index(top_edge_x[1])
                top_x, top_y = point_x[left_top_ind : right_top_ind + 1], point_y[left_top_ind : right_top_ind + 1]

                left_down_ind = point_x.index(bottom_edge_x[0])
                right_down_ind = point_x.index(bottom_edge_x[1])

                bottom_x, bottom_y = point_x[right_down_ind : left_down_ind + 1], point_y[right_down_ind : left_down_ind + 1]
                if len(bottom_x) == 0 or len(top_x) == 0:
                    #print('Удаление полигона')
                    is_good_rect = False
            else:
                is_good_rect = False
    if is_good_rect:    # Без сортировки хорошо берет плохие печати, с сортировкой справляется с разметкой против часовой или вперемешку
        ### bottom Euclidian distance sorting
        xy_pair_bottom = []
        for i in range(len(bottom_x)):
            xy_pair_bottom.append([bottom_x[i], bottom_y[i]])

        Ap = np.array(xy_pair_bottom[-1]) # "lowest point"
        B = np.array(xy_pair_bottom) # sample array of points
        dist = np.linalg.norm(B - Ap, ord=2, axis=1) # calculate Euclidean distance (2-norm of difference vectors)
        sorted_B = B[np.argsort(dist)]

        bottom_x = []
        bottom_y = []
        for i in range(len(sorted_B)):
            bottom_x.append(sorted_B[i][0])
            bottom_y.append(sorted_B[i][1])

        ### top Euclidian distance sorting
        xy_pair_bottom = []
        for i in range(len(top_x)):
            xy_pair_bottom.append([top_x[i], top_y[i]])

        Ap = np.array(xy_pair_bottom[0]) # "lowest point"
        B = np.array(xy_pair_bottom) # sample array of points
        dist = np.linalg.norm(B - Ap, ord=2, axis=1) # calculate Euclidean distance (2-norm of difference vectors)
        sorted_B = B[np.argsort(dist)]

        top_x = []
        top_y = []
        for i in range(len(sorted_B)):
            top_x.append(sorted_B[i][0])
            top_y.append(sorted_B[i][1])

        bbox_height_left = ((top_x[0] - bottom_x[0])**2+(top_y[0] - bottom_y[0])**2)**0.5
        bbox_height_right = ((top_x[-1] - bottom_x[-1])**2+(top_y[-1] - bottom_y[-1])**2)**0.5
        mid_arithmetic_h = round((bbox_height_left + bbox_height_right)/2, 1)
        if mid_arithmetic_h > 150:
            ##print('Большой полигон, H = ', mid_arithmetic_h)
            #is_good_rect = False
            bottom_x, bottom_y, top_x, top_y = [], [], [], []
    else:
        bottom_x, bottom_y, top_x, top_y = [], [], [], []
        mid_arithmetic_h = 0
    
    return is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h

def draw_black_rect(im, point_pairs):
    cnt = []
    for i in range(len(point_pairs) - 2, -2, -2):
        start_point = (point_pairs[i], point_pairs[i + 1])
        end_point = (point_pairs[i - 2], point_pairs[i + 1 - 2])
        cnt.append(start_point) 
        cnt.append(end_point) 
    black = cv2.fillPoly(im, np.array([cnt]), (0,0,0), lineType=cv2.LINE_AA)
    return black

def find_4_dots(point_x, point_y):
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

    curve_out_of_repeats = []
    for j in range(len(curve)): # add delta for the reason of not mess in equal angles
        delta = 10**(-10)
        curve_out_of_repeats.append(curve[j] + delta*j)

    edge_angles = []
    edge_x = []
    edge_y = []
    for i in range(len(curve_out_of_repeats)):
        if curve_out_of_repeats[i] < 0.5: # cos(pi/3)
            edge_angles.append(curve_out_of_repeats[i])
            edge_x.append(point_x[i])
            edge_y.append(point_y[i])
    
    if len(edge_angles) != 4:
        #print('Убираем edge_angles = ', edge_angles)
        #print('Запасной вариант! cos(pi/4)')
        edge_angles = []
        edge_x = []
        edge_y = []
        for i in range(len(curve_out_of_repeats)):
            if curve_out_of_repeats[i] < 0.7:
                edge_angles.append(curve_out_of_repeats[i])
                edge_x.append(point_x[i])
                edge_y.append(point_y[i])

    if len(edge_angles) != 4:
        #print('Убираем edge_angles = ', edge_angles)
        #print('Запасной вариант! cos(pi/6)')
        edge_angles = []
        edge_x = []
        edge_y = []
        for i in range(len(curve_out_of_repeats)):
            if curve_out_of_repeats[i] < 0.86:
                edge_angles.append(curve_out_of_repeats[i])
                edge_x.append(point_x[i])
                edge_y.append(point_y[i])

    if len(edge_angles) != 4:
        #print('Убираем edge_angles = ', edge_angles)
        #print('Запасной вариант! Нарушает порядок')
        curve_out_of_edges = curve_out_of_repeats.copy()
        edge_angles = []
        edge_x = []
        edge_y = []
        for i in range(4):
            edge_angles.append(min(curve_out_of_edges))
            curve_out_of_edges.remove(min(curve_out_of_edges))
        #print('edge_angles = ', edge_angles)
        for i in edge_angles:
            index = curve_out_of_repeats.index(i)

            edge_x.append(point_x[index])
            edge_y.append(point_y[index])

        edge_x, edge_y = zip(*sorted(zip(edge_x, edge_y)))
    return edge_x, edge_y

def top_down_points(im, bottom_x, bottom_y, top_x, top_y):
    # Radius of circle
    radius = 1
    thickness = 1
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

    return im

with open('input_total - Copy.txt', encoding = 'utf8') as fp:
    lines = fp.readlines()
    img_counter = -1
    #data_annotation = open('./ds_images.csv', 'w')

    for line in lines:
        js_line = json.loads(line)
        file_name = js_line['file'].split('/')
        file_name = file_name[-1].split('?')
        file_path = './HuaweiToloka_total/' + file_name[0]
        im = cv2.imread(file_path)
        if im is None:
            continue
        print(file_name[0])
        height, width = im.shape[:2]

        second = js_line['result']
        #try:          
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
                print(file_name[0])
                cv2.imwrite('./two_line_frames/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # функция определяющая где верх и низ полигона и возвращающая их точки
                is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h = find_bbox_coord(point_x, point_y)
            if is_good_rect and mid_arithmetic_h <= 150:
                # kx_plus_b, возвращающая точки под полигоном с шагом 1 пикс по оХ
                poligon_dots, x_plus_delta, y_plus_delta = kx_plus_b(bottom_x, bottom_y) 
                poligon_dots_top, x_plus_delta_top, y_plus_delta_top = kx_plus_b(top_x, top_y)
                top_down_points(im, x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top)
            elif is_good_rect and mid_arithmetic_h > 150:
                poligon_dots = 0

            else:
                black = draw_black_rect(im, point_pairs)

                im = black
                poligon_dots = 0
                mid_arithmetic_h = 'empty'
                x_plus_delta, y_plus_delta = [], []
                            
            All_image_dots = All_image_dots + poligon_dots
            
        if All_image_dots > 0:
            img_counter = img_counter + 1
            #cv2.imwrite('./two_line_frames/image_{}.jpg'.format(img_counter), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite('./two_line_frames/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
        #except:
        #    cv2.imwrite('./error_img/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#print('end_time = ', time.time() - start_time)
