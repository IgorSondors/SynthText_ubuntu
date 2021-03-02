"""
check csv writing 

"""
import numpy as np
import h5py 
import cv2
import json
import time

start_time = time.time()

def top_bottom_dots(point_x, point_y, edge_x, edge_y):
    if point_x[0] in edge_x:
        is_good_rect = True

        index_0 = point_x.index(edge_x[0])
        index_1 = point_x.index(edge_x[1])
        index_2 = point_x.index(edge_x[2])
        index_3 = point_x.index(edge_x[3])

        line_0_x = point_x[index_0 : index_1 + 1]
        line_0_y = point_y[index_0 : index_1 + 1]

        line_1_x = point_x[index_1 : index_2 + 1]
        line_1_y = point_y[index_1 : index_2 + 1]

        line_2_x = point_x[index_2 : index_3 + 1]
        line_2_y = point_y[index_2 : index_3 + 1]

        line_3_x = point_x[index_3 : ] + [point_x[0]]
        line_3_y = point_y[index_3 : ] + [point_y[0]]

        lines = [[line_0_x, line_0_y], [line_1_x, line_1_y], [line_2_x, line_2_y], [line_3_x, line_3_y]]
        lines_len = []
        for j in lines:
            line_len = 0
            for i in range(len(j[0]) - 1):
                x1 = j[0][i]
                y1 = j[1][i]
                x2 = j[0][i + 1]
                y2 = j[1][i + 1]
                
                next_len = ((x2 - x1)**2+(y2 - y1)**2)**0.5
                line_len = line_len + next_len
            lines_len.append(line_len)
        lines_len, lines = zip(*sorted(zip(lines_len, lines),reverse = True))
        lines_top_bottom = lines[0 : 3]
        
        first_mid_y = 0
        for i in lines_top_bottom[0][1]:
            first_mid_y = first_mid_y + i/len(lines_top_bottom[0][1])
        second_mid_y = 0
        for i in lines_top_bottom[1][1]:
            second_mid_y = second_mid_y + i/len(lines_top_bottom[1][1])

        if first_mid_y > second_mid_y:
            bottom_x, bottom_y = lines_top_bottom[0][0], lines_top_bottom[0][1]
            top_x, top_y = lines_top_bottom[1][0], lines_top_bottom[1][1]
        else:
            bottom_x, bottom_y = lines_top_bottom[1][0], lines_top_bottom[1][1]
            top_x, top_y = lines_top_bottom[0][0], lines_top_bottom[0][1]
    else:
        is_good_rect = False
        bottom_x, bottom_y, top_x, top_y = [], [], [], []
    return bottom_x, bottom_y, top_x, top_y, is_good_rect

def find_bbox_coord(point_x, point_y):
    is_good_rect = True
    bottom_x, bottom_y = [], []
    top_x, top_y = [], []
    if len(point_x) < 4:
        is_good_rect = False
    if len(point_x) == 4:
        out_of_repeats_x = []
        out_of_repeats_y = []
        delta = 10**(-6)
        for j in range(len(point_x)): # add delta for the reason of not mess in equal angles
            out_of_repeats_x.append(point_x[j] + delta*j)
            out_of_repeats_y.append(point_y[j] + delta*j)
        point_x, point_y = out_of_repeats_x, out_of_repeats_y
        
        quadrate_width = ((point_x[1] - point_x[0])**2+(point_y[1] - point_y[0])**2)**0.5
        quadrate_height = ((point_x[1] - point_x[2])**2+(point_y[1] - point_y[2])**2)**0.5
        aspect_ratio = quadrate_width / quadrate_height
        if aspect_ratio > 0.7 and aspect_ratio < 1.3:
            is_good_rect = False   
            print('Квадрат. Закрашиваем')
        elif quadrate_width * quadrate_height < 100:
            is_good_rect = False   
            print('Квадрат. Закрашиваем')
        else:
            print('Прямоугольник')
            edge_x, edge_y = point_x, point_y
            bottom_x, bottom_y, top_x, top_y, is_good_rect = top_bottom_dots(point_x, point_y, edge_x, edge_y)
                    
    elif len(point_x) > 4:
        print('Многоугольник')
        out_of_repeats_x = []
        out_of_repeats_y = []
        delta = 10**(-4)
        for j in range(len(point_x)): # add delta for the reason of not mess in equal angles
            out_of_repeats_x.append(point_x[j] + delta*j)
            out_of_repeats_y.append(point_y[j] + delta*j)
        point_x, point_y = out_of_repeats_x, out_of_repeats_y
        
        edge_x, edge_y = find_4_dots(point_x, point_y)

        bottom_x, bottom_y, top_x, top_y, is_good_rect = top_bottom_dots(point_x, point_y, edge_x, edge_y)
          
    if is_good_rect:
    
        bottom_x, bottom_y = Euclidian_distance_sorting(bottom_x, bottom_y, bottom = True)
        top_x, top_y = Euclidian_distance_sorting(top_x, top_y, bottom = False)

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

def Euclidian_distance_sorting(bottom_x, bottom_y, bottom):
    xy_pair_bottom = []
    for i in range(len(bottom_x)):
        xy_pair_bottom.append([bottom_x[i], bottom_y[i]])

    if bottom:
        Ap = np.array(xy_pair_bottom[-1]) # "lowest point"
    else:
        Ap = np.array(xy_pair_bottom[0])
    B = np.array(xy_pair_bottom) # sample array of points
    dist = np.linalg.norm(B - Ap, ord=2, axis=1) # calculate Euclidean distance (2-norm of difference vectors)
    sorted_B = B[np.argsort(dist)]

    bottom_x = []
    bottom_y = []
    for i in range(len(sorted_B)):
        bottom_x.append(sorted_B[i][0])
        bottom_y.append(sorted_B[i][1])
    return bottom_x, bottom_y

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
                x_plus_delta.append(int(next_x))
                y_plus_delta.append(round(next_y,1))
            poligon_dots = poligon_dots + dots_between_edges
        except:
            print('Расстояние между точками 0 пикселей! Пропуск точки')

    x_plus_delta.append(int(bottom_x[-1]))
    y_plus_delta.append(round(bottom_y[-1],1))
    poligon_dots = poligon_dots + 1
    return poligon_dots, x_plus_delta, y_plus_delta

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

    curve_out_of_edges = curve_out_of_repeats.copy()
    edge_angles = []
    edge_indexes = []
    edge_x = []
    edge_y = []
    for i in range(4):
        edge_angles.append(min(curve_out_of_edges))
        curve_out_of_edges.remove(min(curve_out_of_edges))

    for i in edge_angles:
        index = curve_out_of_repeats.index(i)
        edge_indexes.append(index)
    edge_indexes = sorted(edge_indexes)

    for i in edge_indexes:
        edge_x.append(point_x[i])
        edge_y.append(point_y[i])
        
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
        file_path = './error_img/' + file_name[0]
        im = cv2.imread(file_path)
        if im is None:
            continue
        print(file_name[0])
        height, width = im.shape[:2]

        second = js_line['result']
        try:          
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
                    #cv2.imwrite('./two_line_frames/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    # функция определяющая где верх и низ полигона и возвращающая их точки
                    is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h = find_bbox_coord(point_x, point_y)
                if is_good_rect and mid_arithmetic_h <= 150:
                    # kx_plus_b, возвращающая точки под полигоном с шагом 1 пикс по оХ
                    poligon_dots, x_plus_delta, y_plus_delta = kx_plus_b(bottom_x, bottom_y) 
                    poligon_dots_top, x_plus_delta_top, y_plus_delta_top = kx_plus_b(top_x, top_y)
                    top_down_points(im, x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top)
                elif is_good_rect and mid_arithmetic_h > 150:
                    poligon_dots = 0
                    x_plus_delta, y_plus_delta = [], []

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
        except:
            cv2.imwrite('./error_img/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

print('end_time = ', time.time() - start_time)
