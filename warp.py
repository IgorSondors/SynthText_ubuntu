"""
warping strides from real dataset 
"""
import numpy as np
import cv2
import math
import scipy
import scipy.interpolate
import json
import random
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
        mid_arithmetic_h = int((bbox_height_left + bbox_height_right)/2)
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

def how_many_dots(bottom_x, bottom_y, top_x, top_y):

    bottom_length = 0
    for i in range(len(bottom_x) - 1):
        bottom_length = bottom_length + (((bottom_x[i+1] - bottom_x[i])**2 + (bottom_y[i+1] - bottom_y[i])**2)**0.5)
    top_length = 0
    for i in range(len(top_x) - 1):
        top_length = top_length + (((top_x[i+1] - top_x[i])**2 + (top_y[i+1] - top_y[i])**2)**0.5)

    how_many_dots = max(len(bottom_x), len(top_x))
    return  how_many_dots, bottom_length, top_length

def kx_plus_b(bottom_x, bottom_y, top_bottom_ratio):
    x_plus_delta = []
    y_plus_delta = []
    pixel_step = 1 * top_bottom_ratio
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

    return x_plus_delta, y_plus_delta

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

def warp_image(img, src, dst, width, height):
    grid_x, grid_y = np.mgrid[0:height, 0:width]
    #grid_z = griddata(dst, src, (grid_x, grid_y), method='cubic')
    grid_z = scipy.interpolate.griddata(dst, src, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(height, width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    #warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)
    return warped

def crop_areas(opencv_img, poligons, poligon_height):
    print('crop_areas')
    print('x', poligons[0][0],poligons[0][-1])
    print('y', poligons[1][0],poligons[1][-1])
    boarder = 192
    src = []
    dst = []
    dist = 0
    stripe_height = 32
    dist = 0.0
    dist_cur = 0
    hor_scale = float(stripe_height/poligon_height)
    # xu - x upper, xl - x lower
    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

        src.append([poligons[3][i], poligons[2][i]])
        dst.append([0, dist_cur*hor_scale])

    dist = 0.0
    dist_cur = 0
    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

    
        src.append([poligons[1][i], poligons[0][i]])
        dst.append([stripe_height, dist_cur*hor_scale])

    src_arr = np.array(src)
    dst_arr = np.array(dst)
    
    warped = warp_image(opencv_img, src_arr, dst_arr, int(dst[len(poligons[0]) - 1][1]), 32)
    warped = cv2.copyMakeBorder( warped, top=0, bottom=0, left=boarder, right=boarder, borderType=cv2.BORDER_CONSTANT )
    
    return warped

with open('input.txt', encoding = 'utf8') as fp:
    lines = fp.readlines()
    poligon_counter = -1

    for line in lines:
        js_line = json.loads(line)
        file_name = js_line['file'].split('/')
        file_name = file_name[-1].split('?')
        file_path = './images/' + file_name[0]
        im = cv2.imread(file_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im is None:
            continue
        height, width = im.shape[:2]

        second = js_line['result']        
        # second - один словарик для одного из изображений из множества {"file":"name.jpg","result":[...]}, 
        # включающий в себя [...] = [{"type":"polygon","data":[{"x":0.4,"y":0.4}, {}...], "readable":t/f}, ...]
        for s in second: # s - один из множества полигонов одной картинки
            is_good_rect = False
            if s['readable']:
                is_good_rect = True
            data = s['data']

            point_x = []
            point_y = []

            for d in data: # Множество точек одного полигона
                x = d['x']*width
                y = d['y']*height

                point_x.append(int(x))
                point_y.append(int(y))

            if is_good_rect:
                # функция определяющая где верх и низ полигона и возвращающая их точки
                is_good_rect, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h = find_bbox_coord(point_x, point_y)
            if is_good_rect and mid_arithmetic_h <= 150:
                
                print('bottom_x = ', bottom_x,'\n', 'bottom_y = ', bottom_y,'\n', 'top_x = ', top_x,'\n', 'top_y = ', top_y)
                number_of_dots, bottom_length, top_length = how_many_dots(bottom_x, bottom_y, top_x, top_y)

                if len(bottom_x) == len(top_x) == 4:
                    x_plus_delta, y_plus_delta = bottom_x, bottom_y
                    x_plus_delta_top, y_plus_delta_top = top_x, top_y
                else:
                    top_bottom_ratio = 1
                    x_plus_delta, y_plus_delta = kx_plus_b(bottom_x, bottom_y, top_bottom_ratio)
                    top_bottom_ratio = top_length / bottom_length
                    x_plus_delta_top, y_plus_delta_top = kx_plus_b(top_x, top_y, top_bottom_ratio)

                    if len(x_plus_delta) != len(x_plus_delta_top):
                        how_many_dots_to_remove = abs(len(x_plus_delta) - len(x_plus_delta_top))
                        rand_int = []
                        if len(x_plus_delta) > len(x_plus_delta_top):
                            for j in range(how_many_dots_to_remove):
                                rand_int.append(random.randint(1, len(x_plus_delta) - 2))
                            rand_int.sort(reverse = True)
                            for k in rand_int:
                                del x_plus_delta[k]
                                del y_plus_delta[k]
                        else:
                            for j in range(how_many_dots_to_remove):
                                rand_int.append(random.randint(1, len(x_plus_delta_top) - 2))
                            rand_int.sort(reverse = True)
                            for k in rand_int:
                                del x_plus_delta_top[k]
                                del y_plus_delta_top[k]   

                #if len(x_plus_delta) == len(x_plus_delta_top):
                poligons = [x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top]

                poligon_counter = poligon_counter + 1
                
                warped = crop_areas(im, poligons, mid_arithmetic_h)
                cv2.imwrite('./stripes/{}_{}.jpg'.format(file_name[0][:-4], poligon_counter), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])   
                #else:
                #    raise print('incorrect len', len(x_plus_delta), len(x_plus_delta_top))
print('end_time = ', time.time() - start_time)
