import os
import numpy as np
import math
import h5py 
import itertools
from PIL import Image, ImageOps
import re
import cv2
import random
from sklearn.linear_model import LinearRegression

from common import *

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))

    my_ch_label = open('results/my.csv', 'a')
    
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        print("image name        : ", colorize(Color.RED, k, bold=True))
        print("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print("  ** text         : ", colorize(Color.GREEN, txt))

        img = Image.fromarray(rgb, 'RGB')
        #gray_image = ImageOps.grayscale(img)
        img.save('results/my_images/'+k[:-2])
        #gray_image.save('results/my_label/images/'+k[:-2])

        name_jpg = k
        chars_quantity = charBB.shape[-1]

        pixel_step_x = 1
        All_x_down_right_word, All_y_down_left_word, All_w_char_list, All_h_char_list, All_char_list = [], [], [], [], []
        
        all_symbols = ''
        new_txt = []
        ochered = -1
        
        for j in range(len(txt)):
            all_symbols = all_symbols + txt[j]
            all_symbols = re.sub('[(\n) ]', '', all_symbols)
            txt_for_split = re.sub('(\n)', ' ', txt[j])
            splitted = txt_for_split.split()
            new_txt = new_txt + splitted
        print('new_txt = ', new_txt)
        #print('charBB = ', charBB)
        for j in range(len(new_txt)): #кол-во слов
            ochered = ochered + 1
            print('номер = ', j,  ' Слово = ', new_txt[j], ', кол-во букв = ', len(str(new_txt[j])),)
            i = 0
            num_ch_per_w = len(str(new_txt[j])) #кол-во букв для данного слова
            x_down_left_word = []
            y_down_left_word = []
            x_down_right_word = []
            y_down_right_word = []
            word = ''
            w_of_next_ch_word = []
            h_of_next_ch_word = []

            x_top_left_word, x_top_right_word, y_top_left_word, y_top_right_word = [], [], [], []
            my_ch_label.write(k[:-2]+'_{}'.format(j))

            while i < num_ch_per_w:
                i = i + 1

                print('Буква = ', all_symbols[ochered])

                x_down_left = charBB[0][3][ochered]
                y_down_left = charBB[1][3][ochered]
                x_top_left = charBB[0][0][ochered]
                y_top_left = charBB[1][0][ochered]
                x_top_right = charBB[0][1][ochered]
                y_top_right = charBB[1][1][ochered]
                x_down_right = charBB[0][2][ochered]
                y_down_right = charBB[1][2][ochered]               

                value_of_symbol = all_symbols[ochered]
                word = word + value_of_symbol

                w_of_next_ch = ((x_down_right - x_down_left)**2+(y_down_right - y_down_left)**2)**0.5
                h_of_next_ch = ((x_top_right - x_down_right)**2+(y_top_right - y_down_right)**2)**0.5

                w_of_next_ch_word.append(int(w_of_next_ch))
                h_of_next_ch_word.append(int(h_of_next_ch))

                x_down_left_word.append(int(x_down_left))
                y_down_left_word.append(int(y_down_left))
                x_down_right_word.append(int(x_down_right))
                y_down_right_word.append(int(y_down_right))

                x_top_left_word.append(int(x_top_left))
                y_top_left_word.append(int(y_top_left))
                x_top_right_word.append(int(x_top_right))
                y_top_right_word.append(int(y_top_right))

                if i <= num_ch_per_w - 1:
                    ochered = ochered + 1
            
            tg_alpha, b, model = lin_reg(x_down_left_word, y_down_left_word, x_down_right_word, y_down_right_word)
            try:            
                x_down_right_word, y_down_left_word = dot_word_crop(tg_alpha, b, k, rgb, word, w_of_next_ch_word, h_of_next_ch_word, my_ch_label, j,   
                                                                    x_down_left_word, x_down_right_word, x_top_left_word, x_top_right_word, 
                                                                    y_top_left_word, y_top_right_word)
            except:
                print('Bad word!!!', '\n', word)
            my_ch_label.write('\n')
    db.close()
    
def dot_word_crop(tg_alpha, b, k, rgb, word, w_of_next_ch_word, h_of_next_ch_word, my_ch_label, j,   
                    x_down_left_word, x_down_right_word, x_top_left_word, x_top_right_word, 
                    y_top_left_word, y_top_right_word):

    y_down_right_word, y_down_left_word = [], []
    for i in x_down_left_word:
        next_y = tg_alpha * i + b
        y_down_left_word.append(next_y)

    for i in x_down_right_word:
        next_y = tg_alpha * i + b
        y_down_right_word.append(next_y)

    x_top_left, x_top_right = int(x_top_left_word[0]), int(x_top_right_word[-1])
    y_top_left, y_top_right = int(y_top_left_word[0]), int(y_top_right_word[-1])
    x_down_left, x_down_right = int(x_down_left_word[0]), int(x_down_right_word[-1])
    y_down_left, y_down_right = int(y_down_left_word[0]), int(y_down_right_word[-1])
    
    img = rgb
    #img = cv2.circle(img, (x_top_left, y_top_left), radius=0, color=(0, 0, 255), thickness=2)
    #img = cv2.circle(img, (x_top_right, y_top_right), radius=0, color=(0, 0, 255), thickness=2)
    #img = cv2.circle(img, (x_down_right, y_down_right), radius=0, color=(0, 0, 255), thickness=2)
    #img = cv2.circle(img, (x_down_left, y_down_left), radius=0, color=(0, 0, 255), thickness=2)

    h_of_word = min(h_of_next_ch_word)
    w_of_word = (((x_down_right - x_down_left)**2+(y_down_right - y_down_left)**2)**0.5)

    y_top_left, y_top_right, y_down_left, y_down_right, x_top_left, x_top_right, x_down_left, x_down_right, delta_b1, delta_b2 = random_transform_coord(
                                                                                                        tg_alpha, word, h_of_word, 
                                                                                                        y_top_left, y_top_right, y_down_left, y_down_right, 
                                                                                                        x_top_left, x_top_right, x_down_left, x_down_right)

    h_of_word_new = h_of_word + delta_b1 + delta_b2

    pts1 = np.float32([[x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left]])
    pts2 = np.float32([[0,0],[w_of_word,0],[w_of_word,h_of_word_new],[0,h_of_word_new]])
    #pts2 = np.float32([[0,0],[w_of_word,0],[w_of_word,h_of_word],[0,h_of_word]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(int(w_of_word), int(h_of_word_new)))

    w_of_word_new = w_of_word * 2
    h_of_word_new = 32

    print('word = ',word)
    Rx = w_of_word_new / w_of_word
    Ry = h_of_word_new / h_of_word_new
    
    print('Rx =', Rx)
    #img = cv2.circle(img, (x_top_left, y_top_left), radius=0, color=(0, 255, 0), thickness=2)
    #img = cv2.circle(img, (x_top_right, y_top_right), radius=0, color=(0, 255, 0), thickness=2)
    #img = cv2.circle(img, (x_down_right, y_down_right), radius=0, color=(0, 255, 0), thickness=2)
    #img = cv2.circle(img, (x_down_left, y_down_left), radius=0, color=(0, 255, 0), thickness=2)
    
    cv2.imwrite('results/my_words/{}_{}'.format(k[:-2], j), dst)

    img_word = cv2.imread('results/my_words/{}_{}'.format(k[:-2], j), 3)

    resized_img_word = cv2.resize(img_word, (int(w_of_word_new), int(h_of_word_new)))
    resized_img_word = cv2.hconcat((resized_img_word, np.zeros((np.shape(resized_img_word)[0], 32, 3), dtype=np.uint8) ))
    resized_img_word = cv2.hconcat((np.zeros((np.shape(resized_img_word)[0], 32, 3), dtype=np.uint8), resized_img_word ))

    #cv2.imwrite('results/my_words/{}_{}'.format(k[:-2], j), img_word)

    coordinates, coordinates_old = apply_center_coord_transform(x_down_left_word, x_down_right_word, x_top_left_word, x_top_right_word, 
                                                                y_down_right_word, y_down_left_word, y_top_left_word, y_top_right_word, 
                                                                w_of_next_ch_word, h_of_next_ch_word, M)
    #img_word = cv2.hconcat((img_word, np.zeros((np.shape(img_word)[0], 32, 3), dtype=np.uint8) ))
    #img_word = cv2.hconcat((np.zeros((np.shape(img_word)[0], 32, 3), dtype=np.uint8), img_word ))

    #print('img_word.shape = ', img_word.shape)
    my_ch_label.write(',' + str(img_word.shape[0]) + ',' + str(img_word.shape[1]))

    ch_code = []
    for i in word:
        code = ''
        if i == word[0]:
            code = code + str()
        ch_code.append(code)

    for i in range(len(coordinates)):
        # Horizontal border
        my_ch_label.write(',' + str(coordinates[i][0]* Rx + 32) + ',' + str(word[i]) + ',' + 'code_{}'.format(ch_code[0]))
            
        #img_word = cv2.circle(img_word, (coordinates[i][0] + 32, coordinates[i][1]), radius=0, color=(0, 0, 255), thickness=2)

        resized_img_word = cv2.circle(resized_img_word, (int(coordinates[i][0] * Rx + 32), 16), radius=0, color=(0, 0, 255), thickness=2)
        
        #cv2.imwrite('results/my_words/{}_{}'.format(k[:-2], j), img_word)
        #cv2.imwrite('results/resized/{}_{}'.format(k[:-2], j), resized_img_word)
        cv2.imwrite('results/resized/{}_{}'.format(k[:-2], word), resized_img_word)

    """for i in coordinates_old: # Very slow
        # Horizontal border
        
        img = cv2.circle(img, (i[0], i[1]), radius=0, color=(0, 0, 255), thickness=2)
        
        cv2.imwrite("results/dots/{}_my.png".format(k[:-2]),img)"""
    
    return x_down_right_word, y_down_left_word

def apply_center_coord_transform(x_down_left_word, x_down_right_word, x_top_left_word, x_top_right_word, 
                                y_down_right_word, y_down_left_word, y_top_left_word, y_top_right_word, 
                                w_of_next_ch_word, h_of_next_ch_word, M):
    coordinates = []
    coordinates_old = []
    for i in range(len(x_down_left_word)):

        middle_x = (x_down_left_word[i] + x_down_right_word[i] + x_top_left_word[i] + x_top_right_word[i])/4
        middle_y  = (y_down_left_word[i] + y_down_right_word[i] + y_top_left_word[i] + y_top_right_word[i])/4
        coordinates_old.append((int(middle_x), int(middle_y)))
        middle_x, middle_y = perspective_transform_coordinates([middle_x, middle_y], M)
        coordinates.append((int(middle_x), int(middle_y)))
    return coordinates, coordinates_old

def random_transform_coord(tg_alpha, word, h_of_word, y_top_left, y_top_right, y_down_left, y_down_right, x_top_left, x_top_right, x_down_left, x_down_right):    

    print('Before')
    print(word)
    print('[x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left] = ', 
    [x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left])
    
    delta_b = 11
    delta_b1 = (random.randint(3000, 8000)) / 10000 * h_of_word
    delta_b2 = (random.randint(3000, 8000)) / 10000 * h_of_word
    

    if tg_alpha != 0:
        print('tg_alpha != 0:',word, tg_alpha)
        if tg_alpha < 0:
            tg_alpha = abs(tg_alpha)
            alpha = math.atan(tg_alpha)

            y_down_left, y_down_right = int(y_down_left + delta_b1 * math.cos(alpha)), int(y_down_right + delta_b1 * math.cos(alpha))
            x_down_left = int(x_down_left +  delta_b1 * math.sin(alpha))
            x_down_right = int(x_down_right +  delta_b1 * math.sin(alpha))

            y_top_left, y_top_right = int(y_top_left - delta_b2 * math.cos(alpha)), int(y_top_right - delta_b2 * math.cos(alpha))
            x_top_left = int(x_top_left -  delta_b2 * math.sin(alpha))
            x_top_right = int(x_top_right -  delta_b2 * math.sin(alpha))

        elif tg_alpha > 0:
            tg_alpha = abs(tg_alpha)
            alpha = math.atan(tg_alpha)

            y_down_left, y_down_right = int(y_down_left + delta_b1 * math.cos(alpha)), int(y_down_right + delta_b1 * math.cos(alpha))
            x_down_left = int(x_down_left -  delta_b1 * math.sin(alpha))
            x_down_right = int(x_down_right -  delta_b1 * math.sin(alpha))

            y_top_left, y_top_right = int(y_top_left - delta_b2 * math.cos(alpha)), int(y_top_right - delta_b2 * math.cos(alpha))
            x_top_left = int(x_top_left +  delta_b2 * math.sin(alpha))
            x_top_right = int(x_top_right +  delta_b2 * math.sin(alpha))
    else:
        y_top_left, y_top_right, y_down_left, y_down_right = y_top_left - delta_b2, y_top_right - delta_b2, y_down_left + delta_b1, y_down_right + delta_b1
    print('After')
    print(word)
    print('[x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left] = ', 
    [x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left])

    return y_top_left, y_top_right, y_down_left, y_down_right, x_top_left, x_top_right, x_down_left, x_down_right, delta_b1, delta_b2


def lin_reg(x_down_left_word, y_down_left_word, x_down_right_word, y_down_right_word):
    x = np.array(x_down_left_word + x_down_right_word).reshape((-1, 1))
    y = np.array(y_down_left_word + y_down_right_word)

    model = LinearRegression().fit(x, y)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    k = model.coef_[0]
    b = model.intercept_
    return k, b, model

def perspective_transform_coordinates(coordinates, m_matrix):
    """
    Function Description: Apply perspective transformation to coordinates.
    Parameters:
    Return Value:
    Exception Description:
    Change History:
    2020-07-30 12:00 function created.
    """
    """center = (0, 0)
    newrow = [0, 0, 1]
    r_matrix = cv2.getRotationMatrix2D(center, 0.0, 1.0)
    r_matrix = np.vstack([r_matrix, newrow])
    m_matrix = np.dot(M, r_matrix)"""

    # Perform the actual coordinates processing
    coordinates.append(1)
    new_coordinates = np.dot(m_matrix, coordinates)
    new_coordinates[0] = round(new_coordinates[0] / new_coordinates[2], 1)
    new_coordinates[1] = round(new_coordinates[1] / new_coordinates[2], 1)
    return new_coordinates[:2]

if __name__=='__main__':
    main('/home/sondors/SynthText_ubuntu/results/1.h5')