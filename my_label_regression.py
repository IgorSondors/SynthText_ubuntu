from __future__ import division
import os
import os.path as osp
import numpy as np

import h5py 
from common import *
import itertools
from PIL import Image, ImageDraw, ImageOps
import re
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from sklearn.linear_model import LinearRegression

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))

    my_ch_label = open('results/my_label/my_ch_label.csv', 'a')
    
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        print("image name        : ", colorize(Color.RED, k, bold=True))
        print("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print("  ** text         : ", colorize(Color.GREEN, txt))
        # print("  ** text         : ", colorize(Color.GREEN, txt.encode('utf-8')))
        #print('charBB', charBB)
        img = Image.fromarray(rgb, 'RGB')
        gray_image = ImageOps.grayscale(img)
        #img.save('results/my_label/images/'+k[:-2])
        gray_image.save('results/my_label/images/'+k[:-2])

        #plt.close(1)
        #plt.figure(1)
        #plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
        #H,W = rgb.shape[:2]

        img_w, img_h = img.size[0], img.size[1]
        name_jpg = k
        chars_quantity = charBB.shape[-1]

        pixel_step_x = 1
        All_x_under_word, All_y_under_word, All_w_char_list, All_h_char_list, All_char_list = [], [], [], [], []
        
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
            #for i in range(len(new_txt[j])):
            i = 0
            num_ch_per_w = len(str(new_txt[j])) #кол-во букв для данного слова
            x_down_left_word = []
            y_down_left_word = []
            x_down_right_word = []
            y_down_right_word = []
            word = ''
            w_of_next_ch_word = []
            h_of_next_ch_word = []
            while i < num_ch_per_w:
                i = i + 1
                #print('i = ', i)
                #ochered = ochered + 1
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

                #plt.plot(x_plus_delta, y_plus_delta, 'r.')

                if i <= num_ch_per_w - 1:
                    ochered = ochered + 1
            
            tg_alpha, b, model = lin_reg(x_down_left_word, y_down_left_word, x_down_right_word, y_down_right_word)
            
            
            x_under_word, y_under_word, w_char_list, h_char_list, char_list = dot_per_pix(tg_alpha, b, x_down_left_word, x_down_right_word, word, w_of_next_ch_word, h_of_next_ch_word, pixel_step_x)
            
            All_x_under_word = All_x_under_word + x_under_word
            All_y_under_word = All_y_under_word + y_under_word
            All_w_char_list = All_w_char_list + w_char_list
            All_h_char_list = All_h_char_list + h_char_list
            All_char_list = All_char_list + char_list

        number_of_dots = len(All_x_under_word)

        print(len(All_x_under_word), len(All_y_under_word), len(All_w_char_list), len(All_h_char_list), len(All_char_list))
        
        my_ch_label.write(k[:-2] + ',' + str(img_h) + ',' + str(img_w) + ',' + str(number_of_dots))

        for z in range(number_of_dots):

                my_ch_label.write(',' + str(All_x_under_word[z]) + ',' + str(All_y_under_word[z]) + ',' + (str(All_w_char_list[z])) + ',' + str(All_h_char_list[z]) + ',' + str(All_char_list[z]))
            
        my_ch_label.write('\n')

        #plt.gca().set_xlim([0,W-1])
        #plt.gca().set_ylim([H-1,0])
        #plt.show(block=False)
        #plt.savefig('results/400/{}.png'.format(k), dpi= 'figure')
        """if 'q' in input("next? ('q' to exit) : "):
                break"""

    db.close()
    
def dot_per_pix(tg_alpha, b, x_down_left_word, x_down_right_word, word, w_of_next_ch_word, h_of_next_ch_word, pixel_step_x):

    x_under_word, y_under_word = [], []
    char_list = []
    w_char_list, h_char_list = [], []

    start_x = x_down_left_word[0]
    end_x = x_down_right_word[-1]

    print('len(word) = ', len(word), 'len(x_down_left_word) = ', len(x_down_left_word))
    pixel_step_number = int((end_x - start_x)/pixel_step_x) + 1

    if pixel_step_number >= 0:
        for i in range(pixel_step_number):      
            next_x = start_x + i
            if next_x <= end_x + 1:
                x_under_word.append(next_x)
                next_y = tg_alpha * next_x + b
                y_under_word.append(next_y)
                
                for j in range(len(word)):
                    if  len(x_under_word) - 1 == len(char_list):
                        if next_x >= x_down_left_word[j] and next_x <= x_down_right_word[j]: # between left and right edges of bbox or edge of bbox
                            char_list.append(word[j])
                            w_char_list.append(w_of_next_ch_word[j])
                            h_char_list.append(h_of_next_ch_word[j])
                        elif next_x >= x_down_right_word[-1]: # + pixel after right edge of word
                            char_list.append(word[-1])
                            w_char_list.append(w_of_next_ch_word[-1])
                            h_char_list.append(h_of_next_ch_word[-1])
                        elif next_x >= x_down_left_word[j] and next_x >= x_down_right_word[j] and next_x <= x_down_left_word[j+1]: # between two bboxes
                            char_list.append(word[j+1])
                            w_char_list.append(w_of_next_ch_word[j+1])
                            h_char_list.append(h_of_next_ch_word[j+1])
                        

    """elif pixel_step_number < 0: # mirrored words
        for i in range((pixel_step_number)*(-1)):      
            next_x = start_x - i
            x_under_word.append(next_x)
            next_y = tg_alpha * next_x + b
            y_under_word.append(next_y)
            for j in range(len(word)):
                if next_x <= x_down_left_word[j] and next_x >= x_down_right_word[j]:
                    char_list.append(word[j])
                    w_char_list.append(w_of_next_ch_word[j])
                    h_char_list.append(h_of_next_ch_word[j])"""

    return x_under_word, y_under_word, w_char_list, h_char_list, char_list

def lin_reg(x_down_left_word, y_down_left_word, x_down_right_word, y_down_right_word):
    x = np.array(x_down_left_word + x_down_right_word).reshape((-1, 1))
    y = np.array(y_down_left_word + y_down_right_word)

    model = LinearRegression().fit(x, y)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    k = model.coef_[0]
    b = model.intercept_
    return k, b, model

if __name__=='__main__':
    main('/home/sondors/SynthText_ubuntu/results/dset_500.h5')