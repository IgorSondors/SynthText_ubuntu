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
        num_dots_uder_ch = 30
        
        my_ch_label.write(k[:-2] + ',' + str(img_h) + ',' + str(img_w) + ',' + str(chars_quantity * num_dots_uder_ch))

        all_symbols = ''
        for j in range(len(txt)):
            all_symbols = all_symbols + txt[j]
            all_symbols = re.sub('[(\n) ]', '', all_symbols)
        print('all_symbols ', all_symbols)

        for i in range(chars_quantity):

            x_down_left = charBB[0][3][i]
            y_down_left = charBB[1][3][i]
            x_top_left = charBB[0][0][i]
            y_top_left = charBB[1][0][i]
            x_top_right = charBB[0][1][i]
            y_top_right = charBB[1][1][i]
            x_down_right = charBB[0][2][i]
            y_down_right = charBB[1][2][i]

            print(x_down_left, y_down_left, x_top_left,
            y_top_left,
            x_top_right,
            y_top_right,
            x_down_right,
            y_down_right)

            value_of_symbol = all_symbols[i]

            w_of_next_ch = ((x_down_right - x_down_left)**2+(y_down_right - y_down_left)**2)**0.5
            h_of_next_ch = ((x_top_right - x_down_right)**2+(y_top_right - y_down_right)**2)**0.5

            my_ch_label.write(',' + str(x_down_left) + ',' + str(y_down_left) + ',' + str(w_of_next_ch) + ',' + str(h_of_next_ch) + ',' + value_of_symbol)

            x_plus_delta, y_plus_delta = kx_plus_b(x_down_left, y_down_left, x_down_right, y_down_right, num_dots_uder_ch)

            #plt.plot(x_plus_delta, y_plus_delta, 'r.')

            for j in range(num_dots_uder_ch):

                my_ch_label.write(',' + str(x_plus_delta[j]) + ',' + str(y_plus_delta[j]) + ',' + str(w_of_next_ch) + ',' + str(h_of_next_ch) + ',' + value_of_symbol)
        
        my_ch_label.write('\n')

        #plt.gca().set_xlim([0,W-1])
        #plt.gca().set_ylim([H-1,0])
        #plt.show(block=False)
        #plt.savefig('results/400/{}.png'.format(k), dpi= 'figure')
        """if 'q' in input("next? ('q' to exit) : "):
                break"""


    db.close()

def kx_plus_b(x_down_left, y_down_left, x_down_right, y_down_right, num_dots_uder_ch):
    k = (y_down_left - y_down_right)/(x_down_left - x_down_right)
    b = y_down_left - k * x_down_left
    delta_x = (x_down_right - x_down_left)/num_dots_uder_ch
    x_plus_delta = []
    y_plus_delta = []

    next_x = x_down_left
    next_y = y_down_left
    for i in range(num_dots_uder_ch):
        next_x = next_x + delta_x
        next_y = k * next_x + b
        x_plus_delta.append(next_x)
        y_plus_delta.append(next_y)

    return x_plus_delta, y_plus_delta



if __name__=='__main__':
    main('/home/sondors/SynthText_ubuntu/results/dset_500.h5')