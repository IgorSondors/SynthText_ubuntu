from __future__ import division
import os
import os.path as osp
import numpy as np

import h5py 
from common import *
import itertools
from PIL import Image, ImageDraw
import re


def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))

    my_ch_label = open('results/my_label/my_ch_label.txt', 'a')
    
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
        img.save('results/my_label/images/'+k[:-2])

        name_jpg = k
        chars_quantity = charBB.shape[-1]

        my_ch_label.write(k[:-2])

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

            value_of_symbol = all_symbols[i]

            w_of_next_ch = ((x_down_right - x_down_left)**2+(y_down_right - y_down_left)**2)**1/2

            h_of_next_ch = ((x_top_right - x_down_right)**2+(y_top_right - y_down_right)**2)**1/2

            
            my_ch_label.write(',' + str(x_down_left) + ',' + str(y_down_left) + ',' + str(w_of_next_ch) + ',' + str(h_of_next_ch) + ',' + value_of_symbol)

        my_ch_label.write('\n')

    db.close()

if __name__=='__main__':
    main('results/dset_kr.h5')