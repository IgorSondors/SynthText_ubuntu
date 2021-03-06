#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *
import cv2



def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    H,W = text_im.shape[:2]
    print("image shape", (H, W))
    
    # plot the character-BB:
    #print(type(charBB_list))
    #print(charBB_list)
    #print(len(charBB_list[0][0]))
    #print((charBB_list[0][0]))
    for i in range(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)
            """vcol = ['r','g','b','k']
            for j in range(4):
                plt.scatter(bb[0,j],bb[1,j],color=vcol[j])"""
    
    # plot the word-BB:
    #print('wordBB', wordBB)
    #print('wordBB len', len(wordBB))
    #print('wordBB.shape[-1]', wordBB.shape)
    '''for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        """vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])"""'''

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    counter = -1
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']
        
        print("db['data'][k]", db['data'][k])
        print("db['data'][k]", db['data'][k][0])
        #print('wordBB', wordBB)
        #print('wordBB len', len(wordBB))
        #print('wordBB.shape[-1]', wordBB.shape)
        #print('wordBB[0]', wordBB[0])
        #print('wordBB[1]', wordBB[1])

        """print('charBB', charBB)
        print('charBB len', len(charBB))
        print('charBB.shape[-1]', charBB.shape)
        print('charBB[0]', charBB[0])
        print('charBB[1]', charBB[1])"""
        counter = counter + 1
        
        txt_utf = []
        for i in txt:
            txt_utf.append(i)#.decode("utf-8"))

        viz_textbb(rgb, [charBB], wordBB)
        #cv2.imwrite('results/120_thresh/{}.png'.format(k), rgb)
        #plt.savefig('results/130_thresh/{}.png'.format(k), dpi= 'figure')
        #plt.savefig('results/hz/{}.png'.format(k), dpi= 'figure')

        print ("image name        : ", colorize(Color.RED, k, bold=True), 'Image_{}'.format(counter))
        #print ("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        #print ("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print ("  ** text         : ", colorize(Color.GREEN, txt_utf))


        if 'q' in input("next? ('q' to exit) : "):
            break
    db.close()

if __name__=='__main__':
    main('/home/sondors/SynthText_ubuntu/results/test_data.h5')
