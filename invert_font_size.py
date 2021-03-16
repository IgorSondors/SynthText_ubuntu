# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np 
import matplotlib.pyplot as plt 
#import cPickle as cp
import pickle as cp

pygame.init()


ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]
#print(type(ys))
#print(A)
xs = []
models = {} #linear model

FS = FontState()
plt.figure()
#print('FS.fonts', FS.fonts)
#print('FS.fonts len', len(FS.fonts))
for i in range(len(FS.fonts)):#xrange(len(FS.fonts)):
	print(FS.fonts[i])
	font = freetype.Font(FS.fonts[i], size=12)
	h = []
	for y in ys:
		#print(ys)
		#print('y is', y)
		#print(font)
		#print('font is', font)
		#print('font.get_sized_glyph_height(float(y))', font.get_sized_glyph_height(float(y)))
		h.append(font.get_sized_glyph_height(float(y)))
		#print('h is', h)
	print(h, len(h))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	xs.append(h)
	#print(m)

print(models)

with open('font_px2pt.cp','wb') as f:
	cp.dump(models,f)
#plt.plot(xs,ys[i])
#plt.show()
