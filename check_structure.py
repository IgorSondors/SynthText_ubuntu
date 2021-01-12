import h5py 
import numpy as np
from PIL import Image

db = h5py.File('results/SynthText.h5', 'r')
#db = h5py.File('data/dset.h5', 'r')
keys = db.keys()
print(list(keys))
for key in keys:
    print(db[key].values())

imnames_sorted = sorted(db['data'].keys())
print(imnames_sorted)
imname = imnames_sorted[0]

print(type(db['data'][imname]))
print(type(db['data'][imname][:]))
img = Image.fromarray((db['data'][imname])[:])
print(img)
L = db['data'][imname].attrs['txt']
for i in L:

    print(i.decode("utf-8"))

"""imnames = db['image'].keys()
imnames_sorted = sorted(db['image'].keys())
imname = imnames_sorted[4]

#print(type(imnames),imnames)
#print(type(imnames_sorted),imnames_sorted)

#print(type(db['image'][imname]))
#print(type(db['image'][imname][:]))
img = Image.fromarray((db['image'][imname])[:])

depth = db['depth'][imname][:].T
depth = depth[:,:,1]
print(len(depth))
sz = depth.shape[:2][::-1]
print(sz)

seg = db['seg'][imname][:].astype('float32')
area = db['seg'][imname].attrs['area']
label = db['seg'][imname].attrs['label']

#print(type(seg), 'seg', seg)

#print(type(area), 'seg area', area)
#print(type(label), 'seg label', label)

img1 = img.resize(sz,Image.ANTIALIAS)
print(type(img), img, img.size)
print(type(img1), img1, img1.size)
img = np.array(img.resize(sz,Image.ANTIALIAS))
seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

#print(type(seg), 'seg', seg)"""