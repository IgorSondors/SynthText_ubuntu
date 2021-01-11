import h5py 

db = h5py.File('results/SynthText.h5', 'r')
#db = h5py.File('data/dset.h5', 'r')
keys = db.keys()
print(list(keys))
for key in keys:
    print(db[key].values())