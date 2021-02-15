import os
folder1 = '/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/dots_word'

f1 = next(os.walk(folder1))
names1 = f1[2]
for i in names1:
    print(i)
    os.remove(os.path.join('/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/dots_word', i))

folder2 = '/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/real_frames'

f2 = next(os.walk(folder2))
names2 = f2[2]
for i in names2:
    print(i)
    os.remove(os.path.join('/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/real_frames', i))

folder3 = '/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/dots_images'

f3 = next(os.walk(folder3))
names3 = f3[2]
for i in names3:
    print(i)
    os.remove(os.path.join('/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/dots_images', i))

folder4 = '/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/images'
f4 = next(os.walk(folder4))
names4 = f4[2]
for i in names4:
    print(i)
    os.remove(os.path.join('/home/sondors/SynthText_ubuntu/results/my_label/ocr_strides/images', i))

#### Detector

folder5 = '/home/sondors/SynthText_ubuntu/results/my_label/ocr_symbols/real_frames'
f5 = next(os.walk(folder5))
names5 = f5[2]
for i in names5:
    print(i)
    os.remove(os.path.join('/home/sondors/SynthText_ubuntu/results/my_label/ocr_symbols/real_frames', i))