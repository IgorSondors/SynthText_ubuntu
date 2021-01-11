## How to use this source

### Preparation

Put your text data and font as follow

```
data
├── dset.h5
├── fonts
│   ├── fontlist.txt                        : your font list
│   ├── ubuntu
│   ├── ubuntucondensed
│   ├── Below150pix                         : your font
│   └── ubuntumono
├── models
│   ├── char_freq.cp
│   ├── colors_new.cp
│   └── font_px2pt.cp
└── newsgroup
    └── newsgroup.txt                       : your text source
```

### Install pip dependencies

```
jsonpickle==1.4.1
matplotlib==3.3.3
Pillow==8.0.1
pygame==1.9.3
wrapt==1.12.1
opencv-python==4.4.0.46
h5py==2.10.0
numpy==1.19.2
scipy==1.5.4
```

### Generate font model and char model
```
python invert_font_size.py
python char_freq.py

mv char_freq.cp data/models/
mv font_px2pt.cp data/models/
```
### Check the structure of a .h5 file

```
python check_structure.py
```
### Generating samples

```
python gen.py
```
### Visualize

```
python visualize_results.py
```

# SynthText
Code for generating synthetic text images as described in ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Adding New Images
Segmentation and depth-maps are required to use new images as background. Sample scripts for obtaining these are available [here](https://github.com/ankush-me/SynthText/tree/master/prep_scripts).

* `predict_depth.m` MATLAB script to regress a depth mask for a given RGB image; uses the network of [Liu etal.](https://bitbucket.org/fayao/dcnf-fcsp/) However, more recent works (e.g., [this](https://github.com/iro-cp/FCRN-DepthPrediction)) might give better results.
* `run_ucm.m` and `floodFill.py` for getting segmentation masks using [gPb-UCM](https://github.com/jponttuset/mcg).

For an explanation of the fields in `dset.h5` (e.g.: `seg`,`area`,`label`), please check this [comment](https://github.com/ankush-me/SynthText/issues/5#issuecomment-274490044).


### Pre-generated Dataset
A dataset with approximately 800000 synthetic scene-text images generated with this code can be found [here](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Pre-processed Background Images
The 8,000 background images used in the paper, along with their segmentation and depth masks, have been uploaded here:
`http://zeus.robots.ox.ac.uk/textspot/static/db/<filename>`, where, `<filename>` can be:

- `imnames.cp` [180K]: names of filtered files, i.e., those files which do not contain text
- `bg_img.tar.gz` [8.9G]: compressed image files (more than 8000, so only use the filtered ones in imnames.cp)
- `depth.h5` [15G]: depth maps
- `seg.h5` [6.9G]: segmentation maps

### TO DO
- [] Генерировать карты глубины картинок произвольного размера
[] Добавлять черную рамку на картинку после генерации текста, пересчитывать в соответствии с этим координаты всех символов
[] Задавать границы размера символов, в пределах которых будет генерироваться текст
[] Менять плотность текста
[] Вращать картинки и пересчитывать координаты символов в соответствии с этим (функция есть)
## Output samples for Russian text

**Synthetic Rus Text Samples 1**

![example 1](https://github.com/IgorSondors/SynthText_ubuntu/blob/main/results/example1.jpg)


**Synthetic Rus Text Samples 2**

![example 2](https://github.com/IgorSondors/SynthText_ubuntu/blob/main/results/example2.jpg)
