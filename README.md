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

For an explanation of the fields in `dset.h5` (e.g.: `seg`,`area`,`label`), please check this [comment](https://github.com/ankush-me/SynthText/issues/5#issuecomment-274490044).


### Pre-generated Dataset
A dataset with approximately 800000 synthetic scene-text images generated with this code can be found [here](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Pre-processed Background Images
The 8,000 background images used in the paper, along with their segmentation and depth masks, have been uploaded here:
`http://thor.robots.ox.ac.uk/~vgg/data/scenetext/preproc//<filename>`, where, `<filename>` can be:

|    filenames    | size |                      description                     |
|:--------------- | ----:|:---------------------------------------------------- |
| `imnames.cp`    | 180K | names of images which do not contain background text | 
| `bg_img.tar.gz` | 8.9G | images (filter these using `imnames.cp`)             |
| `depth.h5`      |  15G | depth maps                                           | 
| `seg.h5`        | 6.9G | segmentation maps                                    |


Note: due to large size, `depth.h5` is also available for download as 3-part split-files of 5G each.
These part files are named: `depth.h5-00, depth.h5-01, depth.h5-02`. Download using the path above, and put them together using `cat depth.h5-0* > depth.h5`.

### TO DO
- [ ] Генерировать карты глубины картинок произвольного размера
- [ ] Добавлять черную рамку на картинку после генерации текста, пересчитывать в соответствии с этим координаты всех символов
- [ ] Задавать границы размера символов, в пределах которых будет генерироваться текст
- [ ] Менять плотность текста
- [ ] Вращать картинки и пересчитывать координаты символов в соответствии с этим (функция есть)

## Output samples for Russian text

**Synthetic Rus Text Samples 1**

![example 1](https://github.com/IgorSondors/SynthText_ubuntu/blob/main/results/example1.jpg)

**Synthetic Rus Text Samples 2**

![example 2](https://github.com/IgorSondors/SynthText_ubuntu/blob/main/results/example2.jpg)

