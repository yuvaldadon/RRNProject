# Recurrent Refinement Networks Project

This repo is the Pytorch implementation for Tel-Aviv University Deep Learning Project [Recurrent Refinement Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Referring_Image_Segmentation_CVPR_2018_paper.html).  
The project handles the task of Refering Image Segmentation on the [VGPhraseCut](https://people.cs.umass.edu/~chenyun/publication/phrasecut/) databse.  
Based on the original paper [Referring Image Segmentation via Recurrent Refinement
Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Referring_Image_Segmentation_CVPR_2018_paper.html),
CVPR 2018. 
<p align="center">
  <img src="https://i.ibb.co/bb6MckQ/Untitled.png" width="55%"/>
</p>

## Setup

1. Clone this repository
```
git clone https://github.com/yuvaldadon/RRNProject
```
2. Setup the [VGPhraseCut Dataset](https://github.com/ChenyunWu/PhraseCutDataset), so that images are under:
```
RRNProject/PhraseCutDataset/data/VGPhraseCut_v0/images
```
3. Download the [Pre-Trained Deeplab](https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-vocaug-20000.pth) on Pascal-VOC in:
```
RRNProject/tools/deeplabv2_resnet101_msc-vocaug-20000.pth
```

## Usage

Use the model via the notebooks, these contain many configurations in the top section.

1. Train the network using [train.ipynb](train.ipynb) 

General training configurations:
```
- path = '../RRNProject/'
- output_path = '../RRNProject/output/'
- opt.train_iter = 200000      # max iteration to train
- opt.train_log_every = 300    # num of iterations to log training info
- opt.checkpoint_every = 25000 # num of iterations to save checkpoint
- opt.load_checkpoint = None   # path to .pth to continue training
- opt.checkpoint = '../RRNProject/output/checkpoint.pth' # path to .pth to continue training
```

2. Test the network using [test.ipynb](test.ipynb)  

General testing configurations:
```
opt.test_log_every = 50      # num of iterations to log test info
opt.save_im_every = 20000    # num of iterations to save mask segmentation output
opt.checkpoint = '../RRNProject/output/checkpoint.pth'   # path to .pth to load model
```

## References
- [Referring Image Segmentation via Recurrent Refinement
Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Referring_Image_Segmentation_CVPR_2018_paper.html),
CVPR 2018. 
- [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)
