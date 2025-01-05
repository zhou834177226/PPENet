# Part in Part Embedding Network for Zero-Shot Learning [ICASSP 2025]
The current project page provides [pytorch](http://pytorch.org/) code that implements the following paper:   
**Title:**      "Part in Part Embedding Network for Zero-Shot Learning"

**URL:**   

## Citation




# Overview of PPEN

<img width="1870" alt="PPEN" src="https://github.com/user-attachments/assets/97be9720-dd42-44fe-911e-77ba3cb0400f" />


## How to Run

The model uses the CUB, SUN, and AWA2 datasets.

[CUB Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

[SUN Dataset](https://vision.princeton.edu/projects/2010/SUN/)

[AWA2 Dataset](https://cvml.ista.ac.at/AwA2/)

The pretrained parameters, including the ResNet pretrained weights and the outputs of the GloVe model, have been included in the project.

```train
cd /GEMZSL/tools
python train.py
```


## Training Settings

The files are stored in this directory: config/GEMZSL


If you use our work in your research please cite us:

