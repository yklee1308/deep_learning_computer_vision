# Deep Learning for Classification
## Introduction
This project is a deep-learning framework for image classification made by Youngkeun Lee.

- [Installation](#Installation)
- [Models](#Models)
- [Datasets](#Datasets)
- [Training](#Training)
- [Testing](#Testing)

## Installation
### Conda
1. Clone this repo to your path ${PATH}.
```
cd ${PATH}
git clone https://github.com/yklee1308/deep_learning_computer_vision.git
```

2. Create and activate Conda environment (python 3.8).
```
conda create -n {ENV_NAME} python=3.8
conda activate {ENV_NAME}
```

3. Install PyTorch (>= 1.8) and dependencies.
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
cd ${REPO_PATH}/classification
pip install -r requirements.txt
```

## Models
The classification models supported by this project are as follows:
- [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
- [GoogleNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

## Datasets
The classification datasets supported by this project are as follows:
- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [ImageNet](https://www.image-net.org/download.php)

### Dataset Format
All datasets should be converted to the following format:
```
{DATASET_NAME}
    |————{DATASET_NAME}_img_train
    |        |————CLASS_1
    |        |        |————image_1.jpg
    |        |        |————image_2.jpg
    |        |        |       ⋮ 
    |        |        └————image_n.jpg
    |        |
    |        |————CLASS_2
    |        |        ⋮
    |        └————CLASS_N
    |
    └————{DATASET_NAME}_img_test
    |        |————CLASS_1
    |        |        |————image_0.jpg
    |        |        |————image_1.jpg
    |        |        |       ⋮ 
    |        |        └————image_m.jpg
    |        |
    |        |————CLASS_2
    |        |        ⋮
    |        └————CLASS_N
    |
    └————{DATASET_NAME}_classes.txt
```
The names of the class folders, image files and file extensions can be changed. However, the names of the folders and file with {DATASET_NAME} should follow the given format.