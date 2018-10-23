# Application of few shot learning

This is the code repository of my master thesis, the goal of the thesis is going to use the techniques of few shot learning to solve the problem where we can use the advantage of deep learning even if we only have less data.

# Motivation
* The general motivation of this project is that I want to apply the techniques of unsupervised or semi-supervised learning on the recent developing meta-learning problem, few-shot learning. 
* The baseline model I will use is prototypical network, which learns a shallow CNN to extract such features that can largely distinguish the inter-differences of testing set. Prototypical network doesn't learn all the features, but it learns what are the most important features for classification. That is so-called Meta Learning.
* Meta learning model doesn't performs well when it encounters these problems: 1. Dataset is not large enough, which means the model cannot learn as much as possible if there are only few data, and one of the solutions is to use VAE or GAN to generate more data. 2. There are many unlabelled data, which means the model cannot use unlabelled data for training, and one of the solutions is to use unsupervised algorithms to cluster the unlabelled data first and then use the clutered data for training.
* Now (Oct 16th), I am working on the first problem.
## Things to do

### Prerequisites

What things you need to install the software and how to install them

```
python3
```

### Installing

A step by step series of examples that tell you how to get a development env running
```
until finished
```
### Structure of dataset

If you want to use your own dataset to train and test the model, you should make it look like:
```
data/
model/
├── generate_dataset.py
├── train.py
├── test.py
├── util.py
└── Readme.md

result/
```

## Train the model

First of all, train the model using different dataset with different parameters:
```
python train.py -d [DATASET] -n [#WAY] -s [#SHOT]
```
NOTICE: You can choose DATASET as omniglot, miniimagenet, and the name of the folder you create above.
#WAY is the number of classes used for training, #SHOT is the number of training data per iteration.

## Test the model

Test the model with a new image:
```
python test.py -f ./directory/to/image.jpg
```
or choose a number of random images from training set:
```
python test.py -f [#IMAGE]
```

## Authors

* **Yuan Yao** - *Initial work* - 

## Acknowledgments
* 
