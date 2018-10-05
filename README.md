# Application of few shot learning

This is the code repository of my master thesis, the goal of the thesis is going to use the techniques of few shot learning to solve the problem where we can use the advantage of deep learning even if we only have less data.

## Things to do

* create generate_dataset.py: For each image sets, you can generate a torch dataset for training and testing.
* traing.py, test.py
* 
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
├── own_dataset/
	├── images_1_label_1.jpg  
	├── images_2_label_1.jpg 
	...
  └── images_k_label_n.jpg

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
