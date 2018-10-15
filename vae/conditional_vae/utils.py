import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

def create_dateset(train_images,train_labels):
    img,lab = [[] for _ in range(10)],[[] for _ in range(10)]
    for i in range(len(train_images)):
        index = train_labels[i].tolist().index(1)
        img[index].append(train_images[i])
        lab[index].append(train_labels[i])
    for i in range(len(img)):
        img[i] = np.asarray(img[i])
        lab[i] = np.asarray(lab[i])
    return img,lab

def select_random(img,lab,shot=5,way=5):
    training_set, training_label = [], []
    select_class = random.sample(range(0,len(img)),way)
    for i in select_class:
        index = random.sample(range(1, len(img[i])),shot)
        for j in index:
            training_set.append(img[i][j])
            training_label.append(lab[i][j])
    return np.asarray(training_set), np.asarray(training_label)
