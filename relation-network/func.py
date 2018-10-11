import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import utils
import math

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


def data_folders(data_folder='dir/to/dataset'):
    dataset_name = data_folder.split('/')[-1]
    if dataset_name == 'omniglot_resized':
        character_folders = [os.path.join(data_folder, family, character) \
                    for family in os.listdir(data_folder) \
                    if os.path.isdir(os.path.join(data_folder, family)) \
                    for character in os.listdir(os.path.join(data_folder, family))]
        random.seed(1)
        random.shuffle(character_folders)
        num_train = 1200
        metatrain_character_folders = character_folders[:num_train]
        metaval_character_folders = character_folders[num_train:]

    else:
        character_folders = [os.path.join(data_folder, label) \
                    for label in os.listdir(data_folder) \
                    if os.path.isdir(os.path.join(data_folder, label))]
        random.seed(1)
        random.shuffle(character_folders)
        num_train = len(character_folders)*3//4
        metatrain_character_folders = character_folders[:num_train]
        metaval_character_folders = character_folders[num_train:]
#        train_folder = data_folder + '/train'
#        test_folder = data_folder + '/val'

#        metatrain_character_folders = [os.path.join(train_folder, label) \
#                    for label in os.listdir(train_folder) \
#                    if os.path.isdir(os.path.join(train_folder, label))]
#        metaval_character_folders = [os.path.join(test_folder, label) \
#                    for label in os.listdir(test_folder) \
#                    if os.path.isdir(os.path.join(test_folder, label))]
#        random.seed(1)
#        random.shuffle(metatrain_character_folders)
#        random.shuffle(metaval_character_folders)
    return metatrain_character_folders,metaval_character_folders

def get_data_loader(task, dataset_name, num_per_class=1, split='train',shuffle=True,rotation=0):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    if dataset_name == 'omniglot_resized':
        dataset = utils.Omniglot(task,split=split,transform=transforms.Compose([utils.Rotate(rotation),transforms.ToTensor(),normalize]))
    else:
        dataset = utils.Colordataset(task,split=split,transform=transforms.Compose([utils.Rotate(rotation),transforms.ToTensor(),normalize]))
    if split == 'train':
        sampler = utils.ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = utils.ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
