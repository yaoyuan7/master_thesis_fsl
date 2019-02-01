import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
import torchvision.datasets as dset

import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import random
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def data_folders(data_folder='dir/to/dataset'):
    dataset_name = data_folder.split('/')[-1]
    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)
    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders

def get_data_loader(task, dataset_name, num_per_class=1, split='train',shuffle=True,rotation=0):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader


class create_task(object):
    def __init__(self, character_folders, num_classes, train_num,test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("You need to specify the dataset")

class Omniglot(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pn = nn.Sequential(conv_block(1, 64),conv_block(64, 64),conv_block(64, 64),conv_block(64, 64))
        self.fc1 = nn.Linear(64*1*1, 256)
        self.fc21 = nn.Linear(256, 64)
        self.fc22 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 64*1*1)
        self.relu = nn.ReLU()
    def encode(self, x):
        label = x[1]
        x_ = x[0]
        if label == 'support':
            emb_x = self.pn(x_)
            emb_x = emb_x.view(emb_x.shape[0],-1)
            if emb_x.shape[0] == 300:
                emb_x = emb_x.reshape(5, int(emb_x.shape[0]/5), -1).mean(dim=0)
                emb_x = emb_x.view(emb_x.shape[0],-1)
            elif emb_x.shape[0] == 100:
                emb_x = emb_x.reshape(5, int(emb_x.shape[0]/5), -1).mean(dim=0)
                emb_x = emb_x.view(emb_x.shape[0],-1)
        elif label == 'query':
            emb_x = self.pn(x_)
            emb_x = emb_x.view(emb_x.shape[0],-1)
        h1 = self.relu(self.fc1(emb_x))
        return emb_x,self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1,64*1*1)
    def forward(self, x):
        emb_x, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return emb_x, self.decode(z), mu, logvar


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(size_average=False)
        self.rec_loss = nn.BCELoss(size_average=False)
    def calculate_dist(self, a, b):
        #a: query(多的) b:support(少的)
        #way = int(b.shape[0]/5)
        #shot = 5
        #b = b.reshape(shot, way, -1).mean(dim=0)
        a = a.unsqueeze(1).expand(a.shape[0], b.shape[0], -1)
        b = b.unsqueeze(0).expand(a.shape[0], b.shape[0], -1)
        logits = -torch.pow(a-b,2).sum(2)
        return logits
    def forward(self,embedding_support, recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels):
        MSE_support = self.mse_loss(recon_support, embedding_support)/x_support.shape[0]
        MSE_query = self.mse_loss(recon_query, embedding_query)/x_query.shape[0]
        MSE = MSE_support + MSE_query
        logits = self.calculate_dist(mu_query,mu_support)
        CE = torch.nn.functional.cross_entropy(logits,labels)
        #mu_support = mu_support.reshape(5,5,-1).mean(dim=0)
        #logvar_support = logvar_support.reshape(5,5,-1).mean(dim=0)
        KLD_support = -0.5 * torch.sum(1 + logvar_support - mu_support.pow(2) - logvar_support.exp())
        KLD_query = -0.5 * torch.sum(1 + logvar_query - mu_query.pow(2) - logvar_query.exp())
        #KLD = KLD_support + KLD_query
        KLD  = KLD_support/mu_support.shape[0] + KLD_query/mu_query.shape[0]
        #print(MSE,CE,KLD)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        return 0.1*KLD + CE, logits
