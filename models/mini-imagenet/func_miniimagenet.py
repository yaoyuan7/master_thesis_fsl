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

class miniimagenet(Dataset):
    def __init__(self, setname):
        ROOT_PATH = '../../sharedLocal/miniimagenet/'
        csv_path = os.path.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor()])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class dataset_sampler():
    def __init__(self, label, n_batch, n_cls, n_per,shuffle=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pn = nn.Sequential(conv_block(3, 64),conv_block(64, 64),conv_block(64, 64),conv_block(64, 64))
        self.fc1 = nn.Linear(64*5*5, 1600)
        self.fc21 = nn.Linear(1600,1600)
        self.fc22 = nn.Linear(1600,1600)
        self.fc3 = nn.Linear(1600, 1600)
        self.fc4 = nn.Linear(1600, 64*5*5)
        self.relu = nn.ReLU()
    def encode(self, x):
        label = x[1]
        x_ = x[0]
        if label == 'support':
            emb_x = self.pn(x_)
            emb_x = emb_x.view(emb_x.shape[0],-1)
            way=int(emb_x.shape[0]/5)
            emb_x = emb_x.reshape(5,way,-1).mean(dim=0)
            emb_x = emb_x.view(emb_x.shape[0],-1)
        elif label == 'query':
            emb_x = self.pn(x_)
            emb_x = emb_x.view(emb_x.shape[0],-1)
        #h1 = self.relu(self.fc1(emb_x))
        h1 = self.relu(emb_x)
        return emb_x,self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    def decode(self, z):
        #h3 = F.relu(self.fc3(z))
        h3 = F.relu(z)
        return torch.sigmoid(self.fc4(h3)).view(-1,64*5*5)
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
    def forward(self,embedding_support, recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels,args):
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
        return args.weight*KLD + CE, logits
