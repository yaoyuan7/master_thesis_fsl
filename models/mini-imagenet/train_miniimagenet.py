import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from func_miniimagenet import VAE, Loss, dataset_sampler,miniimagenet
#from func_omniglot import VAE, Loss, data_folders,get_data_loader, create_task
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import argparse

def count_acc(logits,labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.FloatTensor).mean().item()

parser = argparse.ArgumentParser()
parser.add_argument('--weight',type=float,default=1)
parser.add_argument('--gpu',default=0)
args = parser.parse_args()

num_epochs = 250
learning_rate = 0.001
cnt = 0
n_batch_train = 100
n_batch_val = 100

n_train_way = 15
n_val_way = 5#fix

n_train_shot = 5#fix
n_val_shot = 5
# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=True,transform=transforms.ToTensor(),download=True)
#val_dataset = torchvision.datasets.MNIST(root='../../sharedLocal/',train=False,transform=transforms.ToTensor())

train_dataset = miniimagenet('train')
train_sampler = dataset_sampler(train_dataset.label, n_batch_train, n_train_way, n_train_shot + 15)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=train_sampler, pin_memory=True)

val_dataset = miniimagenet('val')
val_sampler = dataset_sampler(val_dataset.label, n_batch_val, n_val_way, n_val_shot + 15)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_sampler=val_sampler, pin_memory=True)

pn_loss_list = []
pn_acc_list = []
val_loss_list = []
val_acc_list = []

torch.cuda.set_device(3)
model = VAE().cuda()
crit = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    print("epoch {}...".format(epoch))
    print('training....')
    for batch_idx, (data, label) in enumerate(train_loader):
        data,label = data.cuda(), label.cuda()
        x_support,x_query = data[:n_train_way*n_train_shot],data[n_train_way*n_train_shot:n_train_way*(n_train_shot+15)]
        optimizer.zero_grad()
        embedding_support, recon_support, mu_support, logvar_support = model([x_support,'support'])
        embedding_query, recon_query, mu_query, logvar_query = model([x_query,'query'])
        labels = torch.arange(n_train_way).repeat(15)
        labels = labels.type(torch.cuda.LongTensor)
        loss, logits = crit(embedding_support,recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels,args)
        accuracy = count_acc(logits, labels)
        total_loss = total_loss + loss.item()
        total_acc += accuracy
        loss.backward()
        optimizer.step()
    pn_loss_list.append(total_loss/batch_idx)
    pn_acc_list.append(total_acc/batch_idx)

    if epoch%5 == 0:
        val_loss = 0
        val_acc = 0
        print('validation....')
        for batch_idx, (data, label) in enumerate(val_loader):
            data,label = data.cuda(), label.cuda()
            x_support,x_query = data[:n_val_way*n_val_shot],data[n_val_way*n_val_shot:n_val_way*(n_val_shot+15)]
            embedding_support, recon_support, mu_support, logvar_support = model([x_support,'support'])
            embedding_query, recon_query, mu_query, logvar_query = model([x_query,'query'])
            labels = torch.arange(n_val_way).repeat(15)
            labels = labels.type(torch.cuda.LongTensor)
            loss, logits = crit(embedding_support,recon_support, mu_support, logvar_support, x_support, embedding_query,recon_query, mu_query, logvar_query, x_query, labels,args)
            accuracy = count_acc(logits, labels)
            #val_loss = val_loss + loss.item()
            #val_acc += accuracy
            val_loss_list.append(loss.item())
            val_acc_list.append(accuracy)

        print(pn_acc_list[-1],val_acc_list[-1])
        print(pn_loss_list[-1],val_loss_list[-1])

with open('{}_pn_loss_list.txt'.format(args.weight),'w') as f:
    for item in pn_loss_list:
        f.write('%s\n'%item)
with open('{}_pn_acc_list.txt'.format(args.weight),'w') as f:
    for item in pn_acc_list:
        f.write('%s\n'%item)
with open('{}_val_loss_list.txt'.format(args.weight),'w') as f:
    for item in val_loss_list:
        f.write('%s\n'%item)
with open('{}_val_acc_list.txt'.format(args.weight),'w') as f:
    for item in val_acc_list:
        f.write('%s\n'%item)
